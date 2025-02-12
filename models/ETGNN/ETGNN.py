from __future__ import division
from __future__ import unicode_literals
import sympy as sym
import torch

from .layers import (BesselBasisLayer, EmbeddingBlock_triplet, InteractionBlock_triplet, SphericalBasisLayer, Embedder,
                            EmbeddingBlock, InteractionBlock, OutputBlock)
from torch_sparse import SparseTensor
from torch_scatter import scatter

from torch_geometric.nn import radius_graph
from torch_geometric.nn.acts import swish
from easydict import EasyDict
from ..layers import GaussianSmearing, cuttoff_envelope, CosineCutoff, BesselBasis

class ETGNN(torch.nn.Module):
    r"""DimeNet++ transforms messages based on the angle between them in a
    rotation-equivariant fashion.

    Args:
        hidden_channels (int): Hidden embedding size.
        out_channels (int): Size of each output sample.
        num_blocks (int): Number of building blocks.
        num_bilinear (int): Size of the bilinear layer tensor.
        num_spherical (int): Number of spherical harmonics.
        num_radial (int): Number of radial basis functions.
        cutoff: (float, optional): Cutoff distance for interatomic
            interactions. (default: :obj:`5.0`)
        envelope_exponent (int, optional): Shape of the smooth cutoff.
            (default: :obj:`5`)
        num_before_skip: (int, optional): Number of residual layers in the
            interaction blocks before the skip connection. (default: :obj:`1`)
        num_after_skip: (int, optional): Number of residual layers in the
            interaction blocks after the skip connection. (default: :obj:`2`)
        num_output_layers: (int, optional): Number of linear layers for the
            output blocks. (default: :obj:`3`)
        act: (function, optional): The activation funtion.
            (default: :obj:`swish`)
    """

    def __init__(self, config):
        super(ETGNN, self).__init__()

        self.hidden_channels = config.ETGNN.hidden_channels
        self.num_blocks = config.ETGNN.num_blocks
        self.int_emb_size = config.ETGNN.int_emb_size
        self.basis_emb_size = config.ETGNN.basis_emb_size
        self.num_spherical = config.ETGNN.num_spherical
        self.num_radial = config.ETGNN.num_radial
        self.cutoff = config.ETGNN.cutoff
        self.envelope_exponent = config.ETGNN.envelope_exponent
        self.num_before_skip = config.ETGNN.num_before_skip
        self.num_after_skip = config.ETGNN.num_after_skip
        self.act = config.ETGNN.activation
        self.num_node_features = config.ETGNN.num_node_features
        self.num_triplet_features = config.ETGNN.num_triplet_features
        self.num_residual_triplet = config.ETGNN.num_residual_triplet
        self.export_triplet = config.ETGNN.export_triplet

        if sym is None:
            raise ImportError('Package `sympy` could not be found.')

        # Cosine basis function expansion layer
        self.cutoff_func = config.ETGNN.cutoff_func
        if 'e' == self.cutoff_func.lower()[0]:  # "envelope func used in Dimnet"
            self.cutoff_func = cuttoff_envelope(cutoff=self.cutoff, exponent=self.envelope_exponent)
        elif 'c' == self.cutoff_func.lower()[0]:  # "cosinecutoff" func
            self.cutoff_func = CosineCutoff(cutoff=self.cutoff)
        else:
            print(f'There is no {self.cutoff_func} cutoff function!')
            exit()
        self.rbf_func = config.ETGNN.rbf_func
        if self.rbf_func.lower() == 'gaussian': 
            self.rbf_func = GaussianSmearing(start=0.0, stop=self.cutoff, num_gaussians=self.num_radial, cutoff_func=self.cutoff_func)
        elif self.rbf_func.lower() == 'bessel':
            self.rbf_func = BesselBasis(cutoff=self.cutoff, n_rbf=self.num_radial, cutoff_func=self.cutoff_func)
        else:
            print(f'There is no {self.rbf_func} rbf function!')
            exit()

        self.sbf = SphericalBasisLayer(self.num_spherical, self.num_radial, self.cutoff,
                                       self.cutoff_func)

        # Embedding and output blocks
        self.emb = EmbeddingBlock(self.num_radial, self.hidden_channels, self.act)
        if self.export_triplet:
            self.emb_tri = EmbeddingBlock_triplet(self.num_radial, self.num_spherical, self.hidden_channels, self.num_triplet_features, self.act)

        #self.scaler = torch.nn.parameter.Parameter(torch.tensor([1.]))
        #self.scaler_tri = torch.nn.parameter.Parameter(torch.tensor([1.]))

        # Interaction blocks
        self.interaction_blocks = torch.nn.ModuleList([
            InteractionBlock(self.hidden_channels,
                             self.int_emb_size,
                             self.basis_emb_size,
                             self.num_spherical,
                             self.num_radial, self.num_before_skip, self.num_after_skip, self.act)
            for _ in range(self.num_blocks)
        ])

        if self.export_triplet:
            self.interaction_blocks_tri = torch.nn.ModuleList([
                InteractionBlock_triplet(self.hidden_channels,
                                self.int_emb_size,
                                self.basis_emb_size,
                                self.num_spherical,
                                self.num_radial, self.num_before_skip, self.num_after_skip, 
                                self.num_triplet_features, self.num_residual_triplet, self.act)
                for _ in range(self.num_blocks)
            ])
        else:
            self.interaction_blocks_tri =[None]*self.num_blocks

        self.output_block = OutputBlock(num_radial=self.num_radial, hidden_channels=self.hidden_channels, out_emb_size=self.num_node_features, act=self.act)

        self.reset_parameters()

    def reset_parameters(self):
        self.emb.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()
        if self.export_triplet:
            for interaction in self.interaction_blocks_tri:
                interaction.reset_parameters()
        self.output_block.reset_parameters()

    def triplets(self, edge_index, num_nodes, cell_shift):
        row, col = edge_index  # j->i

        value = torch.arange(row.size(0), device=row.device)
        adj_t = SparseTensor(
            row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes)
        )
        adj_t_row = adj_t[row]
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

        # Node indices (k->j->i) for triplets.
        idx_i = col.repeat_interleave(num_triplets)
        idx_j = row.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()

        # Edge indices (k-j, j->i) for triplets.
        idx_kj = adj_t_row.storage.value()
        idx_ji = adj_t_row.storage.row()
 
        """
        idx_i -> pos[idx_i]
        idx_j -> pos[idx_j] - nbr_shift[idx_ji]
        idx_k -> pos[idx_k] - nbr_shift[idx_ji] - nbr_shift[idx_kj]
        """
        # Remove i == k triplets with the same cell_shift.
        relative_cell_shift = cell_shift[idx_kj] + cell_shift[idx_ji]
        mask = (idx_i != idx_k) | torch.any(relative_cell_shift != 0, dim=-1)
        idx_i, idx_j, idx_k, idx_kj, idx_ji = idx_i[mask], idx_j[mask], idx_k[mask], idx_kj[mask], idx_ji[mask]

        return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji
    
    def forward(self, data, batch=None):
        z = data.z
        pos = data.pos
        edge_index = data.edge_index
        nbr_shift = data.nbr_shift
        cell_shift = data.cell_shift # shape(Nedges, 3)

        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets(edge_index, z.size(0), cell_shift)

        # Calculate distances.
        dist = (pos[i] - (pos[j] - nbr_shift)).pow(2).sum(dim=-1).sqrt()

        # Calculate angles -- revised version
        pos_i = pos[idx_i]
        pos_j = pos[idx_j] - nbr_shift[idx_ji]
        pos_k = pos[idx_k] - nbr_shift[idx_ji] - nbr_shift[idx_kj] 

        pos_ji = pos_j - pos_i
        pos_kj = pos_k - pos_j

        a = (pos_ji * pos_kj).sum(dim=-1)
        b = torch.cross(pos_ji, pos_kj).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf_func(dist)
        sbf = self.sbf(dist, angle, idx_kj)

        # Embedding block.
        x, node_attr = self.emb(z, rbf, i, j)  # embed atomic numbers

        if self.export_triplet:
            triplet_attr = self.emb_tri(rbf, sbf, node_attr, idx_k, idx_j, idx_i, idx_ji)

        # Interaction blocks.
        for interaction_block, interaction_block_triplet in zip(self.interaction_blocks, self.interaction_blocks_tri):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji, i)
            if self.export_triplet:
                triplet_attr = interaction_block_triplet(x, rbf, sbf, triplet_attr, idx_kj, idx_ji)
        
        node_attr = self.output_block(x, rbf, i)

        graph_representation = EasyDict()
        graph_representation['node_attr'] = node_attr
        graph_representation['edge_attr'] = x #mji
        if self.export_triplet:
            graph_representation['triplet_attr'] = triplet_attr
            graph_representation['triplet_index'] = (idx_i, idx_j, idx_k, idx_kj, idx_ji)
        else:
            graph_representation['triplet_attr'] = None
            graph_representation['triplet_index'] = (idx_i, idx_j, idx_k, idx_kj, idx_ji)

        return graph_representation
