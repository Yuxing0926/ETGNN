from __future__ import division
from __future__ import unicode_literals

import numpy as np
import pandas as pd
import sympy as sym
import torch
from math import sqrt, pi as PI

from torch.nn import Linear, Embedding, Module, ModuleList, Parameter

from torch_scatter import scatter
from torch_geometric.nn.acts import swish
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.nn.models.dimenet_utils import bessel_basis, real_sph_harm


class Envelope(Module):
    def __init__(self, exponent):
        super(Envelope, self).__init__()
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x):
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        return 1. / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p2

class BesselBasisLayer(Module):
    def __init__(self, num_radial, cutoff=5.0, envelope_exponent=5):
        super(BesselBasisLayer, self).__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        # Initialize frequencies at canonical positions
        self.freq = Parameter(torch.Tensor(num_radial))

        self.reset_parameters()

    def reset_parameters(self):
        torch.arange(1, self.freq.numel() + 1, out=self.freq).mul_(PI)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) / self.cutoff
        return self.envelope(dist) * (self.freq * dist).sin()

class SphericalBasisLayer(Module):
    def __init__(self, num_spherical, num_radial, cutoff=5.0,
                 cutoff_func:callable=Envelope(5)):
        super(SphericalBasisLayer, self).__init__()
        assert num_radial <= 64
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.cutoff_func = cutoff_func

        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = real_sph_harm(num_spherical)
        self.sph_funcs = []
        self.bessel_funcs = []

        x, theta = sym.symbols('x theta')
        modules = {'sin': torch.sin, 'cos': torch.cos}
        for i in range(num_spherical):
            if i == 0:
                sph1 = sym.lambdify([theta], sph_harm_forms[i][0], modules)(0)
                self.sph_funcs.append(lambda x: torch.zeros_like(x) + sph1)
            else:
                sph = sym.lambdify([theta], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(sph)
            for j in range(num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    def forward(self, dist, angle, idx_kj):
        dist = dist / self.cutoff
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        rbf = self.cutoff_func(dist).unsqueeze(-1) * rbf

        cbf = torch.stack([f(angle) for f in self.sph_funcs], dim=1)

        n, k = self.num_spherical, self.num_radial
        out = (rbf[idx_kj].view(-1, n, k) * cbf.view(-1, n, 1)).view(-1, n * k)
        return out

class Embedder(Module):
    def __init__(self,
                 d_model,
                 compute_device=None):
        super().__init__()
        self.d_model = d_model
        self.compute_device = torch.device('cuda')

        mat2vec = './mat2vec.csv'  # element embedding

        cbfv = pd.read_csv(mat2vec, index_col=0).values
        feat_size = cbfv.shape[-1]
        self.fc_mat2vec = Linear(feat_size,
                                 self.d_model).to(self.compute_device)
        zeros = np.zeros((1, feat_size))
        cat_array = np.concatenate([zeros, cbfv])
        cat_array = torch.as_tensor(cat_array, dtype=torch.float)
        self.cbfv = Embedding.from_pretrained(cat_array) \
            .to(self.compute_device, dtype=torch.float)

    def forward(self, src, i):
        mat2vec_emb = self.cbfv(src)
        x_emb = self.fc_mat2vec(mat2vec_emb[i])
        return x_emb

class EmbeddingBlock(Module):
    def __init__(self, num_radial, hidden_channels, act=swish):
        super(EmbeddingBlock, self).__init__()
        self.act = act

        # Atom embeddings: 118 elements. Use 119 for 0-based indexing.
        self.emb = Embedding(119, hidden_channels)
        self.lin_rbf = Linear(num_radial, hidden_channels)
        self.lin = Linear(3 * hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))
        self.lin_rbf.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, rbf, i, j):
        x = self.emb(x)
        rbf = self.act(self.lin_rbf(rbf))
        return self.act(self.lin(torch.cat([x[i], x[j], rbf], dim=-1))), x

class ResidualLayer(Module):
    def __init__(self, hidden_channels, act=swish):
        super(ResidualLayer, self).__init__()
        self.act = act
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin1.weight, scale=2.0)
        self.lin1.bias.data.fill_(0)
        glorot_orthogonal(self.lin2.weight, scale=2.0)
        self.lin2.bias.data.fill_(0)

    def forward(self, x):
        return x + self.act(self.lin2(self.act(self.lin1(x))))

class OutputBlock(Module):
    def __init__(self, num_radial, hidden_channels, out_emb_size, act=swish):
        super(OutputBlock, self).__init__()
        self.act = act

        self.lin_rbf = Linear(num_radial, hidden_channels, bias=False)

        self.up_projection = Linear(hidden_channels, out_emb_size, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        glorot_orthogonal(self.up_projection.weight, scale=2.0)

    def forward(self, x, rbf, i, num_nodes=None):
        x = self.lin_rbf(rbf) * x
        node_attr = scatter(x, i, dim=0, dim_size=num_nodes)
        node_attr = self.up_projection(node_attr)

        return node_attr # node_attr: (Nnodes, out_emb_size)

class InteractionBlock(Module):
    def __init__(self, hidden_channels, int_emb_size, basis_emb_size,
                 num_spherical, num_radial, num_before_skip,
                 num_after_skip, act=swish):
        super(InteractionBlock, self).__init__()
        self.act = act

        # Transformations of Bessel and spherical basis representations
        self.lin_rbf1 = Linear(num_radial, basis_emb_size, bias=False)
        self.lin_rbf2 = Linear(basis_emb_size, hidden_channels, bias=False)

        self.lin_sbf1 = Linear(num_radial * num_spherical, basis_emb_size,
                               bias=False)
        self.lin_sbf2 = Linear(basis_emb_size, int_emb_size,
                               bias=False)

        # Dense transformations of input messages.
        self.lin_kj = Linear(hidden_channels, hidden_channels)
        self.lin_ji = Linear(hidden_channels, hidden_channels)

        # Embedding projections for interaction triplets
        self.down_projection = Linear(hidden_channels, int_emb_size,
                                      bias=False)
        self.up_projection = Linear(int_emb_size, hidden_channels, bias=False)

        # Residual layers before skip connection
        self.layers_before_skip = ModuleList([
            ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)
        ])
        self.lin = Linear(hidden_channels, hidden_channels)  # last before skip

        # Residual layers after skip connection
        self.layers_after_skip = ModuleList([
            ResidualLayer(hidden_channels, act) for _ in range(num_after_skip)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf2.weight, scale=2.0)

        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)

        glorot_orthogonal(self.down_projection.weight, scale=2.0)
        glorot_orthogonal(self.up_projection.weight, scale=2.0)

        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

    def forward(self, x, rbf, sbf, idx_kj, idx_ji, i):
        # Initial transformation
        x_ji = self.act(self.lin_ji(x))
        x_kj = self.act(self.lin_kj(x))

        # Transform via Bessel basis
        rbf = self.lin_rbf1(rbf)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf

        # Down-project embeddings and generate interaction triplet embeddings
        x_kj = self.act(self.down_projection(x_kj))

        # Transform via 2D spherical basis
        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf

        # Aggregate interactions and up-project embeddings
        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x.size(0))
        x_kj = self.up_projection(x_kj)

        # Transformations before skip connection
        h = x_ji + x_kj
        for layer in self.layers_before_skip:
            h = layer(h)
        h = self.act(self.lin(h)) + x

        # Transformations after skip connection
        for layer in self.layers_after_skip:
            h = layer(h)
        return h

class InteractionBlock_triplet(Module):
    def __init__(self, hidden_channels, int_emb_size, basis_emb_size,
                 num_spherical, num_radial, num_before_skip,
                 num_after_skip, num_triplet_features, num_residual_triplet, act=swish):
        super(InteractionBlock_triplet, self).__init__()
        self.act = act

        # Transformations of Bessel and spherical basis representations
        self.lin_rbf1 = Linear(num_radial, basis_emb_size, bias=False)
        self.lin_rbf2 = Linear(basis_emb_size, hidden_channels, bias=False)

        self.lin_sbf1 = Linear(num_radial * num_spherical, basis_emb_size,
                               bias=False)
        self.lin_sbf2 = Linear(basis_emb_size, int_emb_size,
                               bias=False)

        # Dense transformations of input messages.
        self.lin_kj = Linear(hidden_channels, hidden_channels)
        self.lin_ji = Linear(hidden_channels, hidden_channels)

        # Embedding projections for interaction triplets
        self.down_projection = Linear(hidden_channels, int_emb_size,
                                      bias=False)
        self.up_projection = Linear(int_emb_size, hidden_channels, bias=False)

        # Residual layers before skip connection
        self.layers_before_skip = ModuleList([
            ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)
        ])
        self.lin = Linear(hidden_channels, hidden_channels)  # last before skip

        self.lin_h = Linear(hidden_channels, num_triplet_features, bias=False)
        # Residual layers after skip connection
        self.layers_after_skip = ModuleList([
            ResidualLayer(num_triplet_features, act) for _ in range(num_after_skip)
        ])
        
        self.linear_triplet = Linear(num_triplet_features, num_triplet_features, bias=False)
        self.lin_rbf3 = Linear(num_radial, basis_emb_size, bias=False)
        self.lin_rbf4 = Linear(basis_emb_size, num_triplet_features, bias=False)
        self.residual_triplet = ModuleList([
            ResidualLayer(num_triplet_features, act) for _ in range(num_residual_triplet)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf3.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf4.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_h.weight, scale=2.0)

        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)

        glorot_orthogonal(self.down_projection.weight, scale=2.0)
        glorot_orthogonal(self.up_projection.weight, scale=2.0)

        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.linear_triplet.weight, scale=2.0)
        for res_layer in self.residual_triplet:
            res_layer.reset_parameters()

    def forward(self, x, rbf, sbf, triplet_attr, idx_kj, idx_ji):
        # Initial transformation
        x_ji = self.act(self.lin_ji(x))
        x_kj = self.act(self.lin_kj(x))

        # Transform via Bessel basis
        rbf_copy = rbf
        rbf = self.lin_rbf1(rbf)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf

        # Down-project embeddings and generate interaction triplet embeddings
        x_kj = self.act(self.down_projection(x_kj))

        # Transform via 2D spherical basis
        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf

        # Aggregate interactions and up-project embeddings
        #x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x.size(0))
        x_kj = self.up_projection(x_kj)

        # Transformations before skip connection
        h = x_ji[idx_ji] + x_kj
        for layer in self.layers_before_skip:
            h = layer(h)
        h = self.act(self.lin(h)) + x[idx_ji]

        # Transformations after skip connection
        h = self.lin_h(h)
        for layer in self.layers_after_skip:
            h = layer(h)
        
        rbf_copy = self.lin_rbf3(rbf_copy[idx_ji])
        h = h*self.lin_rbf4(rbf_copy)
        h = self.act(self.linear_triplet(triplet_attr)) + h
        for layer in self.residual_triplet:
            h = layer(h)

        h = triplet_attr + h

        return h # triplet_attr

class EmbeddingBlock_triplet(Module):
    def __init__(self, num_radial, num_spherical, num_node_features,
                    num_triplet_features, act=swish):
        super(EmbeddingBlock_triplet, self).__init__()
        self.act = act
        
        self.lin_rbf = Linear(num_radial, num_triplet_features, bias=False)
        self.lin_sbf = Linear(num_radial*num_spherical, num_triplet_features, bias=False)
        self.lin = Linear(3*num_node_features+2*num_triplet_features, num_triplet_features, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_rbf.reset_parameters()
        self.lin.reset_parameters()
        self.lin_sbf.reset_parameters()

    def forward(self, rbf, sbf, node_attr, idx_k, idx_j, idx_i, idx_ji):
        rbf = self.act(self.lin_rbf(rbf))
        sbf = self.act(self.lin_sbf(sbf))
        mkji = torch.cat([node_attr[idx_i], node_attr[idx_j], node_attr[idx_k], rbf[idx_ji], sbf], dim=-1)
        mkji = self.act(self.lin(mkji))
        return mkji





