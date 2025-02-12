"""
/*
 * @Author: Yang Zhong 
 * @Date: 2021-10-12 23:42:11 
 * @Last Modified by: Yang Zhong
 * @Last Modified time: 2021-11-07 19:15:27
 */
 """
import torch
import torch.nn as nn
import numpy as np
import os
from GraphData.data_parsing import cif_parse
from GraphData.graph_data import graph_data_module
from input.config_parsing import read_config
from models.outputs import (Born, Born_node_vec, scalar, trivial_scalar, Force, Force_node_vec, 
                            crystal_tensor, piezoelectric, mu, node_scalar)
import pytorch_lightning as pl
from models.Model import Model
from models.version import soft_logo
from pytorch_lightning.loggers import TensorBoardLogger
from models.ETGNN.ETGNN import ETGNN
from torch.nn import functional as F
import pprint
import warnings
import sys
from models.utils import get_hparam_dict

def prepare_data(config):
    train_ratio = config.dataset_params.train_ratio
    val_ratio = config.dataset_params.val_ratio
    test_ratio = config.dataset_params.test_ratio
    batch_size = config.dataset_params.batch_size
    split_file = config.dataset_params.split_file
    graph_data_path = config.dataset_params.graph_data_path
    if not os.path.exists(graph_data_path):
        os.mkdir(graph_data_path)
    graph_data_path = os.path.join(graph_data_path, 'graph_data.npz')
    if os.path.exists(graph_data_path):
        print(f"Loading graph data from {graph_data_path}!")
    else:
        print(f"building graph data to {graph_data_path}!")
        cif_parse(config)

    graph_data = np.load(graph_data_path, allow_pickle=True)
    graph_data = graph_data['graph'].item()
    graph_dataset = list(graph_data.values())
    
    nbr_counts = []
    for data in graph_dataset:
        nbr_counts += data.nbr_counts.tolist()
    nbr_counts = np.array(nbr_counts)
    mean_num_nbrs = np.mean(nbr_counts)
    print(f'Average number of neighbors for each node: {mean_num_nbrs} !')
    
    graph_dataset = graph_data_module(graph_dataset, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, 
                                        batch_size=batch_size, split_file=split_file)
    graph_dataset.setup(stage=config.setup.stage)

    return graph_dataset

def build_model(config):
    Gnn_net = ETGNN(config)  

    # second order tensor
    if config.setup.property.lower() in ['born', 'dielectric','alpha', 'epsilon', 'polarizability', 'shielding']:
        output_module = crystal_tensor(l_pred_atomwise_tensor=config.setup.l_pred_atomwise_tensor, include_triplet=Gnn_net.export_triplet, num_node_features=Gnn_net.num_node_features, num_edge_features=Gnn_net.hidden_channels, 
                                           num_triplet_features=Gnn_net.num_triplet_features, activation=Gnn_net.act, use_batch_norm=config.ETGNN.use_batch_norm, bias=config.ETGNN.bias, n_h=3, cutoff_triplet=config.ETGNN.cutoff_triplet, l_minus_mean=config.setup.l_minus_mean)

    #Force
    elif config.setup.property.lower() == 'force':
        output_module = Force(num_edge_features=Gnn_net.hidden_channels, activation=Gnn_net.act, use_batch_norm=config.ETGNN.use_batch_norm, bias=config.ETGNN.bias, n_h=3)
    
    #node_scalar
    elif config.setup.property.lower() == 'node_scalar':
        output_module = node_scalar(num_node_features=Gnn_net.num_node_features, activation=Gnn_net.act, use_batch_norm=config.ETGNN.use_batch_norm, bias=config.ETGNN.bias, n_h=3)

    #piezoelectric
    elif config.setup.property.lower() in ['piezoelectric', 'beta']:
        output_module = piezoelectric(include_triplet=Gnn_net.export_triplet, num_node_features=Gnn_net.num_node_features, num_edge_features=Gnn_net.hidden_channels,
                                          num_triplet_features=Gnn_net.num_triplet_features, activation=Gnn_net.act, use_batch_norm=config.ETGNN.use_batch_norm, bias=config.ETGNN.bias, n_h=3, cutoff_triplet=config.ETGNN.cutoff_triplet)
            
    # scalar_per_atom
    elif config.setup.property.lower() == 'scalar_per_atom':
        output_module = scalar('mean', False, num_node_features=Gnn_net.num_node_features, n_h=3, activation=Gnn_net.act)
    
    # scalar
    elif config.setup.property.lower() == 'scalar':
        output_module = scalar('sum', False, num_node_features=Gnn_net.num_node_features, n_h=3, activation=Gnn_net.act)
    
    # mu
    elif config.setup.property.lower() == 'mu':
        output_module = mu(num_edge_features=Gnn_net.hidden_channels, activation=Gnn_net.act, use_batch_norm=config.ETGNN.use_batch_norm, bias=config.ETGNN.bias, n_h=3)
    
    else:
        NotImplementedErrord = Exception('The prediction of this property is not currently supported!')
        raise NotImplementedErrord
    
    return Gnn_net, output_module

def train_and_eval(config):
    data = prepare_data(config)

    graph_representation, output_module = build_model(config)

    # define metrics
    losses = config.losses_metrics.losses
    metrics = config.losses_metrics.metrics

    # Training
    if config.setup.stage == 'fit':
        # laod network weights
        if config.setup.load_from_checkpoint and not config.setup.resume:
            model = Model.load_from_checkpoint(checkpoint_path=config.setup.checkpoint_path,
            representation=graph_representation,
            output=output_module,
            losses=losses,
            validation_metrics=metrics,
            lr=config.optim_params.lr,
            lr_decay=config.optim_params.lr_decay,
            lr_patience=config.optim_params.lr_patience
            )   
        else:            
            model = Model(
            representation=graph_representation,
            output=output_module,
            loss=losses,
            metric=metrics,
            lr=config.optim_params.lr,
            lr_decay=config.optim_params.lr_decay,
            lr_patience=config.optim_params.lr_patience,
            )
        callbacks = [
            pl.callbacks.LearningRateMonitor(),
            pl.callbacks.EarlyStopping(
                monitor="training/total_loss",
                patience=config.optim_params.stop_patience, min_delta=1e-6,
            ),
            pl.callbacks.ModelCheckpoint(
                filename="{epoch}-{val_loss:.6f}",
                save_top_k=1,
                verbose=False,
                monitor='validation/total_loss',
                mode='min',
            )
        ]
        
        tb_logger = TensorBoardLogger(
            save_dir=config.profiler_params.train_dir, name="", default_hp_metric=False)
        
        trainer = pl.Trainer(
            gpus=config.setup.num_gpus,
            callbacks=callbacks,
            progress_bar_refresh_rate=1,
            logger=tb_logger,
            max_epochs=config.optim_params.max_epochs,
            default_root_dir=config.profiler_params.train_dir,
            min_epochs=config.optim_params.min_epochs,
            resume_from_checkpoint = config.setup.checkpoint_path if config.setup.resume else None
        )
        
        print("Start training.")
        trainer.fit(model, data)
        print("Training done.")
        
        # Eval
        print("Start eval.")
        results = trainer.test(model, data)
        # log hyper-parameters in tensorboard.
        hparam_dict = get_hparam_dict(config)
        metric_dict = dict() 
        for result_dict in results:
            metric_dict.update(result_dict)
        trainer.logger.experiment.add_hparams(hparam_dict, metric_dict)
        print("Eval done.")
    
    # Prediction
    if config.setup.stage == 'test':
        model = Model.load_from_checkpoint(checkpoint_path=config.setup.checkpoint_path,
        representation=graph_representation,
        output=output_module,
        loss=losses,
        metric=metrics,
        lr=config.optim_params.lr,
        lr_decay=config.optim_params.lr_decay,
        lr_patience=config.optim_params.lr_patience
        )
        tb_logger = TensorBoardLogger(
            save_dir=config.profiler_params.train_dir, name="", default_hp_metric=False)
        trainer = pl.Trainer(gpus=config.setup.num_gpus, logger=tb_logger)
        trainer.test(model=model, datamodule=data)

if __name__ == '__main__':
    print(soft_logo)
    configure = read_config(config_file_name='config.yaml')
    pprint.pprint(configure)
    if configure.setup.ignore_warnings:
        warnings.filterwarnings('ignore')
    train_and_eval(configure)
