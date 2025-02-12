"""
/*
 * @Author: Yang Zhong 
 * @Date: 2021-10-29 15:03:46 
 * @Last Modified by: Yang Zhong
 * @Last Modified time: 2021-10-29 16:45:04
 */
"""
import yaml
from easydict import EasyDict
from models.utils import get_activation, get_losses, get_metrics
import pprint

"""
Default configuration parameters.
"""
config_default = dict()

"""The parameters for setup"""
config_default_setup = dict()
config_default_setup['l_pred_atomwise_tensor'] = True
config_default_setup['l_pred_crystal_tensor'] = False
config_default_setup['property'] = 'scalar_per_atom'
config_default_setup['num_gpus'] = [1]
config_default_setup['accelerator'] = None # 'dp' 'ddp' 'ddp_cpu'
config_default_setup['precision'] = 32
config_default_setup['stage'] = 'fit'
config_default_setup['resume'] = False
config_default_setup['checkpoint_path'] = './'
config_default_setup['ignore_warnings'] = False
config_default_setup['l_minus_mean'] = False
config_default['setup'] = config_default_setup

"""The parameters for dataset."""
config_default_dataset = dict()
config_default_dataset['crystal_path'] = 'crystals'
config_default_dataset['file_type'] = 'poscar'
config_default_dataset['id_prop_path'] = './'
config_default_dataset['radius'] = 6.0
config_default_dataset['max_num_nbr'] = 32
config_default_dataset['graph_data_path'] = './graph_data'
config_default_dataset['rank_tensor'] = 0
config_default_dataset['train_ratio'] = 0.6
config_default_dataset['val_ratio'] = 0.2
config_default_dataset['test_ratio'] = 0.2
config_default_dataset['batch_size'] = 200
config_default_dataset['split_file'] = None
config_default['dataset_params'] = config_default_dataset

"""The parameters for optimizer."""
config_default_optimizer = dict()
config_default_optimizer['lr'] = 5e-4
config_default_optimizer['lr_decay'] = 0.5
config_default_optimizer['lr_patience'] = 5
config_default_optimizer['stop_patience'] = 30
config_default_optimizer['min_epochs'] = 100
config_default_optimizer['max_epochs'] = 500
config_default['optim_params'] = config_default_optimizer

"""The parameters for losses_metrics."""
config_default_metric = dict()
config_default_metric['losses'] = [{'metric': 'mse', 'loss_weight': 1.0}, {'metric': 'cosine_similarity', 'loss_weight': 0.0}]
config_default_metric['metrics'] = ['mae', 'cosine_similarity']
config_default['losses_metrics'] = config_default_metric

"""The parameters for profiler"""
config_default_profiler = dict()
config_default_profiler['train_dir'] = 'train_data'
config_default_profiler['progress_bar_refresh_rat'] = 1
config_default['profiler_params'] = config_default_profiler

"""The parameters for ETGNN"""
config_default_ETGNN = dict()
config_default_ETGNN['hidden_channels'] = 128
config_default_ETGNN['num_blocks'] = 4
config_default_ETGNN['int_emb_size'] = 64
config_default_ETGNN['basis_emb_size'] = 8
config_default_ETGNN['num_spherical'] = 7
config_default_ETGNN['num_radial'] = 6
config_default_ETGNN['cutoff'] = 6.0
config_default_ETGNN['envelope_exponent'] = 5
config_default_ETGNN['num_before_skip'] = 1
config_default_ETGNN['num_after_skip'] = 2
config_default_ETGNN['activation'] = 'swish'
config_default_ETGNN['num_node_features'] = 64
config_default_ETGNN['num_triplet_features'] = 128
config_default_ETGNN['num_residual_triplet'] = 2
config_default_ETGNN['export_triplet'] = False
config_default_ETGNN['cutoff_func'] = 'envelope'
config_default_ETGNN['rbf_func'] = 'bessel'
config_default_ETGNN['cutoff_triplet'] = 3.0
config_default_ETGNN['use_batch_norm'] = False
config_default_ETGNN['bias'] = True
config_default['ETGNN'] = config_default_ETGNN


def read_config(config_file_name: str = 'config_default.yaml'):
    with open(config_file_name, encoding='utf-8') as rstream:
        data = yaml.load(rstream, yaml.SafeLoader)
    for key in data.keys():
        config_default[key].update(data[key])
    config = EasyDict(config_default)
    config.ETGNN.activation = get_activation(config.ETGNN.activation)
    config.losses_metrics.losses = get_losses(config.losses_metrics.losses)
    config.losses_metrics.metrics = get_metrics(config.losses_metrics.metrics)
    return config
