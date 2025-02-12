"""
/*
 * @Author: Yang Zhong 
 * @Date: 2021-11-29 22:13:49 
 * @Last Modified by: Yang Zhong
 * @Last Modified time: 2021-11-29 22:26:42
 */
"""
from torch_sparse import SparseTensor
import torch
import torch.nn as nn
import numpy as np
from torch.nn import (Linear, Bilinear, Sigmoid, Softplus, ELU, ReLU, SELU, SiLU,
                      CELU, BatchNorm1d, ModuleList, Sequential, Tanh, BatchNorm1d as BN)
from torch_geometric.nn.acts import swish
from typing import Callable, Union
import re
import torch.nn.functional as F
import matplotlib.pyplot as plt
from easydict import EasyDict
from scipy.stats import gaussian_kde


def linear_bn_act(in_features: int, out_features: int, lbias: bool = False, activation: Callable = None, use_batch_norm: bool = False):
    if use_batch_norm:
        if activation is None:
            return Sequential(Linear(in_features, out_features, lbias), BN(out_features))
        else:
            return Sequential(Linear(in_features, out_features, lbias), BN(out_features), activation)
    else:
        if activation is None:
            return Linear(in_features, out_features, lbias)
        else:
            return Sequential(Linear(in_features, out_features, lbias), activation)

class SSP(nn.Module):
    r"""Applies element-wise :math:`\text{SSP}(x)=\text{Softplus}(x)-\text{Softplus}(0)`

    Shifted SoftPlus (SSP)

    Args:
        beta: the :math:`\beta` value for the Softplus formulation. Default: 1
        threshold: values above this revert to a linear function. Default: 20

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    """

    def __init__(self, beta=1, threshold=20):
        super(SSP, self).__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, input):
        sp0 = F.softplus(torch.Tensor([0]), self.beta, self.threshold).item()
        return F.softplus(input, self.beta, self.threshold) - sp0

    def extra_repr(self):
        return 'beta={}, threshold={}'.format(self.beta, self.threshold)

class SWISH(nn.Module):
    def __init__(self):
        super(SWISH, self).__init__()

    def forward(self, input):
        return swish(input)

def get_activation(name):
    act_name = name.lower()
    m = re.match(r"(\w+)\((\d+\.\d+)\)", act_name)
    if m is not None:
        act_name, alpha = m.groups()
        alpha = float(alpha)
        print(act_name, alpha)
    else:
        alpha = 1.0
    if act_name == 'softplus':
        return Softplus()
    elif act_name == 'ssp':
        return SSP()
    elif act_name == 'elu':
        return ELU(alpha)
    elif act_name == 'relu':
        return ReLU()
    elif act_name == 'selu':
        return SELU()
    elif act_name == 'swish':
        return SWISH()
    elif act_name == 'silu':
        return SiLU()
    elif act_name == 'celu':
        return CELU(alpha)
    else:
        raise NameError("Not supported activation: {}".format(name))

def scatter_plot(pred: np.ndarray = None, target: np.ndarray = None):
    fig, ax = plt.subplots()
    ax.scatter(x=pred, y=target, s=25, c='g', alpha=0.5, marker=".")
    ax.set_title('Prediction VS Target')
    ax.set_aspect('equal')
    min_val, max_val = np.min([target, pred]), np.max([target, pred])
    ax.plot([min_val, max_val], [min_val, max_val],
            ls="--", linewidth=1, c='r')
    plt.xlabel('Prediction', fontsize=15)
    plt.ylabel('Target', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    return fig

def scatter_plot_density(pred: np.ndarray = None, target: np.ndarray = None):
    fig, ax = plt.subplots()
    # Calculate the point density
    xy = np.vstack([pred, target])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    pred, target, z = pred[idx], target[idx], z[idx]
    # scatter plot
    ax.scatter(x=pred, y=target, s=25, c=z, marker=".")
    ax.set_title('Prediction VS Target')
    ax.set_aspect('equal')
    min_val, max_val = np.min([target, pred]), np.max([target, pred])
    ax.plot([min_val, max_val], [min_val, max_val],
            ls="--", linewidth=1, c='r')
    plt.xlabel('Prediction', fontsize=15)
    plt.ylabel('Target', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    return fig

class cosine_similarity_loss(nn.Module):
    def __init__(self):
        super(cosine_similarity_loss, self).__init__()

    def forward(self, pred, target):
        vec_product = torch.sum(pred*target, dim=-1)
        pred_norm = torch.norm(pred, p=2, dim=-1)
        target_norm = torch.norm(target, p=2, dim=-1)
        loss = torch.tensor(1.0).type_as(
            pred) - vec_product/(pred_norm*target_norm)
        loss = torch.mean(loss)
        return loss

class sum_zero_loss(nn.Module):
    def __init__(self):
        super(sum_zero_loss, self).__init__()

    def forward(self, pred, target):
        loss = torch.sum(pred, dim=0).pow(2).sum(dim=-1).sqrt()
        return loss

class Euclidean_loss(nn.Module):
    def __init__(self):
        super(Euclidean_loss, self).__init__()

    def forward(self, pred, target):
        dist = (pred - target).pow(2).sum(dim=-1).sqrt()
        loss = torch.mean(dist)
        return loss

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        return torch.sqrt(self.mse(pred, target))

def get_losses(losses_list: Union[list, tuple] = None):
    for loss_dict in losses_list:
        if loss_dict['metric'].lower() == 'mse':
            loss_dict['metric'] = nn.MSELoss()
        elif loss_dict['metric'].lower() == 'mae':
            loss_dict['metric'] = nn.L1Loss()
        elif loss_dict['metric'].lower() == 'cosine_similarity':
            loss_dict['metric'] = cosine_similarity_loss()
        elif loss_dict['metric'].lower() == 'sum_zero':
            loss_dict['metric'] = sum_zero_loss()
        elif loss_dict['metric'].lower() == 'euclidean_loss':
            loss_dict['metric'] = Euclidean_loss()
        elif loss_dict['metric'].lower() == 'rmse':
            loss_dict['metric'] = RMSELoss()
        else:
            print(f'This loss function is not supported!')
    return losses_list

def get_metrics(metrics_list: Union[list, tuple] = None):
    metrics = EasyDict()
    for metric in metrics_list:
        if metric.lower() == 'mse':
            metrics.update({metric: nn.MSELoss()})
        elif metric.lower() == 'mae':
            metrics.update({metric: nn.L1Loss()})
        elif metric.lower() == 'cosine_similarity':
            metrics.update({metric: cosine_similarity_loss()})
        elif metric.lower() == 'sum_zero':
            metrics.update({metric: sum_zero_loss()})
        elif metric.lower() == 'euclidean_loss':
            metrics.update({metric: Euclidean_loss()})
        elif metric.lower() == 'rmse':
            metrics.update({metric: RMSELoss()})
        else:
            print(f'This metric function is not supported!')
    return metrics

def get_hparam_dict(config: dict = None):
    hparam_dict = config.ETGNN
    for key in hparam_dict:
        if type(hparam_dict[key]) not in [str, float, int, bool, None]:
            hparam_dict[key] = type(hparam_dict[key]).__name__.split(".")[-1]
    out = {'GNN_Name': 'ETGNN'}
    out.update(dict(hparam_dict))
    return out
