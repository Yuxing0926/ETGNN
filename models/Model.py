"""
/*
 * @Author: Yang Zhong 
 * @Date: 2021-10-09 13:46:53 
 * @Last Modified by: Yang Zhong
 * @Last Modified time: 2021-10-29 21:09:02
 */
"""
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as opt
from typing import List, Dict, Union
from torch.nn import functional as F
from .utils import scatter_plot, scatter_plot_density
import numpy as np
import os
import pandas as pd


class Model(pl.LightningModule):
    def __init__(
            self,
            representation: nn.Module,
            output: nn.Module,
            loss: Union[list, tuple],
            metric: dict,
            lr: float = 1e-3,
            lr_decay: float = 0.1,
            lr_patience: int = 100,
            lr_monitor="training/total_loss"
    ):
        super().__init__()

        self.representation = representation
        self.output_module = output

        self.losses = loss
        self.metrics = metric

        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_patience = lr_patience
        self.lr_monitor = lr_monitor

        self.save_hyperparameters()

        # For gradients
        self.requires_dr = False

    def calculate_loss(self, pred, target, mode):
        loss = torch.tensor(0.0, device=self.device)
        for loss_dict in self.losses:
            loss_fn = loss_dict["metric"]
            loss_i = loss_fn(pred, target)
            loss += loss_dict["loss_weight"] * loss_i            
            loss_log = loss_i.detach().item()
            lossname = type(loss_fn).__name__.split(".")[-1]
            self.log(mode+"/"+lossname, loss_log, on_step=False, on_epoch=True)
        return loss

    def training_step(self, data, batch_idx):
        pred = self(data)
        loss = self.calculate_loss(pred, data.y, 'training')
        self.log("training/total_loss", loss, on_step=False, on_epoch=True)
        #self.check_param()
        return loss

    def validation_step(self, data, batch_idx):
        pred = self(data)
        val_loss = self.calculate_loss(pred, data.y, 'validation').detach().item()
        self.log("validation/total_loss", val_loss, on_step=False, on_epoch=True)
        self.log_metrics(pred, data.y, "validation")
        return {'pred': pred, 'target': data.y}
    
    def validation_epoch_end(self, validation_step_outputs):
        pred = torch.cat([out['pred'] for out in validation_step_outputs])
        pred = pred.detach().cpu().numpy()
        target = torch.cat([out['target'] for out in validation_step_outputs])
        target = target.detach().cpu().numpy()
        figure = scatter_plot(pred.reshape(-1), target.reshape(-1))
        self.logger.experiment.add_figure('validation/PredVSTarget', figure, global_step=self.global_step)

    def test_step(self, data, batch_idx):
        pred, node_attr = self(data), None
        loss = self.calculate_loss(pred, data.y, 'test').detach().item()
        self.log("test/total_loss", loss, on_step=False, on_epoch=True)
        self.log_metrics(pred, data.y, "test")
        return {'pred': pred, 'target': data.y, 'node_attr': node_attr}
    
    def test_epoch_end(self, test_step_outputs):
        pred = torch.cat([out['pred'] for out in test_step_outputs])
        pred = pred.detach().cpu().numpy()
        target = torch.cat([out['target'] for out in test_step_outputs])
        target = target.detach().cpu().numpy()
        figure = scatter_plot(pred.reshape(-1), target.reshape(-1))
        self.logger.experiment.add_figure('test/PredVSTarget', figure, global_step=self.global_step)
        np.save(os.path.join(self.trainer.logger.log_dir, 'prediction.npy'), pred)
        np.save(os.path.join(self.trainer.logger.log_dir, 'target.npy'), target)
        pred = pd.DataFrame(pred)
        target = pd.DataFrame(target)
        pred.to_csv(os.path.join(self.trainer.logger.log_dir, 'prediction.csv'))
        target.to_csv(os.path.join(self.trainer.logger.log_dir, 'target.csv'))
        
        #node_attr = torch.cat([out['node_attr'] for out in test_step_outputs])
        #node_attr = node_attr.detach().cpu().numpy()
        #np.save(os.path.join(self.trainer.logger.log_dir, 'node_attr.npy'), node_attr)

    def forward(self, data):
        self._enable_grads(data)
        representation = self.representation(data)
        pred = self.output_module(data, representation)
        return pred

    def log_metrics(self, pred, target, mode):
        for metric_name in self.metrics.keys():
            metric_fun = self.metrics[metric_name]
            val_loss = metric_fun(pred, target).detach().item()
            self.log(mode+"/"+metric_name, val_loss, on_step=False, on_epoch=True)        
        
    def configure_optimizers(
            self,
    ):
        optimizer = opt.AdamW(self.parameters(), lr=self.lr)
        #optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = {
            "scheduler": opt.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=self.lr_decay,
                patience=self.lr_patience,
                threshold=1e-6,
                cooldown=self.lr_patience // 2,
                min_lr=1e-6,
            ),
            "monitor": self.lr_monitor,
            "interval": "epoch",
            "frequency": 1,
            "strict": True,
        }
        return [optimizer], [scheduler]

    def _enable_grads(self, data):
        if self.requires_dr:
            data.pos.requires_grad_()
    
    def check_param(self):
        for name, parms in self.named_parameters():
            print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
		           '-->grad_value:',parms.grad)
