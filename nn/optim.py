import torch
import torch.nn as nn
from typing import Optional, Sequence
from torch import Tensor
from torch.nn import functional as F
import numpy as np
import warnings


class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles
        
    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        if len(target.shape) == 2:
            target = target.unsqueeze(-1)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[..., i]
            losses.append(
                torch.max(
                   (q-1) * errors, 
                   q * errors
            ).unsqueeze(1))
        loss = torch.mean(
            torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss

class CQR(nn.Module):
    """
    Confirmalized Quantile Regression. modified from:
    https://github.com/yromano/cqr/tree/master
    for more details see the original paper:
    "Conformalized Quantile Regression"
    Y. Romano, E. Patterson, E.J Candess, 2019, https://arxiv.org/pdf/1905.03222
    """
    def __init__(self, quantiles, reduction='mean'):
        super().__init__()
        self.quantiles = quantiles
        self.reduction = reduction

    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        if len(target.shape) == 2:
            target = target.unsqueeze(-1)
        losses = []
        for i, q in enumerate(self.quantiles):
            p = preds[..., i]
            if len(p.shape) == 2:
                p = p.unsqueeze(-1)
            errors = target - p
            losses.append(
                torch.max(
                    (q - 1) * errors,
                    q * errors
                ).unsqueeze(1))
        loss = torch.sum(torch.cat(losses, dim=-1), dim=-1)
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss

        return loss


    def calc_nc_error(self, prediction, y):
        y_lower = prediction[:, 0]
        y_upper = prediction[:, -1]
        error_low = y_lower - y
        error_high = y - y_upper
        err = np.maximum(error_high, error_low)
        return err


    def apply_inverse(self, nc, significance):
        nc = np.sort(nc, 0)
        index = int(np.ceil((1 - significance) * (nc.shape[0] + 1))) - 1
        index = min(max(index, 0), nc.shape[0] - 1)
        return np.vstack([nc[index], nc[index]])
        
    def calibrate(self, preds, target):
        """
        Calibrate the model by calculating the non-conformity scores
        for each prediction and target
        """
        print("calibrate: ", preds.shape, target.shape)
        errs = []
        for i in range(len(self.quantiles)//2):
            y_lower = preds[:, i][:, None]
            y_upper = preds[:,  -(i + 1)][:, None]
            q_pair = np.concatenate((y_lower, y_upper), axis=1)
            q_error = self.calc_nc_error(q_pair, target)
            errs.append(q_error)
        # self.nc_errs = np.array(errs)
        return np.swapaxes(np.array(errs), 0,1)

    def predict(self, preds, nc_errs):
        conformal_intervals = np.zeros_like(preds)
        for i in range(len(self.quantiles) // 2):
            significance = self.quantiles[-(i+1)] - self.quantiles[i]
            err_dist = self.apply_inverse(nc_errs[:, i], significance)
            err_dist = np.broadcast_to(err_dist[:, None, :], (err_dist.shape[0], preds.shape[0], err_dist.shape[1]))
            conformal_intervals[:, i] = preds[: , i] - err_dist[0, :]
            conformal_intervals[: , -(i+1)] = preds[: , -(i + 1)] + err_dist[1, :]
        conformal_intervals[:, len(self.quantiles) // 2] = preds[:, len(self.quantiles) // 2]
        return conformal_intervals