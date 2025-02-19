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
        y_lower = prediction[:, :, 0]  # Shape: (N, num_labels)
        y_upper = prediction[:, :, -1] # Shape: (N, num_labels)
        
        error_low = y_lower - y  # (N, num_labels)
        error_high = y - y_upper # (N, num_labels)
        
        err = np.maximum(error_high, error_low)  # (N, num_labels)
        return err  # Return per-label errors instead of collapsing them


    def apply_inverse(self, nc, significance):
        nc = np.sort(nc,0)
        index = int(np.ceil((1 - significance / 2) * (nc.shape[0] + 1))) - 1
        index = min(max(index, 0), nc.shape[0] - 1)
        return np.vstack([nc[index], nc[index]])


    def calibrate(self, preds, target):
        print("calibrate: ", preds.shape, target.shape)
        errs = []
        
        for i in range(len(self.quantiles) // 2):
            y_lower = preds[:, :, i]  # Shape: (N, num_labels)
            y_upper = preds[:, :, -(i + 1)]  # Shape: (N, num_labels)
            
            q_pair = np.stack([y_lower, y_upper], axis=-1)  # (N, num_labels, 2)
            q_error = self.calc_nc_error(q_pair, target)  # (N, num_labels)
            
            errs.append(q_error)
        errs = np.stack(errs)
        errs = np.swapaxes(np.swapaxes(errs, 0, 1), 1,2)
        return errs  # Shape: (N, num_labels, num_quantile_pairs)

    def predict(self, preds, nc_errs):
        print("nc errors shape: ", nc_errs.shape)
        conformal_intervals = np.zeros_like(preds)  # Shape: (N, num_labels, num_quantiles)

        for i in range(len(self.quantiles) // 2):
            significance = self.quantiles[-(i+1)] - self.quantiles[i]

            for j in range(preds.shape[1]):
                err_dist = self.apply_inverse(nc_errs[:, j, i], significance)  # (2)
                err_dist = np.hstack([err_dist] * preds.shape[0])
                print("err dist: ", err_dist.shape)
                conformal_intervals[:, j, i] = preds[:, j, i] - err_dist[0]  # Lower bound
                conformal_intervals[:, j, -(i+1)] = preds[:, j, -(i+1)] + err_dist[1]  # Upper bound

        conformal_intervals[:, :, len(self.quantiles) // 2] = preds[:, :, len(self.quantiles) // 2]  # Median
        return conformal_intervals
