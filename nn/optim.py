import torch
import torch.nn as nn
from typing import Optional, Sequence
from torch import Tensor
from torch.nn import functional as F
from util.cgs_consts import *
import numpy as np
import warnings

class StephanBoltzmanLoss(nn.Module):
    def __init__(self, reduction: str = 'mean', max_T: float = 5, max_R: float = 10) -> None:
        super(StephanBoltzmanLoss, self).__init__()
        self.reduction = reduction
        self.max_T = max_T
        self.max_R = max_R

    def get_rule(self, x):
        """
        calculate Luminosity using Stephan-Boltzman Law
        assuming the input is in shape (batch_size, labels, num_quantile) and the order of [T, R,...] in solar units
        """
        num_quantile = x.shape[-1]
        median = num_quantile // 2
        T = x[:, 0, median]
        R = x[:, 1, median]
        mask_T_max = T > self.max_T
        mask_R_max = R > self.max_R
        mask_T_min = T < 0
        mask_R_min = R < 0
        mask = mask_T_max | mask_R_max | mask_T_min | mask_R_min
        log_L = torch.log10(sigma_sb * 4 * np.pi * (x[:, 1, median] * R_sun) ** 2 * (x[:, 0, median] * 5778) ** 4 / L_sun) 
        return log_L, mask

        

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        l_hat, l_mask = self.get_rule(input)
        loss = F.mse_loss(l_hat, target[:, 2], reduction=self.reduction)
        if self.reduction == 'none':
            loss = loss.unsqueeze(-1).expand_as(target)
        loss = loss * (~l_mask.unsqueeze(-1)).float()
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
        return loss

class TotalEnergyLoss(nn.Module):
    def __init__(self, reduction: str = 'mean') -> None:
        super(TotalEnergyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(input.sum(-1).unsqueeze(-1), target.sum(-1).unsqueeze(-1), reduction=self.reduction)

class SumLoss(nn.Module):
    def __init__(self, losses: Sequence[nn.Module], weights: Optional[Sequence[float]] = None) -> None:
        super(SumLoss, self).__init__()
        self.losses = losses
        self.weights = weights

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss = 0
        for i, l in enumerate(self.losses):
            # print(i, l),
            # print(l(input, target))
            cur_loss = l(input, target)
            cur_loss.nan_to_num_(0)
            loss += cur_loss * self.weights[i] if self.weights is not None else cur_loss
        return loss

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
    Conformalized Quantile Regression (CQR) for multi-label prediction intervals.

    This implementation supports multiple labels and multiple quantile pairs.
    For each label, it creates prediction intervals using pairs of quantiles
    (e.g., 10th and 90th percentiles for an 80% prediction interval).

    Parameters
    ----------
    quantiles : list of float
        List of quantiles in ascending order. Should include desired lower bounds,
        optionally a median, and upper bounds. For example, [0.1, 0.5, 0.9] for
        80% prediction intervals with median.
    reduction : str, optional (default='mean')
        Reduction method for the loss function during training.
        Options: 'mean', 'sum', or None.

    Notes
    -----
    Expected tensor shapes throughout the pipeline:
    - Input predictions: (n_samples, n_labels, n_quantiles)
    - Input targets: (n_samples, n_labels)
    - Conformity scores: (n_samples, n_labels, n_pairs)
    - Output predictions: (n_samples, n_labels, n_quantiles)

    where n_pairs = len(quantiles) // 2 represents the number of prediction intervals
    """

    def __init__(self, quantiles, reduction='mean'):
        super().__init__()
        self.quantiles = quantiles
        self.reduction = reduction

        # Validate quantiles
        if not all(q1 < q2 for q1, q2 in zip(quantiles[:-1], quantiles[1:])):
            raise ValueError("Quantiles must be in ascending order")

    def forward(self, preds, target):
        """
        Calculate the quantile loss during training.

        Parameters
        ----------
        preds : torch.Tensor
            Shape: (n_samples, n_labels, n_quantiles)
            Model predictions for each quantile.
        target : torch.Tensor
            Shape: (n_samples, n_labels) or (n_samples, n_labels, 1)
            True target values.

        Returns
        -------
        torch.Tensor
            Scalar loss value if reduction is 'mean' or 'sum',
            else tensor of shape (n_samples,)
        """
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)

        # if len(target.shape) == 2:
        #     target = target.unsqueeze(-1)

        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[..., i]
            losses.append(
                torch.max(
                    (q - 1) * errors,
                    q * errors
                ).unsqueeze(1)
            )

        loss = torch.sum(torch.cat(losses, dim=1), dim=1)

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss

    def calc_conformity_scores(self, predictions, y):
        """
        Calculate conformity scores for each quantile pair.

        Parameters
        ----------
        predictions : numpy.ndarray
            Shape: (n_samples, n_labels, n_quantiles)
            Predicted quantiles for each label.
        y : numpy.ndarray
            Shape: (n_samples, n_labels)
            True target values.

        Returns
        -------
        numpy.ndarray
            Shape: (n_samples, n_labels, n_pairs)
            Conformity scores for each quantile pair and label.
            n_pairs = len(quantiles) // 2
        """
        n_pairs = len(self.quantiles) // 2
        scores = []

        for i in range(n_pairs):
            lower_idx = i
            upper_idx = -(i + 1)

            y_lower = predictions[:, :, lower_idx]  # (n_samples, n_labels)
            y_upper = predictions[:, :, upper_idx]  # (n_samples, n_labels)

            # Maximum violation of lower or upper bound
            score = np.maximum(y_lower - y, y - y_upper)  # (n_samples, n_labels)
            scores.append(score)

        return np.stack(scores, axis=-1)  # (n_samples, n_labels, n_pairs)

    def calibrate(self, preds, target):
        """
        Compute conformity scores for calibration set.

        Parameters
        ----------
        preds : numpy.ndarray
            Shape: (n_cal_samples, n_labels, n_quantiles)
            Predicted quantiles for calibration set.
        target : numpy.ndarray
            Shape: (n_cal_samples, n_labels)
            True target values for calibration set.

        Returns
        -------
        numpy.ndarray
            Shape: (n_cal_samples, n_labels, n_pairs)
            Conformity scores for calibration set.
        """
        return self.calc_conformity_scores(preds, target)

    def predict(self, preds, conformity_scores):
        """
        Apply conformalization to create valid prediction intervals.

        Parameters
        ----------
        preds : numpy.ndarray
            Shape: (n_samples, n_labels, n_quantiles)
            Predicted quantiles for test set.
        conformity_scores : numpy.ndarray
            Shape: (n_cal_samples, n_labels, n_pairs)
            Conformity scores from calibration set.

        Returns
        -------
        numpy.ndarray
            Shape: (n_samples, n_labels, n_quantiles)
            Conformalized predictions with valid coverage.

        Notes
        -----
        For each quantile pair (e.g., 10th and 90th percentiles):
        1. Calculates the appropriate correction based on desired coverage
        2. Expands the prediction interval symmetrically using the correction
        3. Preserves the median prediction if it exists
        """
        n_samples, n_labels, _ = preds.shape
        n_pairs = len(self.quantiles) // 2
        conformal_preds = np.zeros_like(preds)

        # Copy the median predictions if they exist
        if len(self.quantiles) % 2 == 1:
            mid_idx = len(self.quantiles) // 2
            conformal_preds[:, :, mid_idx] = preds[:, :, mid_idx]

        for i in range(n_pairs):
            lower_idx = i
            upper_idx = -(i + 1)

            # Calculate desired coverage level for this pair
            alpha = self.quantiles[upper_idx] - self.quantiles[lower_idx]

            # Calculate and apply corrections for each label
            for j in range(n_labels):
                scores = conformity_scores[:, j, i]
                # Find correction that achieves desired coverage
                correction = np.quantile(scores, alpha)

                # Apply symmetric correction to expand interval
                conformal_preds[:, j, lower_idx] = preds[:, j, lower_idx] - correction
                conformal_preds[:, j, upper_idx] = preds[:, j, upper_idx] + correction

        return conformal_preds

    def evaluate_coverage(self, predictions, targets):
        """
        Evaluate empirical coverage for each prediction interval.

        Parameters
        ----------
        predictions : numpy.ndarray
            Shape: (n_samples, n_labels, n_quantiles)
            Predicted intervals.
        targets : numpy.ndarray
            Shape: (n_samples, n_labels)
            True target values.

        Returns
        -------
        list of tuple
            Each tuple contains (expected_coverage, actual_coverage)
            for each quantile pair.
        """
        n_quantile_pairs = len(self.quantiles) // 2
        coverage_stats = []

        for i in range(n_quantile_pairs):
            lower_idx = i
            upper_idx = -(i + 1)

            lower_bound = predictions[:, :, lower_idx]
            upper_bound = predictions[:, :, upper_idx]

            in_interval = np.logical_and(
                targets >= lower_bound,
                targets <= upper_bound
            )

            coverage = np.mean(in_interval)
            expected_coverage = self.quantiles[upper_idx] - self.quantiles[lower_idx]
            coverage_stats.append((expected_coverage, coverage))

        return coverage_stats
