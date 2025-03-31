import torch
from nn.models import Transformer
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist



class MLPHead(nn.Module):
    """
    Simple regression head
    """
    def __init__(self, in_dim, hidden_dim, out_dim, w_dim=0):
        super(MLPHead, self).__init__()
        in_dim += w_dim
        self.predictor = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, out_dim)
            )
    
    def forward(self, x):
        return self.predictor(x)

class MultiModalJEPA(nn.Module):
    def __init__(self, lc_backbone, spectra_backbone, vicreg_predictor_args,
     lc_reg_args, spectra_reg_args, loss_args):
        super(MultiModalJEPA, self).__init__()
        self.lc_backbone = lc_backbone
        self.spectra_backbone = spectra_backbone
        self.vicreg_predictor = Transformer(vicreg_predictor_args)
        self.lc_head = MLPHead(**lc_reg_args)
        self.spectra_head = MLPHead(**spectra_reg_args)
        self.loss_args = loss_args

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    def vicreg_loss(self, x, y):
        # https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py#L239

        repr_loss = F.mse_loss(x, y)

        x = torch.cat(FullGatherLayer.apply(x), dim=0)
        y = torch.cat(FullGatherLayer.apply(y), dim=0)
        batch_size, num_features = x.shape
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2


        cov_x = (x.T @ x) / (batch_size - 1)
        cov_y = (y.T @ y) / (batch_size - 1)
        cov_loss = self.off_diagonal(cov_x).pow_(2).sum().div(
            num_features
        ) + self.off_diagonal(cov_y).pow_(2).sum().div(num_features)
        loss = (
            self.loss_args.sim_coeff * repr_loss
            + self.loss_args.std_coeff * std_loss
            + self.loss_args.cov_coeff * cov_loss
        )
        loss = loss.nan_to_num(0)
        return loss

    def forward(self, lc, spectra, w=None, pred_coeff=1):
        lc_feat = self.lc_backbone(lc)
        if isinstance(lc_feat, tuple):
            lc_feat = lc_feat[0]
        spectra_feat = self.spectra_backbone(spectra)
        if isinstance(spectra_feat, tuple):
            spectra_feat = spectra_feat[0]
        lc_reg_pred = self.lc_head(lc_feat)
        spectra_reg_pred = self.spectra_head(spectra_feat)
        if w is not None:
            w = w.nan_to_num(0)
            spectra_feat = torch.cat((spectra_feat, w), dim=1)
        lc_pred = self.vicreg_predictor(spectra_feat)
        if isinstance(lc_pred, tuple):
            lc_pred = lc_pred[0]
        loss = self.vicreg_loss(lc_feat, lc_pred)
        return {'loss': loss,  'lc_pred': lc_reg_pred, 'spectra_pred': spectra_reg_pred}

class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]
