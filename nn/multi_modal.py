import torch
from nn.models import Transformer
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from nn.DualFormer.dual_attention import DualFormer
from postprocessing.eigenspace_analysis import apply_linear_regression


class LinearRegressionWithScaling(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionWithScaling, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.register_buffer('x_mean', torch.zeros(input_dim))
        self.register_buffer('x_std', torch.ones(input_dim))
        self.register_buffer('y_mean', torch.zeros(output_dim))
        self.register_buffer('y_std', torch.ones(output_dim))
        self.is_fitted = False

    def forward(self, X):
        """Forward pass with automatic scaling and inverse scaling."""
        # if not self.is_fitted:
        #     raise RuntimeError("Model must be fitted before making predictions")

        # Scale input
        X_scaled = (X - self.x_mean) / self.x_std

        # Make prediction in scaled space
        y_scaled = self.linear(X_scaled)

        # Inverse transform to original scale
        y_pred = y_scaled * self.y_std + self.y_mean

        return y_pred

    def score(self, X, y):
        """Calculate R^2 score, similar to scikit-learn's score method."""
        with torch.no_grad():
            y_pred = self(X)
            u = ((y - y_pred) ** 2).sum()
            v = ((y - y.mean(dim=0)) ** 2).sum()
            return 1 - u / v

    def accuracy_within_percentage(self, X, y, percentage=0.1):
        """Calculate accuracy as predictions within percentage of true values."""
        with torch.no_grad():
            y_pred = self(X)
            acc = (torch.abs(y_pred - y) < y * percentage).float().mean(dim=0)
            return acc

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


class DualNet(nn.Module):
    def __init__(self, lc_backbone, spectra_backbone, dual_former_args,
     lc_reg_args, spectra_reg_args, freeze_backbone=False):
        super(DualNet, self).__init__()
        self.lc_backbone = lc_backbone
        self.spectra_backbone = spectra_backbone
        self.dual_former = DualFormer(**dual_former_args)
        self.lc_head = MLPHead(**lc_reg_args)
        self.spectra_head = MLPHead(**spectra_reg_args)
        if freeze_backbone:
            self._freeze_backbone()
        self.freeze_backbone = freeze_backbone
        # self.loss_args = loss_args
    
    def _freeze_backbone(self):
        for param in self.lc_backbone.parameters():
            param.requires_grad = False
        for param in self.spectra_backbone.parameters():
            param.requires_grad = False

    def forward(self, lc, spectra, latent=None):
        lc_feat = self.lc_backbone(lc)
        if isinstance(lc_feat, tuple):
            lc_feat = lc_feat[0]
        spectra_feat = self.spectra_backbone(spectra)
        if isinstance(spectra_feat, tuple):
            spectra_feat = spectra_feat[0]
        if not self.freeze_backbone:
            lc_reg_pred = self.lc_head(lc_feat)
            spectra_reg_pred = self.spectra_head(spectra_feat)
        else:
            lc_reg_pred = None
            spectra_reg_pred = None
        dual_pred = self.dual_former(spectra_feat, lc_feat, latent_variables=latent)
        return {'dual_pred': dual_pred, 'lc_pred':lc_reg_pred, 'spec_pred':spectra_reg_pred}


class FineTuner(nn.Module):
    def __init__(self, model, head_args, method='dualformer'):
        super(FineTuner, self).__init__()
        print("head args: ", head_args)
        self.model = model
        self.head = MLPHead(**head_args)
        # self.head = nn.Sequential(
        #     nn.Linear(head_args['in_dim'], head_args['hidden_dim']),
        #     # nn.LayerNorm(head_args['hidden_dim']),
        #     nn.BatchNorm1d(head_args['hidden_dim']),
        #     nn.SiLU(),
        #     nn.Linear(head_args['hidden_dim'], head_args['out_dim'])
        # )
        # self.head = Transformer(head_args)
        # self.head = LinearRegressionWithScaling(head_args['in_dim'], head_args['out_dim'])
        self.sigma_head = nn.Sequential(
            nn.Linear(head_args['in_dim'], head_args['hidden_dim']),
            nn.LayerNorm(head_args['hidden_dim']),
            nn.SiLU(),
            nn.Linear(head_args['hidden_dim'], 1)
        )

        self.method = method
        if method == 'dualformer':
            self.A = self.model.dual_former.projection_head.weight.detach()
            self.eigenvalues, self.eigenvectors = self._get_eigenspace(self.A)

    def _get_eigenspace(self, weight):
        print("max diff weight", torch.max(torch.abs(weight - weight.t())).item())
        if torch.abs(weight - weight.t()).sum() < 1e-6:
            print("Weight matrix is symmetric. Using eigendecomposition.")
            # Symmetric case - eigenvalues are real
            eigenvalues, eigenvectors = torch.linalg.eigh(weight)
            # Sort in descending order of eigenvalues
            idx = torch.argsort(eigenvalues, descending=True)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
        else:
            print("Warning: Weight matrix is not symmetric. error is: ", torch.abs(weight - weight.t()).sum())
            # Non-symmetric case - eigenvalues may be complex
            eigenvalues_complex, eigenvectors_complex = torch.linalg.eig(weight)
            # For simplicity, we'll take the magnitude of complex eigenvalues
            eigenvalues_mag = torch.abs(eigenvalues_complex)
            idx = torch.argsort(eigenvalues_mag, descending=True)
            eigenvalues = eigenvalues_complex[idx]
            eigenvectors = eigenvectors_complex[:, idx]

            # Note: In practice, you might want to handle complex eigenvectors differently
            # Here we'll just take the real part for demonstration
            if torch.is_complex(eigenvectors):
                print("Warning: Complex eigenvectors detected. Using real part for projections.")
                eigenvectors = eigenvectors.real

        # Normalize eigenvectors
        eigenvectors = F.normalize(eigenvectors, dim=0)

        return eigenvalues, eigenvectors

    def forward(self, lc, spectra, latent=None):
        model_out = self.model(lc, spectra, latent)
        if self.method == 'dualformer':
            dual_pred = model_out['dual_pred']
            lc_emb, spec_emb = dual_pred['emb1'], dual_pred['emb2']
            proj_lc, proj_spec = dual_pred['proj1'], dual_pred['proj2']

            final_features = (spec_emb + lc_emb) @ self.eigenvectors
            # y = torch.cat([spec_emb[:, -2].unsqueeze(-1), spec_emb[:, -4].unsqueeze(-1)], dim=-1)
            # lr_res = apply_linear_regression(final_features.cpu().numpy(), y.cpu().numpy(), 'test')
            # final_features = torch.cat((spec_emb, lc_emb), dim=1)
            # final_features = torch.cat((proj_lc, proj_spec), dim=1)
        elif self.method == 'moco':
            final_features = model_out['q']
        predictions = self.head(final_features)
        sigma = self.sigma_head(final_features)
        return predictions, sigma, final_features
