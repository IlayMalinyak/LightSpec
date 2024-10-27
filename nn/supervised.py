import torch
import torch.nn as nn

class DualModel(nn.Module):
    def __init__(self, spectra_encoder, lc_encoder,
                 spectra_dims, lc_dims, output_dim,
                  lc_only=False) -> None:
        super().__init__()
        self.spectra_encoder = spectra_encoder
        self.lc_encoder = lc_encoder
        self.lc_only = lc_only
        hidden_dim = spectra_dims + lc_dims + 1 if not lc_only else lc_dims + 1
        self.pred_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.SiLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim//2, output_dim),
        )
        
    def forward(self, spectra: torch.Tensor, lc: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        f_y = self.lc_encoder(lc)
        p = p.unsqueeze(-1)
        f_y = torch.cat([f_y, p], dim=-1)
        if self.lc_only:
            x = f_y
        else:
            f_x, _ = self.spectra_encoder(spectra)
            x = torch.cat([f_x, f_y], dim=-1)
        x = self.pred_layer(x)
        return x

