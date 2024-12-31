import torch
import torch.nn as nn
from nn.cnn import ConvBlock, CNNEncoder
from nn.astroconf import Astroconformer

class DualModel(nn.Module):
    def __init__(self, lc_backbone, spectra_backbone,
                 hidden_dim, output_dim, backbone_args, num_quantiles=5) -> None:                
        super().__init__()
        self.lc_backbone = lc_backbone
        self.spectra_backbone = spectra_backbone
        # self.freeze_backbone()
        self.backbone = CNNEncoder(backbone_args)
        # self.backbone = Astroconformer(backbone_args)
        self.backbone.pred_layer = torch.nn.Identity()
        self.pred_layer = nn.Sequential(
            nn.Linear(backbone_args.encoder_dims[-1], hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.SiLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim//2, output_dim*num_quantiles),
        )
    
    def freeze_backbone(self):
        for param in self.lc_backbone.parameters():
            param.requires_grad = False
        for param in self.spectra_backbone.parameters():
            param.requires_grad = False
        
    def forward(self,  lc: torch.Tensor, spectra: torch.Tensor) -> torch.Tensor:
        out_lc = self.lc_backbone(lc)
        out_spectra = self.spectra_backbone(spectra)
        x = torch.cat([out_lc, out_spectra], dim=1)
        x = self.backbone(x)
        x = self.pred_layer(x)
        return x

