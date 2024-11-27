import torch
import torch.nn as nn
from mamba_ssm import Mamba

class MambaEncoder(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.embedding = nn.Sequential(nn.Conv1d(in_channels = args.in_channels,
                kernel_size=3, out_channels = args.encoder_dim, stride=1, padding = 'same', bias = False),
                        nn.BatchNorm1d(args.encoder_dim),
                        nn.SiLU(),
        )
        
        self.layers = nn.ModuleList([Mamba(
                                    # This module uses roughly 3 * expand * d_model^2 parameters
                                    d_model=args.encoder_dim, # Model dimension d_model
                                    d_state=16,  # SSM state expansion factor
                                    d_conv=4,    # Local convolution width
                                    expand=2, )   # Block expansion factor
                                    for _ in range(args.num_layers)])
        self.pool = nn.AdaptiveAvgPool1d(2)
        self.output_dim = args.encoder_dim
        self.min_seq_len = 2 
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print('before', x.shape)
        if len(x.shape)==2:
            x = x.unsqueeze(1)
        if len(x.shape)==3 and x.shape[-1]==1:
            x = x.permute(0,2,1)
        x = self.embedding(x)
        x = x.permute(0,2,1)
        for m in self.layers:
            x = m(x)
            print('after mamba', x.shape)
            if x.shape[-1] > self.min_seq_len:
                x = self.pool(x)
        print('after', x.shape)
        x = x.mean(dim=-1)
        return
    
class MambaSeq2Seq(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.embedding = nn.Sequential(nn.Conv1d(in_channels = args.in_channels,
                kernel_size=3, out_channels = args.encoder_dim, stride=1, padding = 'same', bias = False),
                        nn.BatchNorm1d(args.encoder_dim),
                        nn.SiLU(),
        )
        self.layers = nn.ModuleList([Mamba(
                                    # This module uses roughly 3 * expand * d_model^2 parameters
                                    d_model=args.encoder_dim, # Model dimension d_model
                                    d_state=args.d_state,  # SSM state expansion factor
                                    d_conv=args.d_conv,    # Local convolution width
                                    expand=args.expand, )   # Block expansion factor
                                    for _ in range(args.num_layers)])
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if len(x.shape)==2:
            x = x.unsqueeze(1)
        if len(x.shape)==3 and x.shape[-1]==1:
            x = x.permute(0,2,1)
        x = self.embedding(x)
        x = x.permute(0,2,1)
        for m in self.layers:
            x = m(x)
        x = x.sum(dim=-1)
        return x