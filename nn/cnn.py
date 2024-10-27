import torch
import torch.nn as nn

class ConvBlock(nn.Module):
  def __init__(self, args) -> None:
    super().__init__()
    self.layers = nn.Sequential(
        nn.Conv1d(in_channels=args.encoder_dim,
                out_channels=args.encoder_dim,
                kernel_size=args.kernel_size,
                stride=1, padding='same', bias=False),
        nn.BatchNorm1d(num_features=args.encoder_dim),
        nn.SiLU(),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:  
    return self.layers(x)

class CNNEncoder(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.embedding = nn.Sequential(nn.Conv1d(in_channels = args.in_channels,
                kernel_size=3, out_channels = args.encoder_dim, stride=1, padding = 'same', bias = False),
                        nn.BatchNorm1d(args.encoder_dim),
                        nn.SiLU(),
        )
        
        self.layers = nn.ModuleList([ConvBlock(args)
        for _ in range(args.num_layers)])
        self.pool = nn.AdaptiveAvgPool1d(2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape)==2:
            x = x.unsqueeze(1)
        if len(x.shape)==3 and x.shape[-1]==1:
            x = x.permute(0,2,1)
        x = self.embedding(x)
        for m in self.layers:
            x = m(x)
            x = self.pool(x)
        x = x.mean(dim=-1)
        return x