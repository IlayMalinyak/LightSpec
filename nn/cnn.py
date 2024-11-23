import torch
import torch.nn as nn

class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)

class ConvBlock(nn.Module):
  def __init__(self, args) -> None:
    super().__init__()
    if args.activation == 'silu':
        self.activation = nn.SiLU()
    elif args.activation == 'sine':
        self.activation = Sine(w0=args.sine_w0)
    self.activation = nn.ReLU()
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
        self.pool = nn.MaxPool1d(2)
        self.output_dim = args.encoder_dim
        self.min_seq_len = 2 
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape)==2:
            x = x.unsqueeze(1)
        if len(x.shape)==3 and x.shape[-1]==1:
            x = x.permute(0,2,1)
        x = self.embedding(x)
        for m in self.layers:
            x = m(x)
            if x.shape[-1] > self.min_seq_len:
                x = self.pool(x)
        x = x.mean(dim=-1)
        return x