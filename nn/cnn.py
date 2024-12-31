import torch
import torch.nn as nn
from nn.Modules.conformer import ConformerEncoder, ConformerDecoder
from nn.Modules.mhsa_pro import RotaryEmbedding, ContinuousRotaryEmbedding


class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)

class ConvBlock(nn.Module):
  def __init__(self, args, num_layer) -> None:
    super().__init__()
    if args.activation == 'silu':
        self.activation = nn.SiLU()
    elif args.activation == 'sine':
        self.activation = Sine(w0=args.sine_w0)
    else:
        self.activation = nn.ReLU()
    in_channels = args.encoder_dims[num_layer-1] if num_layer < len(args.encoder_dims) else args.encoder_dims[-1]
    out_channels = args.encoder_dims[num_layer] if num_layer < len(args.encoder_dims) else args.encoder_dims[-1]
    self.layers = nn.Sequential(
        nn.Conv1d(in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=args.kernel_size,
                stride=1, padding='same', bias=False),
        nn.BatchNorm1d(num_features=out_channels),
        self.activation,
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:  
    return self.layers(x)

class CNNEncoder(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        print("Using CNN encoder wit activation: ", args.activation, 'args avg_output: ', args.avg_output)
        if args.activation == 'silu':
            self.activation = nn.SiLU()
        elif args.activation == 'sine':
            self.activation = Sine(w0=args.sine_w0)
        else:
            self.activation = nn.ReLU()
        self.embedding = nn.Sequential(nn.Conv1d(in_channels = args.in_channels,
                kernel_size=3, out_channels = args.encoder_dims[0], stride=1, padding = 'same', bias = False),
                        nn.BatchNorm1d(args.encoder_dims[0]),
                        self.activation,
        )
        
        self.layers = nn.ModuleList([ConvBlock(args, i+1)
        for i in range(args.num_layers)])
        self.pool = nn.MaxPool1d(2)
        self.output_dim = args.encoder_dims[-1]
        self.min_seq_len = 2 
        self.avg_output = args.avg_output
        
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
        if self.avg_output:
            x = x.mean(dim=-1)
        return x

class CNNDecoder(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        print("Using CNN decoder with activation: ", args.activation)
        
        # Reverse the encoder dimensions for upsampling
        decoder_dims = args.encoder_dims[::-1]
        
        if args.activation == 'silu':
            self.activation = nn.SiLU()
        elif args.activation == 'sine':
            self.activation = Sine(w0=args.sine_w0)
        else:
            self.activation = nn.ReLU()
        
        # Initial embedding layer to expand the compressed representation
        self.initial_expand = nn.Linear(decoder_dims[0], decoder_dims[0] * 4)
        
        # Transposed Convolutional layers for upsampling
        self.layers = nn.ModuleList()
        for i in range(args.num_layers):
            if i  < len(decoder_dims) - 1:
                in_channels = decoder_dims[i] 
                out_channels = decoder_dims[i+1]
            else:
                in_channels = decoder_dims[-1]
                out_channels = decoder_dims[-1]
            
            # Transposed Convolution layer
            layer = nn.Sequential(
                nn.ConvTranspose1d(in_channels=in_channels, 
                                   out_channels=out_channels, 
                                   kernel_size=4, 
                                   stride=2, 
                                   padding=1, 
                                   bias=False),
                nn.BatchNorm1d(out_channels),
                self.activation
            )
            self.layers.append(layer)
        
        # Final layer to match original input channels
        self.final_conv = nn.ConvTranspose1d(in_channels=decoder_dims[-1], 
                                             out_channels=args.in_channels, 
                                             kernel_size=3, 
                                             stride=1, 
                                             padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expand the compressed representation
        # x = self.initial_expand(x)
        # x = x.unsqueeze(-1)  # Add sequence dimension
        
        # Apply transposed convolution layers
        for layer in self.layers:
            x = layer(x)
        
        # Final convolution to get back to original input channels
        x = self.final_conv(x)
        
        return x.squeeze()

class CNNEncoderDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.encoder = CNNEncoder(args)
        self.decoder = CNNDecoder(args)
        # create a transformer layers with hidden size as args.encoder_dims[-1]
        if args.transformer_layers > 0:
            self.transformer = nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(
                            d_model=args.encoder_dims[-1],
                            dim_feedforward=args.encoder_dims[-1]*4,
                            nhead=8, 
                            dropout=0.2,
                        ),
                        num_layers=args.transformer_layers,)
        else:
            self.transformer = None
            
    def forward(self, x, y=None):
        # Encode the input
        encoded = self.encoder(x)
        if self.transformer is not None:
            encoded = self.transformer(encoded.permute(0,2,1)).permute(0,2,1)
        # Decode the compressed representation
        reconstructed = self.decoder(encoded)
        
        return reconstructed


class CNNRegressor(nn.Module):
    def __init__(self, args, conformer_args):
        super().__init__()
        # self.encoder = MultiEncoder(args, conformer_args)
        self.backbone = CNNEncoder(args)

        self.head_size = conformer_args.encoder_dim // conformer_args.num_heads
        self.rotary_ndims = int(self.head_size * 0.5)
        self.pe = RotaryEmbedding(self.rotary_ndims)

        self.encoder = ConformerEncoder(conformer_args)
        self.output_dim = conformer_args.encoder_dim
        
    
        
        if args.activation == 'silu':
            self.activation = nn.SiLU()
        elif args.activation == 'sine':
            self.activation = Sine(w0=args.sine_w0)
        else:
            self.activation = nn.ReLU()
        
        self.regressor = nn.Sequential(
            nn.Linear(conformer_args.encoder_dim, conformer_args.encoder_dim//2),
            self.activation,
            nn.Linear(conformer_args.encoder_dim//2, args.output_dim)
        )
    
    def forward(self, x):
        # Encode the input
        x = self.backbone(x)
        x = x.unsqueeze(1)
        # x = x.permute(0,2,1)
        RoPE = self.pe(x, x.shape[1]) # RoPE: [2, B, L, encoder_dim], 2: sin, cos
        x = self.encoder(x, RoPE).squeeze(1)

        # x = self.transformer(x)
        # x = x.sum(dim=1)
        # x, _ = self.encoder(x)
        output = self.regressor(x)
        
        return output


class MultiTaskRegressor(nn.Module):
    def __init__(self, args, conformer_args):
        super().__init__()
        self.encoder = MultiEncoder(args, conformer_args)
        # self.backbone = CNNEncoder(args)
        # self.head_size = conformer_args.encoder_dim // conformer_args.num_heads
        # self.rotary_ndims = int(self.head_size * 0.5)
        # self.pe = RotaryEmbedding(self.rotary_ndims)
        # self.encoder = ConformerEncoder(conformer_args)
        # self.output_dim = conformer_args.encoder_dim
        self.decoder = CNNDecoder(args)
        
        if args.activation == 'silu':
            self.activation = nn.SiLU()
        elif args.activation == 'sine':
            self.activation = Sine(w0=args.sine_w0)
        else:
            self.activation = nn.ReLU()
        
        self.regressor = nn.Sequential(
            nn.Linear(conformer_args.encoder_dim, conformer_args.encoder_dim//2),
            self.activation,
            nn.Linear(conformer_args.encoder_dim//2, args.output_dim)
        )
    
    def forward(self, x, y=None):
        # Encode the input
        # x = self.backbone(x)
        # x_enc = x.clone()
        # x_enc = x_enc.mean(dim=-1).unsqueeze(1)
        # RoPE = self.pe(x_enc, x.shape[1]) # RoPE: [2, B, L, encoder_dim], 2: sin, cos
        # x_enc = self.encoder(x_enc, RoPE).squeeze(1)
        # # x_enc = self.encoder(x)
        x_enc, x = self.encoder(x)
        output_reg = self.regressor(x_enc)
        output_dec = self.decoder(x)
        
        return output_reg, output_dec

class MultiEncoder(nn.Module):
    def __init__(self, args, conformer_args):
        super().__init__()
        self.backbone = CNNEncoder(args)
        self.head_size = conformer_args.encoder_dim // conformer_args.num_heads
        self.rotary_ndims = int(self.head_size * 0.5)
        self.pe = RotaryEmbedding(self.rotary_ndims)
        self.encoder = ConformerEncoder(conformer_args)
        self.output_dim = conformer_args.encoder_dim
        
    def forward(self, x):
        # Encode the input
        x = self.backbone(x)
        if len(x.shape)==2:
            x_enc = x.unsqueeze(1)
        else:
            x_enc = x.permute(0,2,1)
        RoPE = self.pe(x_enc, x_enc.shape[1]) # RoPE: [2, B, L, encoder_dim], 2: sin, cos
        x_enc = self.encoder(x_enc, RoPE)
        if len(x_enc.shape)==3:
            x_enc = x_enc.sum(dim=1)
        return x_enc, x