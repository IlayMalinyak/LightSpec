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
    def __init__(self, args, conformer_args):
        super().__init__()
        
        self.encoder = MultiEncoder(args, conformer_args)
        self.decoder = CNNDecoder(args)
            
    def forward(self, x, y=None):
        # Encode the input
        encoded, _ = self.encoder(x)
        # if self.transformer is not None:
        # Decode the compressed representation
        reconstructed = self.decoder(encoded)
        
        return reconstructed


class CNNRegressor(nn.Module):
    def __init__(self, args, conformer_args):
        super().__init__()
        self.encoder = MultiEncoder(args, conformer_args)
        if args.freeze_encoder:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
         
        if args.activation == 'silu':
            self.activation = nn.SiLU()
        elif args.activation == 'sine':
            self.activation = Sine(w0=args.sine_w0)
        else:
            self.activation = nn.ReLU()
        
        self.regressor = nn.Sequential(
            nn.Linear(conformer_args.encoder_dim, conformer_args.encoder_dim//2),
            nn.BatchNorm1d(conformer_args.encoder_dim//2),
            self.activation,
            nn.Dropout(0.2),

            nn.Linear(conformer_args.encoder_dim//2, conformer_args.encoder_dim//4),
            nn.BatchNorm1d(conformer_args.encoder_dim//4),
            self.activation,
            nn.Dropout(0.2),

            nn.Linear(conformer_args.encoder_dim//4, conformer_args.encoder_dim//8),
            nn.BatchNorm1d(conformer_args.encoder_dim//8),
            self.activation,
            nn.Dropout(0.2),

            nn.Linear(conformer_args.encoder_dim//8, args.output_dim*args.num_quantiles)
        )
    
    def forward(self, x):
        # Encode the input
        # x = self.backbone(x)
        # x = x.unsqueeze(1)
        # # x = x.permute(0,2,1)
        # RoPE = self.pe(x, x.shape[1]) # RoPE: [2, B, L, encoder_dim], 2: sin, cos
        # x = self.encoder(x, RoPE).squeeze(1)

        # if self.return_logits:
        #     return x

        # x = self.transformer(x)
        # x = x.sum(dim=1)
        x, _ = self.encoder(x)
        output = self.regressor(x)
        
        return output


class MultiTaskRegressor(nn.Module):
    def __init__(self, args, conformer_args):
        super().__init__()
        self.encoder = MultiEncoder(args, conformer_args)
        self.decoder = CNNDecoder(args)
        
        if args.activation == 'silu':
            self.activation = nn.SiLU()
        elif args.activation == 'sine':
            self.activation = Sine(w0=args.sine_w0)
        else:
            self.activation = nn.ReLU()
        
        self.regressor = nn.Sequential(
            nn.Linear(conformer_args.encoder_dim, conformer_args.encoder_dim//2),
            nn.BatchNorm1d(conformer_args.encoder_dim//2),
            self.activation,
            nn.Dropout(conformer_args.dropout_p),
            nn.Linear(conformer_args.encoder_dim//2, args.output_dim*args.num_quantiles)
        )
    
    def forward(self, x, y=None):
        x_enc, x = self.encoder(x)
        output_reg = self.regressor(x_enc)
        output_dec = self.decoder(x)
        
        return output_reg, output_dec

class MultiResRegressor(nn.Module):
    def __init__(self, args, conformer_args):
        super().__init__()
        self.model = MultiTaskRegressor(args, conformer_args)
    
    def forward(self, x, x_high, y=None):
        output_reg, output_dec = self.model(x)
        output_reg_high, output_dec_high = self.model(x_high)      
        return output_reg, output_dec, output_reg_high, output_dec_high

class MultiEncoder(nn.Module):
    def __init__(self, args, conformer_args):
        super().__init__()
        self.backbone = CNNEncoder(args)
        self.backbone.avg_output = False
        self.head_size = conformer_args.encoder_dim // conformer_args.num_heads
        self.rotary_ndims = int(self.head_size * 0.5)
        self.pe = RotaryEmbedding(self.rotary_ndims)
        self.encoder = ConformerEncoder(conformer_args)
        self.output_dim = conformer_args.encoder_dim
        self.avg_output = args.avg_output
        
    def forward(self, x):
        # print("nans in x: ", torch.isnan(x).sum(), x.shape)
        # Store backbone output in a separate tensor
        backbone_out = self.backbone(x)
        # print("nans in backbone: ", torch.isnan(backbone_out).sum(), backbone_out.shape)

        
        # Create x_enc from backbone_out
        if len(backbone_out.shape) == 2:
            x_enc = backbone_out.unsqueeze(1).clone()
        else:
            x_enc = backbone_out.permute(0,2,1).clone()
            
        RoPE = self.pe(x_enc, x_enc.shape[1])
        if torch.isnan(x_enc).sum() > 0:
            print("nans in rope x_enc: ", torch.isnan(x_enc).sum(), x_enc.shape)
        if torch.isnan(RoPE).sum() > 0:
            print("nans in rope: ", torch.isnan(x_enc).sum(), x_enc.shape)
        # print("nans in RoPE: ", torch.isnan(RoPE).sum(), RoPE.shape)
        x_enc = self.encoder(x_enc, RoPE)
        if torch.isnan(x_enc).sum() > 0:
            print("nans in x_enc: ", torch.isnan(x_enc).sum(), x_enc.shape)
        # print("nans in x_enc: ", torch.isnan(x_enc).sum(), x_enc.shape)
        
        if len(x_enc.shape) == 3:
            if self.avg_output:
                x_enc = x_enc.sum(dim=1)
            else:
                x_enc = x_enc.permute(0,2,1)
        # print("nans in x_enc: ", torch.isnan(x_enc).sum())
                
        # Return x_enc and the original backbone output
        return x_enc, backbone_out