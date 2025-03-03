import torch
import torch.nn as nn
import torch.nn.functional as F
from nn.Modules.conformer import ConformerEncoder, ConformerDecoder
from nn.Modules.mhsa_pro import RotaryEmbedding, ContinuousRotaryEmbedding
from nn.Modules.flash_mhsa import MHA as Flash_Mha
from nn.Modules.mlp import Mlp as MLP
from nn.simsiam import projection_MLP, SimSiam

import numbers
import torch.nn.init as init
from typing import Union, List, Optional, Tuple
from torch import Size, Tensor


class MLPEncoder(nn.Module):
    def __init__(self, args):
        """
        Initialize an MLP with hidden layers, BatchNorm, and Dropout.

        Args:
            input_dim (int): Dimension of the input features.
            hidden_dims (list of int): List of dimensions for hidden layers.
            output_dim (int): Dimension of the output.
            dropout (float): Dropout probability (default: 0.0).
        """
        super(MLPEncoder, self).__init__()
        
        layers = []
        prev_dim = args.input_dim
        
        # Add hidden layers
        for hidden_dim in args.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim)) 
            layers.append(nn.SiLU())  
            if args.dropout > 0.0:
                layers.append(nn.Dropout(args.dropout))  
            prev_dim = hidden_dim
        self.model = nn.Sequential(*layers)
        self.output_dim = hidden_dim
        
    
    def forward(self, x, y):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x = self.model(x)
        x = x.mean(-1)
        return x

class RMSNorm(nn.Module):
    def __init__(self, normalized_shape: Union[int, List[int], Size], eps: float = 1e-5, bias: bool = False) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(self.normalized_shape))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.normalized_shape))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.ones_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        var = input.pow(2).mean(dim=-1, keepdim=True) + self.eps
        input_norm = input * torch.rsqrt(var)

        rmsnorm = self.weight * input_norm
        
        if self.bias is not None:
            rmsnorm = rmsnorm + self.bias

        return rmsnorm

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
        x = self.embedding(x.float())
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
                                             out_channels=1, 
                                             kernel_size=3, 
                                             stride=1, 
                                             padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expand the compressed representation
        # x = self.initial_expand(x)
        # x = x.unsqueeze(-1)  # Add sequence dimension
        
        # Apply transposed convolution layers
        x = x.float()
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
        # self.projector = projection_MLP(conformer_args.encoder_dim)
        
        if args.activation == 'silu':
            self.activation = nn.SiLU()
        elif args.activation == 'sine':
            self.activation = Sine(w0=args.sine_w0)
        else:
            self.activation = nn.ReLU()
        
        encoder_dim = conformer_args.encoder_dim
        self.regressor = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim//2),
            nn.BatchNorm1d(encoder_dim//2),
            self.activation,
            nn.Dropout(conformer_args.dropout_p),
            nn.Linear(encoder_dim//2, args.output_dim*args.num_quantiles)
        )
    
    def forward(self, x, y=None):
        x_enc, x = self.encoder(x)
        output_reg = self.regressor(x_enc)
        output_dec = self.decoder(x)
        return output_reg, output_dec

class MultiTaskSimSiam(nn.Module):
    def __init__(self, args, conformer_args):
        super().__init__()
        if args.activation == 'silu':
            self.activation = nn.SiLU()
        elif args.activation == 'sine':
            self.activation = Sine(w0=args.sine_w0)
        else:
            self.activation = nn.ReLU()

        self.backbone = MultiEncoder(args, conformer_args)
        self.simsiam = SimSiam(self.backbone) 
        encoder_dim = self.simsiam.output_dim * 2
        self.regressor = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim//2),
            nn.BatchNorm1d(encoder_dim//2),
            self.activation,
            nn.Dropout(conformer_args.dropout_p),
            nn.Linear(encoder_dim//2, args.output_dim*args.num_quantiles)
        )

    def forward(self, x1, x2, y=None):
        out = self.simsiam(x1, x2)
        z = torch.cat([out['z1'], out['z2']], dim=1)
        output_reg = self.regressor(z)
        out['preds'] = output_reg
        return out

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
        if torch.isnan(x).sum() > 0:
            print("nans in x: ", torch.isnan(x).sum(), x.shape)
        backbone_out = self.backbone(x)
        # print("nans in backbone: ", torch.isnan(backbone_out).sum(), backbone_out.shape)
        if torch.isnan(backbone_out).sum() > 0:
            print("nans in backbone: ", torch.isnan(backbone_out).sum(), backbone_out.shape)
        # Create x_enc from backbone_out
        if len(backbone_out.shape) == 2:
            x_enc = backbone_out.unsqueeze(1).clone()
        else:
            x_enc = backbone_out.permute(0,2,1).clone()
            
        RoPE = self.pe(x_enc, x_enc.shape[1]).nan_to_num(0)
        # x_enc = x_enc.nan_to_num(0)
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


class Block(nn.Module):
    """
    Transformer block combining attention and feed-forward layers.

    Attributes:
        attn (nn.Module): Attention layer (MLA).
        ffn (nn.Module): Feed-forward network (MLP or MoE).
        attn_norm (nn.Module): Layer normalization for attention.
        ffn_norm (nn.Module): Layer normalization for feed-forward network.
    """
    def __init__(self, args):
        """
        Initializes the Transformer block.

        Args:
            layer_id (int): Layer index in the transformer.
            args (ModelArgs): Model arguments containing block parameters.
        """
        super().__init__()
        self.attn = Flash_Mha(embed_dim=args.encoder_dim, num_heads=args.num_heads, dropout=args.dropout)
        self.ffn = MLP(in_features=args.encoder_dim)
        self.attn_norm = RMSNorm(args.encoder_dim)
        self.ffn_norm = RMSNorm(args.encoder_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position in the sequence.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor after block computation.
        """
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x

class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = nn.Linear(args.in_channels, args.encoder_dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.num_layers):
            self.layers.append(Block(args))
        self.norm = RMSNorm(args.encoder_dim)
        self.head = MLP(args.encoder_dim, out_features=args.output_dim*args.num_quantiles, dtype=torch.get_default_dtype())
        self.output_dim = args.output_dim*args.num_quantiles
    def forward(self, x, y=None):
        if len(x.shape)==2:
            x = x.unsqueeze(-1)
        elif len(x.shape)==3 and x.shape[1]==1:
            x = x.permute(0,2,1)
        h = self.encoder(x)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)[:, -1]
        output = self.head(h)        
        return output, y
