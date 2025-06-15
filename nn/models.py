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


def get_activation(args):
    
    if args.activation == 'silu':
        return nn.SiLU()
    elif args.activation == 'sine':
        return Sine(w0=args.sine_w0)
    elif args.activation == 'relu':
        return nn.ReLU()
    elif args.activation == 'gelu':
        return nn.GELU()
    else:
        return nn.ReLU()

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
    self.activation = get_activation(args)
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
        self.activation = get_activation(args)
        self.embedding = nn.Sequential(nn.Conv1d(in_channels = args.in_channels,
                kernel_size=3, out_channels = args.encoder_dims[0], stride=1, padding = 'same', bias = False),
                        nn.BatchNorm1d(args.encoder_dims[0]),
                        self.activation,
        )
        self.in_channels = args.in_channels 
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
        else:
            x = x.permute(0,2,1)
        return x


class LSTMEncoder(nn.Module):
    def __init__(self, args):
        super(LSTMEncoder, self).__init__()
        self.t_features = args.seq_len//args.stride
        self.activation = get_activation(args)
        # self.conv = Conv2dSubampling(in_channels=in_channels, out_channels=self.encoder_dims[0])
        self.conv1 = nn.Conv1d(in_channels=args.in_channels, out_channels=args.encoder_dims[0], kernel_size=args.kernel_size, padding='same', stride=1)
        self.pool = nn.MaxPool1d(kernel_size=args.stride)
        self.skip = nn.Conv1d(in_channels=args.in_channels, out_channels=args.encoder_dims[0], kernel_size=1, padding=0, stride=args.stride)
        self.drop = nn.Dropout1d(p=args.dropout_p)
        self.batchnorm1 = nn.BatchNorm1d(args.encoder_dims[0])
                    
        self.lstm = nn.LSTM(args.encoder_dims[0], args.encoder_dims[1], num_layers=args.num_layers,
                            batch_first=True, bidirectional=True, dropout=args.dropout_p)
       
        self.output_dim = args.encoder_dims[1]*2
        self.in_channels = args.in_channels
            #     torch.set_rng_state(rng_state)

    def lstm_attention(self, query, keys, values):
        # Query = [BxQ]
        # Keys = [BxTxK]
        # Values = [BxTxV]
        # Outputs = a:[TxB], lin_comb:[BxV]
        # Here we assume q_dim == k_dim (dot product attention)
        scale = 1/(keys.size(-1) ** -0.5)
        query = query.unsqueeze(1) # [BxQ] -> [Bx1xQ]
        keys = keys.transpose(1,2) # [BxTxK] -> [BxKxT]
        energy = torch.bmm(query, keys) # [Bx1xQ]x[BxKxT] -> [Bx1xT]
        energy = F.softmax(energy.mul_(scale), dim=2) # scale, normalize
        linear_combination = torch.bmm(energy, values).squeeze(1) #[Bx1xT]x[BxTxV] -> [BxV]
        return linear_combination

    def forward(self, x, return_cell=False):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        elif len(x.shape) == 3 and x.shape[-1] == 1:
            x = x.transpose(-1,-2)
        x = x.float()
        skip = self.skip(x)
        x = self.drop(self.pool(self.activation(self.batchnorm1(self.conv1(x)))))
        x = x + skip[:, :, :x.shape[-1]] # [B, C, L//stride]
        x = x.view(x.shape[0], x.shape[1], -1).swapaxes(1,2) # [B, L//stride, C]
        x_f,(h_f,c_f) = self.lstm(x) # [B, L//stride, 2*hidden_size], [2*num_layers, B, hidden_size], [2*num_layers, B, hidden_size
        c_f = torch.cat([c_f[-1], c_f[-2]], dim=1) # [B, 2*hidden_szie]
        features = self.lstm_attention(c_f, x_f, x_f) # [B, 2*hidden_size]
        if return_cell:
            return features, h_f, c_f
        return features

      
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

        self.avg_output = args.avg_output
        encoder_dim = conformer_args.encoder_dim
        latent_dim = args.latent_dim if hasattr(args, 'latent_dim') else 0
        self.output_dim = encoder_dim 
        self.regressor = nn.Sequential(
            nn.Linear(encoder_dim + latent_dim, encoder_dim//2),
            nn.BatchNorm1d(encoder_dim//2),
            self.activation,
            nn.Dropout(conformer_args.dropout_p),
            nn.Linear(encoder_dim//2, args.output_dim*args.num_quantiles)
        )
    
    def forward(self, x, y=None, latent=None):
        x_enc, x = self.encoder(x)
        x = x.permute(0,2,1)
        if latent is not None:
            x_enc = torch.cat([x_enc, latent], dim=1)
        output_reg = self.regressor(x_enc)
        output_dec = self.decoder(x)
        return output_reg, output_dec, x_enc

class SimpleRegressor(nn.Module):
    def __init__(self, encoder, args):
        super().__init__()
        self.encoder = encoder
        encoder_dim = args.encoder_dim
        self.regressor = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim//2),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(encoder_dim//2, args.output_dim*args.num_quantiles)
        )
    def forward(self, x, x_dual=None, w_tune=None):
        x_enc = self.encoder(x)
        if isinstance(x_enc, tuple):
            x_enc = x_enc[0]
        out = self.regressor(x_enc)
        return out, x_enc



class MultiTaskSimSiam(nn.Module):
    def __init__(self, encoder, args):
        super().__init__()
        # if args.activation == 'silu':
        #     self.activation = nn.SiLU()
        # elif args.activation == 'sine':
        #     self.activation = Sine(w0=args.sine_w0)
        # else:
        #     self.activation = nn.ReLU()
        
        self.activation = nn.GELU()
        # self.backbone = MultiEncoder(args, conformer_args)
        self.simsiam = SimSiam(encoder) 
        encoder_dim = self.simsiam.output_dim * 2
        self.output_dim = encoder_dim
        # print("encoder_dim: ", encoder_dim, 'conformer_encoder: ', conformer_args.encoder_dim)
        self.regressor = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim//2),
            nn.BatchNorm1d(self.output_dim//2),
            self.activation,
            nn.Dropout(args.dropout_p),
            nn.Linear(self.output_dim//2, args.output_dim*args.num_quantiles)
        )

    def forward(self, x1, x2, y=None):
        out = self.simsiam(x1, x2)
        z = torch.cat([out['z1'], out['z2']], dim=1)
        output_reg = self.regressor(z)
        out['z'] = z
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

class DoubleInputRegressor(nn.Module):
    def __init__(self, encoder1, encoder2, args):
        super().__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.dims1 = self.encoder1.in_channels
        self.stacked_input = args.stacked_input
        self.output_dim = encoder1.output_dim + encoder2.output_dim
        # print("output_dim: ", self.output_dim, "encoder1: ", encoder1.output_dim, "encoder2: ", encoder2.output_dim)
        if not args.encoder_only:
            self.regressor = nn.Sequential(
                nn.Linear(self.output_dim, self.output_dim//2),
                nn.GELU(),
                nn.Dropout(p=0.3),
                nn.Linear(self.output_dim//2, args.output_dim*args.num_quantiles)
            )
        else:
            self.regressor = nn.Identity()

        
    def forward(self, x1, x2=None):
        if self.stacked_input and (x2 is None):
            x1, x2 = x1[:,-self.dims1:,:], x1[:,:-self.dims1,:]
        x1 = self.encoder1(x1)
        if isinstance(x1, tuple):
            x1 = x1[0]
        x2 = self.encoder2(x2)
        if isinstance(x2, tuple):
            x2 = x2[0]
        x = torch.cat([x1.nan_to_num(0), x2.nan_to_num(0)], dim=1)
        out = self.regressor(x)
        return out, x

class MultiEncoder(nn.Module):
    def __init__(self, args, conformer_args):
        super().__init__()
        if args.backbone == 'lstm':
            self.backbone = LSTMEncoder(args)
        else:
            self.backbone = CNNEncoder(args)
        self.head_size = conformer_args.encoder_dim // conformer_args.num_heads
        self.rotary_ndims = int(self.head_size * 0.5)
        self.pe = RotaryEmbedding(self.rotary_ndims)
        self.encoder = ConformerEncoder(conformer_args)
        self.output_dim = conformer_args.encoder_dim
        # self.avg_output = args.avg_output
        
    def forward(self, x):
        backbone_out = self.backbone(x)
        if len(backbone_out.shape) == 2:
            x_enc = backbone_out.unsqueeze(1)
        else:
            x_enc = backbone_out
        RoPE = self.pe(x_enc, x_enc.shape[1]).nan_to_num(0)
        x_enc = self.encoder(x_enc, RoPE)
        if (len(x_enc.shape) == 3):
            x_enc = x_enc.sum(dim=1)
        return x_enc, backbone_out

class SpectraEncoder(MultiEncoder):
    def __init__(self, 
                 surveys=['LAMOST', 'APOGEE'],
                 wavelength_rope_base=10000,
                 **kwargs):
        super().__init__(**kwargs)
        self.wavelength_embedding = nn.Linear(1, self.rotary_ndims)
        self.surveys = surveys
        # Wavelength-based RoPE (replaces sequential positional encoding)
        self.pe = WavelengthRotaryEmbedding(
            dim=self.rotary_ndims,
            base=wavelength_rope_base,
            wavelength_scale='log'
        )
        
    
        
    def forward(self, flux, wavelength):
        """
        Args:
            flux: [batch_size, seq_len] - spectral flux values
            wavelength: [seq_len] or [batch_size, seq_len] - wavelength values
            survey_ids: [batch_size] - survey identifier
            mask: [batch_size, seq_len] - attention mask
        """
        if len(wavelength.shape) == 2:
            wavelength = wavelength.unsqueeze(1)
        wv = self.wavelength_embedding(wavelength)  # [batch_size, rotary_ndims, seq_len]
        backbone_out = self.backbone(flux)
        if len(backbone_out.shape) == 2:
            x_enc = backbone_out.unsqueeze(1)
        else:
            x_enc = backbone_out
        RoPE = self.pe(wavelength, wavelength.shape[1]).nan_to_num(0)
        x_enc = self.encoder(x_enc, RoPE)
        if (len(x_enc.shape) == 3):
            x_enc = x_enc.sum(dim=1)
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
        # Set first layer flag for initialization purposes
        self.encoder = nn.Linear(args.in_channels, args.encoder_dim)
        self.encoder.is_first_layer = True
        self.pooling_method=args.pooling_method
        
        # Create transformer blocks
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.num_layers):
            self.layers.append(Block(args))
        
        self.norm = RMSNorm(args.encoder_dim)
        self.head = MLP(args.encoder_dim, out_features=args.output_dim*args.num_quantiles, dtype=torch.get_default_dtype())
        self.output_dim = args.output_dim*args.num_quantiles
        
        # Calculate and store scaling factors for DeepNorm if needed
        if getattr(args, 'deepnorm', False) and args.num_layers >= 6:
            layer_coeff = args.num_layers / 6.0
            self.alpha = layer_coeff ** 0.5  
            self.beta = layer_coeff ** -0.5
        else:
            self.alpha = 1.0
            self.beta = 1.0
    def forward(self, x, y=None):
        if len(x.shape)==2:
            x = x.unsqueeze(-1)
        elif len(x.shape)==3 and x.shape[1]==1:
            x = x.permute(0,2,1)
        h = self.encoder(x)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        if self.pooling_method == 'sum':
            h = h.sum(dim=1)
        elif self.pooling_method == 'cls_token':
            h = h[:,0,:]
        elif self.pooling_method == 'mean':
            h = h.mean(dim=1)
        output = self.head(h)        
        return output, y, h


class LSTMFeatureExtractor(nn.Module):
    def __init__(self, seq_len=1024, hidden_size=256, num_layers=5, num_classes=4,
                 in_channels=1, channels=256, dropout=0.2, kernel_size=4 ,stride=4, image=False):
        super(LSTMFeatureExtractor, self).__init__()
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.image = image
        self.stride = stride
        self.t_features = self.seq_len//self.stride
        print("image: ", image)
        # self.conv = Conv2dSubampling(in_channels=in_channels, out_channels=channels)
        if not image:
            self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=channels, kernel_size=kernel_size, padding='same', stride=1)
            self.pool = nn.MaxPool1d(kernel_size=stride)
            self.skip = nn.Conv1d(in_channels=in_channels, out_channels=channels, kernel_size=1, padding=0, stride=stride)
            self.drop = nn.Dropout1d(p=dropout)
            self.batchnorm1 = nn.BatchNorm1d(channels)
        else:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=kernel_size, padding='same', stride=1)
            self.pool = nn.MaxPool2d(kernel_size=stride)
            self.skip = nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=1, padding=0, stride=stride)
            self.drop = nn.Dropout2d(p=dropout)
            self.batchnorm1 = nn.BatchNorm2d(channels)        
             
        self.lstm = nn.LSTM(channels, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout)
       
        self.activation = Sine()
        self.num_features = self._out_shape()
        self.output_dim = self.num_features

    def _out_shape(self) -> int:
        """
        Calculates the number of extracted features going into the the classifier part.
        :return: Number of features.
        """
        # Make sure to not mess up the random state.
        rng_state = torch.get_rng_state()
        # try:
        # print("calculating out shape")
        if not self.image:
            dummy_input = torch.randn(2,self.in_channels, self.seq_len)
        else:
            dummy_input = torch.randn(2,self.in_channels, self.seq_len, self.seq_len)
        # dummy_input = torch.randn(2,self.seq_len, self.in_channels)
        input_length = torch.ones(2, dtype=torch.int64)*self.seq_len
        # print("dummy_input: ", dummy_input.shape)
        # x = self.conv_pre(dummy_input)
        x = self.drop(self.pool(self.activation(self.batchnorm1(self.conv1(dummy_input)))))
        # x = self.conv(dummy_input, input_length)
        x = x.view(x.shape[0], x.shape[1], -1).swapaxes(1,2)
        x_f,(h_f,_) = self.lstm(x)
        h_f = h_f.transpose(0,1).transpose(1,2)
        h_f = h_f.reshape(h_f.shape[0], -1)
        # print("finished")
        return h_f.shape[1] 
        # finally:
        #     torch.set_rng_state(rng_state)

    def forward(self, x, return_cell=False):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        elif len(x.shape) == 3 and x.shape[-1] == 1:
            x = x.transpose(-1,-2)
        skip = self.skip(x)
        x = self.drop(self.pool(self.activation(self.batchnorm1(self.conv1(x)))))
        x = x + skip[:, :, :x.shape[-1]] # [B, C, L//stride]
        x = x.view(x.shape[0], x.shape[1], -1).swapaxes(1,2) # [B, L//stride, C]
        x_f,(h_f,c_f) = self.lstm(x) # [B, L//stride, 2*hidden_size], [B, 2*num_layers, hidden_size], [B, 2*num_layers, hidden_size
        if return_cell:
            return x_f, h_f, c_f
        h_f = h_f.transpose(0,1).transpose(1,2)
        h_f = h_f.reshape(h_f.shape[0], -1)
        return h_f


class LSTM_DUAL_LEGACY(nn.Module):
    def __init__(self, dual_model, encoder_dims, lstm_args, predict_size=128,
                 num_classes=4, num_quantiles=1, freeze=False, ssl=False, **kwargs):
        super(LSTM_DUAL_LEGACY, self).__init__(**kwargs)
        # print("intializing dual model")
        # if lstm_model is not None:
        self.feature_extractor = LSTMFeatureExtractor(**lstm_args)
        self.ssl= ssl
        # self.attention = lstm_model.attention
        if freeze:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
                # for param in self.attention.parameters():
                #     param.requires_grad = False
        num_lstm_features = self.feature_extractor.hidden_size*2
        self.num_features = num_lstm_features + encoder_dims
        self.output_dim = self.num_features
        self.dual_model = dual_model
        self.pred_layer = nn.Sequential(
        nn.Linear(self.num_features, predict_size),
        nn.GELU(),
        nn.Dropout(p=0.3),
        nn.Linear(predict_size,num_classes//2 * num_quantiles),)
        self.conf_layer = nn.Sequential(
        nn.Linear(16, 16),
        nn.GELU(),
        nn.Dropout(p=0.3),
        nn.Linear(16,num_classes//2),)
    def lstm_attention(self, query, keys, values):
        # Query = [BxQ]
        # Keys = [BxTxK]
        # Values = [BxTxV]
        # Outputs = a:[TxB], lin_comb:[BxV]
        # Here we assume q_dim == k_dim (dot product attention)
        scale = 1/(keys.size(-1) ** -0.5)
        query = query.unsqueeze(1) # [BxQ] -> [Bx1xQ]
        keys = keys.transpose(1,2) # [BxTxK] -> [BxKxT]
        energy = torch.bmm(query, keys) # [Bx1xQ]x[BxKxT] -> [Bx1xT]
        energy = F.softmax(energy.mul_(scale), dim=2) # scale, normalize
        linear_combination = torch.bmm(energy, values).squeeze(1) #[Bx1xT]x[BxTxV] -> [BxV]
        return linear_combination
    
    def forward(self, acf, x=None, acf_phr=None):
        if x is None:
            x, acf = acf[:,:-1,:], acf[:,-1,:]
        if len(acf.shape) == 2:
            acf = acf.unsqueeze(1)
        x = x.squeeze()
        # print("acf, x: ", acf.shape, x.shape)
        acf, h_f, c_f = self.feature_extractor(acf, return_cell=True) # [B, L//stride, 2*hidden_size], [B, 2*nlayers, hidden_szie], [B, 2*nlayers, hidden_Size]
        c_f = torch.cat([c_f[-1], c_f[-2]], dim=1) # [B, 2*hidden_szie]
        t_features = self.lstm_attention(c_f, acf, acf) # [B, 2*hidden_size]
        d_features, _ = self.dual_model(x) # [B, encoder_dims]
        # print("nans in features: ", torch.isnan(t_features).sum(), torch.isnan(d_features).sum())
        features = torch.cat([t_features, d_features], dim=1) # [B, 2*hidden_size + encoder_dims]
        if self.ssl:
            return features
        out = self.pred_layer(features)
        return out
       
        # if acf_phr is not None:
        #     phr = acf_phr.reshape(-1,1).float()
        # else:
        #     phr = torch.zeros(features.shape[0],1, device=features.device)
        # mean_features = torch.nn.functional.adaptive_avg_pool1d(features.unsqueeze(1), 16).squeeze(1)
        # mean_features += phr
        # conf = self.conf_layer(mean_features)
        # # print("shapes in models: ", out.shape, conf.shape)
        # return torch.cat([out, conf], dim=1)



