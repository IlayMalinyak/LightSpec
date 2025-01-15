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
        # create a transformer layers with hidden size as args.encoder_dims[-1]
        # if args.transformer_layers > 0:
        #     self.transformer = nn.TransformerEncoder(
        #                 nn.TransformerEncoderLayer(
        #                     d_model=args.encoder_dims[-1],
        #                     dim_feedforward=args.encoder_dims[-1]*4,
        #                     nhead=8, 
        #                     dropout=0.2,
        #                 ),
        #                 num_layers=args.transformer_layers,)
        # else:
        #     self.transformer = None
            
    def forward(self, x, y=None):
        # Encode the input
        encoded, _ = self.encoder(x)
        # if self.transformer is not None:
        #     encoded = self.transformer(encoded.permute(0,2,1)).permute(0,2,1)
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
            nn.Linear(conformer_args.encoder_dim//2, args.output_dim)
        )
    
    def forward(self, x, y=None):
        x_enc, x = self.encoder(x)
        output_reg = self.regressor(x_enc)
        output_dec = self.decoder(x)
        
        return output_reg, output_dec

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

class LightCurveSpectraMoCo(nn.Module):
    def __init__(self, 
                 spectra_encoder, 
                 lightcurve_encoder, 
                 hidden_dim=512,
                 projection_dim=256,  # Final projection dimension
                 K=2048, 
                 m=0.999,
                 T=0.07,
                 freeze_lightcurve=True,
                 freeze_spectra=True,
                 bidirectional=False):
        """
        Modified MoCo model for aligning light curves and spectra
        
        Args:
            light_curve_encoder (nn.Module): Encoder for light curve data
            spectra_encoder (nn.Module): Encoder for spectra data
            feature_dim (int): Dimension of the feature representation
            queue_size (int): Size of the memory queue
            momentum (float): Momentum coefficient for key encoder update
        """
        super().__init__()
        
        # Encoders
        self.light_curve_encoder = lightcurve_encoder
        # self.light_curve_key_encoder = self._copy_encoder(lightcurve_encoder)
        self.spectra_encoder = spectra_encoder
        if freeze_lightcurve:
            self._freeze_encoder(self.light_curve_encoder)
        if freeze_spectra:
            self._freeze_encoder(self.spectra_encoder)
        spectra_out_dim = spectra_encoder.output_dim
        lightcurve_out_dim = lightcurve_encoder.output_dim
        
        
        # Projection heads

        self.light_curve_projector = self._build_projector(lightcurve_out_dim, hidden_dim, projection_dim)
        
        self.spectra_projector = self._build_projector(spectra_out_dim, hidden_dim, projection_dim)


        self._freeze_projector(self.light_curve_projector, self.spectra_projector)
        
        # Register queue for spectra keys
        # self.register_buffer("spectra_queue", torch.randn(projection_dim, K))
        # self.spectra_queue = F.normalize(self.spectra_queue, dim=0)

        self.register_buffer("queue", torch.randn(projection_dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        
        # Queue pointer
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        # Hyperparameters
        # self.feature_dim = feature_dim
        self.queue_size = K
        self.m = m
        self.T = T

    def weighted_contrastive_loss(self, q, k, sample_properties):
        # Normalize query and key vectors
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        
        # Compute similarity matrix
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        
        # Compute L2 distance between sample properties
        # Assumes sample_properties is a tensor of shape (batch_size, n_features)
        prop_distances = torch.cdist(sample_properties, sample_properties, p=2.0)
        
        # Normalize distances to use as weights
        # You might want to adjust this normalization strategy
        distance_weights = (prop_distances - prop_distances.min()) / (prop_distances.max() - prop_distances.min())
        
        # Create a mask for negative pairs (non-diagonal elements)
        N = logits.shape[0]
        positive_mask = torch.eye(N, dtype=torch.bool, device=logits.device)
        negative_mask = ~positive_mask
        
        # Apply weights to negative pairs
        weighted_logits = logits.clone()
        weighted_logits[negative_mask] *= (1 + distance_weights[negative_mask])
        
        # Prepare labels (positive pair on diagonal)
        labels = torch.arange(N, dtype=torch.long, device=logits.device)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(weighted_logits, labels)
        
        return loss, logits, labels

    def _freeze_encoder(self, encoder):
        """Freeze encoder parameters"""
        for name, param in encoder.named_parameters():
            param.requires_grad = False
        
    def _freeze_projector(self, k_projector, q_projector):
        """Freeze projector parameters"""
        for param_k, param_q in zip(k_projector.parameters(), q_projector.parameters()):
            if param_k.shape == param_q.shape:
                param_k.requires_grad = False
                param_k.data = param_q.data
    
    def _build_projector(self, in_dim, hidden_dim, out_dim):
        """Modified projector with layer normalization"""
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
    
    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]
    
    def _update_key_encoder(self):
        """
        Update key encoder parameters using momentum update
        """
        for param_q, param_k in zip(
            self.light_curve_query_encoder.parameters(), 
            self.light_curve_key_encoder.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
    
    def _dequeue_and_enqueue(self, keys):
        """
        Dequeue the oldest batch and enqueue the new batch of keys
        
        Args:
            keys (torch.Tensor): New keys to enqueue
        """
        # Gather keys across all GPUs
        keys = keys.detach()
        
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        # Replace the keys at ptr
        if ptr + batch_size > self.queue_size:
            # Wrap around
            self.queue[:, ptr:] = keys[:self.queue_size - ptr].T
            self.queue[:, :ptr + batch_size - self.queue_size] = keys[self.queue_size - ptr:].T
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        
        # Move pointer
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr
    
    def _momentum_update_key_encoder(self, query_encoder, key_encoder):
        """
        Momentum update where key_encoder is updated based on query_encoder
        only for the parameters that are shared between the two encoders
        
        Args:
            query_encoder (nn.Module): The encoder being trained
            key_encoder (nn.Module): The momentum-updated encoder
        """
        for param_q,  param_k in zip(
            query_encoder.parameters(), 
            key_encoder.parameters()
        ):
            if param_k.shape == param_q.shape:
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    
    def forward(self, light_curves, spectra, w):
        # Query features from light curves
        q = self.spectra_encoder(spectra)
        if isinstance(q, tuple):
            q = q[0]
        q = self.spectra_projector(q)
        q = F.normalize(q, dim=1)

        
        # Key features from spectra with momentum update
        with torch.no_grad():
            # Momentum update of spectra (key) encoder
            self._momentum_update_key_encoder(
                self.spectra_projector,        # Key encoder
                self.light_curve_projector  # Query encoder
            )
            # k, idx_unshuffle = self._batch_shuffle_ddp(self.light_curve_encoder(light_curves)[0])
            k = self.light_curve_encoder(light_curves)
            if isinstance(k, tuple):
                k = k[0]
            # Compute key features using momentum-updated encoder
            k = self.light_curve_projector(k)
            k = F.normalize(k, dim=1)
            # k = self._batch_unshuffle_ddp(k, idx_unshuffle)
        # print(q[0,:10], k[0,:10])
        # Contrastive learning logic
        loss, logits, labels = self.weighted_contrastive_loss(q, k, w)

        # l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        
        # logits = torch.cat([l_pos, l_neg], dim=1)
        # labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        
        # # Compute NT-Xent (Normalized Temperature Cross Entropy) loss
        # loss = F.cross_entropy(logits / self.T , labels)
        
        # Update queue with current spectra keys
        self._dequeue_and_enqueue(k)
        
        return {'loss': loss, 'logits': logits , 'labels': labels}