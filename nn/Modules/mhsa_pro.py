import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from typing import Optional,Tuple
import math
import logging

logger = logging.getLogger(__name__)


rwkv_emb_scale = 0.4 # try 0.4 for char-level english. try 1.0 for chinese.
rwkv_layer_decay = 1.0 # decay weights in higher layers. try 0.5 ~ 1.0.

class AttentionConfig:
  def __init__(self, ctx_len=100, **kwargs):
    self.ctx_len = ctx_len
    for k,v in kwargs.items():
        setattr(self, k, v)


########################################################################################################
# MHA_rotary: Multi-head Attention + Rotary Encoding + GeGLU FFN
########################################################################################################

class WavelengthRotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000, wavelength_scale='log'):
        super().__init__()
        self.dim = dim
        self.base = base
        self.wavelength_scale = wavelength_scale
        
        # Create inverse frequencies for RoPE
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache for efficiency
        self.wavelength_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def _wavelength_to_position(self, wavelength):
        """Convert wavelength to position for RoPE"""
        if self.wavelength_scale == 'log':
            # Log scale for astronomical wavelengths
            log_wave = torch.log(wavelength)
            # Normalize to reasonable range (assuming wavelengths in Angstroms)
            positions = (log_wave - 8.0) / 2.0  # Roughly [0, 2] for 3000-20000 Angstroms
        elif self.wavelength_scale == 'linear':
            # Linear normalization
            positions = (wavelength - wavelength.min()) / (wavelength.max() - wavelength.min())
        else:
            raise ValueError("wavelength_scale must be 'log' or 'linear'")
        
        return positions

    def forward(self, wavelength):
        """Fully vectorized version"""
        # Convert to positions (vectorized)
        valid_mask = wavelength > 0
        log_wave = torch.log(torch.clamp(wavelength, min=1e-8))  # Avoid log(0)
        positions = (log_wave - 8.0) / 2.0
        positions = positions * valid_mask.float()  # Zero out padding
        
        # Vectorized einsum for all batch items at once
        # Reshape for broadcasting: [batch*seq, 1] × [1, dim//2] → [batch*seq, dim//2]
        batch_size, seq_len = wavelength.shape
        positions_flat = positions.view(-1, 1)  # [batch*seq, 1]
        inv_freq_expanded = self.inv_freq.unsqueeze(0)  # [1, dim//2]
        
        freqs_flat = positions_flat * inv_freq_expanded  # [batch*seq, dim//2]
        freqs = freqs_flat.view(batch_size, seq_len, -1)  # [batch, seq, dim//2]
        
        # Duplicate and compute cos/sin
        emb = torch.cat([freqs, freqs], dim=-1)  # [batch, seq, dim]
        return torch.stack([emb.cos(), emb.sin()])


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_len=None):
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()
            self.sin_cached = emb.sin()
        return torch.stack([self.cos_cached, self.sin_cached])

class ContinuousRotaryEmbedding(torch.nn.Module):
    '''Continuous rotary position embedding'''
    def __init__(self, dim, sequence_scale):
        super().__init__()
        base=10000
        self.sequence_scale = sequence_scale
        self.register_buffer('inv_freq', 1. / (base ** (torch.arange(0, dim, 2))))
    
    def forward(self, t):
        t = (t + 0.5)* self.sequence_scale 
        freqs = torch.einsum('ij,k->ijk', t, self.inv_freq) # freqs: [B, L, dim//2]
        emb = torch.cat((freqs, freqs), dim=-1).unsqueeze(1) # emb: [B, 1, L, dim], 1 for broadcast in head_num dim
        return torch.stack([emb.cos(), emb.sin()])
    
def rotate_half(x):
    # print('in rotate half: ', x.shape)
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), -1)

@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    # print('in apply_rotary_pos_emb: ', q.shape, k.shape, cos.shape, sin.shape)
    cos, sin = cos[...,:q.shape[2],:], sin[...,:q.shape[2],:]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

def apply_seq_len_rotary_pos_emb(x, cos, sin):
    """
    Args:
        x: [batch_size, seq_len, dim] 
        cos: [batch_size, seq_len, dim] - batch-specific cos encodings
        sin: [batch_size, seq_len, dim] - batch-specific sin encodings
    """
    # print('in apply_seq_len_rotary_pos_emb: ', x.shape, cos.shape, sin.shape)
    # Split x into pairs for rotation
    x1, x2 = x[..., ::2], x[..., 1::2]
    
    # Apply rotation (now cos/sin are per-batch)
    rotated_x1 = x1 * cos[..., ::2] - x2 * sin[..., ::2]
    rotated_x2 = x1 * sin[..., 1::2] + x2 * cos[..., 1::2]
    
    # Recombine
    rotated_x = torch.zeros_like(x)
    rotated_x[..., ::2] = rotated_x1
    rotated_x[..., 1::2] = rotated_x2
    
    return rotated_x

class MHA_rotary(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.collect_attention_map = False
        self.attention_map = None
        assert args.encoder_dim % args.num_heads == 0
        self.num_heads = args.num_heads
        self.head_size = args.encoder_dim // args.num_heads

        if args.timeshift:
            self.time_shift = nn.ZeroPad2d((0,0,1,0))

        self.query = nn.Linear(args.encoder_dim, args.encoder_dim)
        self.key = nn.Linear(args.encoder_dim, args.encoder_dim)
        self.value = nn.Linear(args.encoder_dim, args.encoder_dim)

        # self.register_buffer("mask", torch.tril(torch.ones(config.ctx_len, config.ctx_len)))
        
        self.rotary_ndims = int(self.head_size * 0.5)
        
        self.rotary_emb = RotaryEmbedding(self.rotary_ndims)

        self.output = nn.Linear(args.encoder_dim, args.encoder_dim)

    def forward(self, x, RoPE, key_padding_mask=None):
        B, T, C = x.size()

        if hasattr(self, 'time_shift'):
            x = torch.cat([self.time_shift(x)[:, :-1, :C//2], x[:, :, C//2:]], dim = -1)

        q = self.query(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)       # (B, T, C) -> (B, nh, T, hs)
        k = self.key(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)         # (B, T, C) -> (B, nh, T, hs)
        v = self.value(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)       # (B, T, C) -> (B, nh, T, hs)

        q, query_pass = q[..., :self.rotary_ndims], q[..., self.rotary_ndims:]
        k, key_pass = k[..., :self.rotary_ndims], k[..., self.rotary_ndims:]
        
        # cos, sin = self.rotary_emb(q, seq_len=T)
        cos, sin = RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)                                     # rotary encoding
        q = torch.cat((q, query_pass), dim=-1)
        k = torch.cat((k, key_pass), dim=-1)  
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))                 # self-attention: (B, nh, T, hs) * (B, nh, hs, T) -> (B, nh, T, T)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask[:, None, None, :]           # (B, T) -> (B, 1, 1, T)
            att = att.masked_fill(key_padding_mask == 0, float('-inf'))
        att = F.softmax(att, dim = -1)                                                  # softmax

        x = att @ v                                                                     # (B, nh, T, T) * (B, nh, T, hs) -> (B, nh, T, hs)
        x = x.transpose(1, 2).contiguous().view(B, T, -1)                               # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)

        x = self.output(x)


        if self.collect_attention_map:
            self.attention_map = att
        
        return x

class MHA_decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.collect_attention_map = False
        self.attention_map = None
        assert args.encoder_dim % args.num_heads == 0
        self.num_heads = args.num_heads
        self.head_size = args.decoder_dim // args.num_heads

        if args.timeshift:
            self.time_shift = nn.ZeroPad2d((0,0,1,0))

        self.query = nn.Linear(args.decoder_dim, args.decoder_dim)
        self.key = nn.Linear(args.decoder_dim, args.decoder_dim)
        self.value = nn.Linear(args.decoder_dim, args.decoder_dim)

        # self.register_buffer("mask", torch.tril(torch.ones(config.ctx_len, config.ctx_len)))
        
        self.rotary_ndims = int(self.head_size * 0.5)
        
        self.rotary_emb = RotaryEmbedding(self.rotary_ndims)

        self.output = nn.Linear(args.decoder_dim, args.decoder_dim)

    def forward(self, x, memory,RoPE, key_padding_mask=None):
        B, T, C = x.size()
        _, L, M = memory.size()

        # print("x size: ", x.size(), 'memory size: ', memory.size())
        # print('B, T, C: ', B, T, C, 'L: ', L)

        q = self.query(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)       # (B, T, C) -> (B, nh, T, hs)
        k = self.key(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)         # (B, T, C) -> (B, nh, T, hs)
        v = self.value(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)       # (B, T, C) -> (B, nh, T, hs)

        q, query_pass = q[..., :self.rotary_ndims], q[..., self.rotary_ndims:]
        k, key_pass = k[..., :self.rotary_ndims], k[..., self.rotary_ndims:]
        
        # cos, sin = self.rotary_emb(q, seq_len=T)
        cos, sin = RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)                                     # rotary encoding
        q = torch.cat((q, query_pass), dim=-1)
        k = torch.cat((k, key_pass), dim=-1)  
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))                 # self-attention: (B, nh, T, hs) * (B, nh, hs, T) -> (B, nh, T, T)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask[:, None, None, :]           # (B, T) -> (B, 1, 1, T)
            att = att.masked_fill(key_padding_mask == 0, float('-inf'))
        att = F.softmax(att, dim = -1)                                                  # softmax

        x = att @ v  
        # print("after attention vals: ", x.shape)                                                                   # (B, nh, T, T) * (B, nh, T, hs) -> (B, nh, T, hs)
        x = x.transpose(1, 2).contiguous().view(B, T, -1)                               # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)

        # x = self.output(x)

        # print("after linear: ", x.shape)                                                                   # (B, nh, T, T) * (B, nh, T, hs) -> (B, nh, T, hs)


        # cross attention:
        q = self.query(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)       # (B, T, C) -> (B, nh, T, hs)
        k = self.key(memory).view(B, L, self.num_heads, self.head_size).transpose(1, 2)         # (B, T, C) -> (B, nh, T, hs)
        v = self.value(memory).view(B, L, self.num_heads, self.head_size).transpose(1, 2)       # (B, T, C) -> (B, nh, T, hs)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))                 # self-attention: (B, nh, T, hs) * (B, nh, hs, T) -> (B, nh, T, T)
        # print("att size: ", att.size())
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask[:, None, None, :]           # (B, T) -> (B, 1, 1, T)
            att = att.masked_fill(key_padding_mask == 0, float('-inf'))
        att = F.softmax(att, dim = -1)                                                  # softmax

        x = att @ v                                                                     # (B, nh, T, T) * (B, nh, T, hs) -> (B, nh, T, hs)
        # print("x deocder size: ", x.size())
        x = x.transpose(1, 2).contiguous().view(B, T, -1)                               # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)
        # print("x deocder size transposed: ", x.size())
        x = self.output(x)

        if self.collect_attention_map:
            self.attention_map = att

        return x

    class GeGLU(torch.nn.Module):
        def __init__(self, config, layer_id, time_shift = False):
            super().__init__()
            self.layer_id = layer_id

            if time_shift:
                self.time_shift = nn.ZeroPad2d((0,0,1,0))

            hidden_sz = 3 * config.n_ffn
            self.key = nn.Linear(config.n_embd, hidden_sz)
            self.value = nn.Linear(config.n_embd, hidden_sz)
            self.weight = nn.Linear(hidden_sz, config.n_embd)

        def forward(self, x):
            B, T, C = x.size()
            if hasattr(self, 'time_shift'):
                x = torch.cat([self.time_shift(x)[:, :-1, :C//2], x[:, :, C//2:]], dim = -1)
            
            k = self.key(x)
            v = self.value(x)        
            y = self.weight(F.gelu(k) * v)
            return y