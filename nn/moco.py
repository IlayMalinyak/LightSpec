# code from https://github.com/facebookresearch/moco/blob/main/moco/builder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from nn.simsiam import projection_MLP
from nn.models import Transformer
from nn.Modules.conformer import ConformerEncoder
import time
import numpy as np
from collections import deque
from copy import deepcopy



class MultimodalMoCo(nn.Module):
    """
    Multimodal MoCo model with shared encoder for light curves and spectra.
    """
    def __init__(
        self,
        spectra_encoder,  # pre-trained spectra encoder
        lightcurve_encoder,  # pre-trained light curve encoder
        projection_args,
        projection_dim=128,  # Final projection dimension
        hidden_dim=512,  # Hidden dimension of projection MLP
        num_layers=8,
        K=65536,  # Queue size
        m=0.999,  # Momentum coefficient
        T=0.07,  # Temperature
        freeze_lightcurve=True,  # Whether to freeze light curve encoder
        freeze_spectra=True,  # Whether to freeze spectra encoder
        bidirectional=True,  # Whether to train in both directions
        transformer=False,
    ):
        super(MultimodalMoCo, self).__init__()
        
        self.K = K
        self.m = m
        self.T = T
        self.bidirectional = bidirectional
        self.criterion = nn.CrossEntropyLoss()
        
        if freeze_lightcurve:
            self._freeze_encoder(lightcurve_encoder)
        if freeze_spectra:
            self._freeze_encoder(spectra_encoder)
        
        self.spectra_encoder_q = spectra_encoder
        self.lightcurve_encoder_q = lightcurve_encoder
        
        with torch.no_grad():
            spectra_out_dim = spectra_encoder.output_dim
            lightcurve_out_dim = lightcurve_encoder.output_dim
        
        self.shared_encoder_q = Transformer(projection_args)
        
        self.shared_encoder_k = copy.deepcopy(self.shared_encoder_q)
        
    
        self._freeze_encoder(self.shared_encoder_k)
        
        self.register_buffer("lightcurve_queue", torch.randn(projection_args.output_dim, K))
        self.lightcurve_queue = F.normalize(self.lightcurve_queue, dim=0)
        self.register_buffer("lightcurve_queue_ptr", torch.zeros(1, dtype=torch.long))
        
        if bidirectional:
            self.register_buffer("spectra_queue", torch.randn(projection_args.output_dim, K))
            self.spectra_queue = F.normalize(self.spectra_queue, dim=0)
            self.register_buffer("spectra_queue_ptr", torch.zeros(1, dtype=torch.long))


    def contrastive_loss(self, q, k, queue, sample_properties=None):
        """
        Compute contrastive loss using queue for negative samples
        """
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        
        l_neg = torch.einsum('nc,ck->nk', [q, queue.clone().detach()])
        
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits = logits / self.T
        
        if sample_properties is not None:
            curr_distances = torch.cdist(sample_properties, sample_properties, p=2.0)
            curr_distances = (curr_distances - curr_distances.min()) / (curr_distances.max() - curr_distances.min())
            
            weights = torch.ones_like(logits)
            weights[:, 1:] = 1 + curr_distances  # Apply weights only to negative pairs
            logits = logits * weights
        
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss, logits, labels

    def _freeze_encoder(self, encoder):
        """Freeze encoder parameters"""
        for name, param in encoder.named_parameters():
            param.requires_grad = False
            
            
    def _build_projector(self, in_dim, hidden_dim, out_dim, num_layers, transformer=False):
        """Modified projector with layer normalization and optional transformer architecture."""
        if transformer:
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=hidden_dim,
                        dim_feedforward=hidden_dim*4,
                        nhead=8, 
                        dropout=0.2,
                    ),
                    num_layers=num_layers,
                ),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(in_dim),
                nn.Dropout(0.2),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(out_dim),
                nn.Dropout(0.2),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim)
            )
    
    @torch.no_grad()
    def _momentum_update(self):
        """Update momentum encoders"""
        self._momentum_update_encoder(
            self.spectra_encoder_q, self.spectra_encoder_k,
            self.spectra_proj_q, self.spectra_proj_k
        )
        self._momentum_update_encoder(
            self.lightcurve_encoder_q, self.lightcurve_encoder_k,
            self.lightcurve_proj_q, self.lightcurve_proj_k
        )
    
    def _momentum_update_encoder(self, encoder_q, encoder_k, proj_q, proj_k):
        """Update one encoder-projector pair"""
        for param_q, param_k in zip(encoder_q.parameters(), encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(proj_q.parameters(), proj_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, queue, queue_ptr):
        """Update queue with handling for variable batch sizes"""
        batch_size = keys.shape[0]
        ptr = int(queue_ptr)
        
        if ptr + batch_size > self.K:
            first_part = self.K - ptr 
            queue[:, ptr:] = keys.T[:, :first_part]  
            remaining = batch_size - first_part
            if remaining > 0:  
                queue[:, :remaining] = keys.T[:, first_part:]  
            ptr = remaining if remaining > 0 else 0
        else:
            queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K  
        
        queue_ptr[0] = ptr
        return queue, queue_ptr
    
    def forward(self, lightcurves, spectra, w=None):
       
        spectra_feat = self.spectra_encoder_q(spectra)
        if isinstance(spectra_feat, tuple):
            spectra_feat = spectra_feat[0]
        lightcurve_feat = self.lightcurve_encoder_q(lightcurves)
        if isinstance(lightcurve_feat, tuple):
            lightcurve_feat = lightcurve_feat[0]

        q_s, _ = self.shared_encoder_q(spectra_feat.unsqueeze(-1))
        q_l, _ = self.shared_encoder_q(lightcurve_feat.unsqueeze(-1))

        with torch.no_grad():
            k_s, _ = self.shared_encoder_k(spectra_feat.unsqueeze(-1))
            k_l, _ = self.shared_encoder_k(lightcurve_feat.unsqueeze(-1))

        loss_s, logits_s, labels = self.contrastive_loss(
            q_s, k_l, self.lightcurve_queue
        )

        self.lightcurve_queue, self.lightcurve_queue_ptr = self._dequeue_and_enqueue(
            k_l, self.lightcurve_queue, self.lightcurve_queue_ptr
        )

        if self.bidirectional:
            loss_l, logits_l, labels_l = self.contrastive_loss(
                q_l, k_s, self.spectra_queue
            )
            
            self.spectra_queue, self.spectra_queue_ptr = self._dequeue_and_enqueue(
                k_s, self.spectra_queue, self.spectra_queue_ptr
            )
            
            loss = (loss_s + loss_l) / 2
            logits = logits_s + logits_l
            q = q_l + q_s
            k = k_l + k_s
        else:
            loss = loss_s
            loss_l = None
            logits = logits_s
            q = q_l
            k = k_s

        return {
            'loss': loss,
            'logits': logits,
            'loss_s': loss_s,
            'loss_l': loss_l,
            'labels': labels,
            'q': q,
            'k': k
        }

