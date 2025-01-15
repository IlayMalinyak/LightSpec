# code from https://github.com/facebookresearch/moco/blob/main/moco/builder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from nn.simsiam import projection_MLP
import time
import numpy as np



class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, encoder_q, encoder_k, dim=128, K=1024, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
            )

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.criterion = nn.CrossEntropyLoss()

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

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

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        loss = self.criterion(logits, labels)

        return {'loss': loss, 'logits': logits, 'labels': labels}


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
    

class MultimodalMoCo(nn.Module):
    def __init__(
        self,
        spectra_encoder,  # pre-trained spectra encoder
        lightcurve_encoder,  # pre-trained light curve encoder
        projection_dim=32,  # Final projection dimension
        hidden_dim=512,  # Hidden dimension of projection MLP
        num_layers=2,
        K=65536,  # Queue size
        m=0.999,  # Momentum coefficient
        T=0.07,  # Temperature
        freeze_lightcurve=True,  # Whether to freeze light curve encoder
        freeze_spectra=True,  # Whether to freeze spectra encoder
        bidirectional=True,  # Whether to train in both directions
        transformer=False
    ):
        super(MultimodalMoCo, self).__init__()
        
        self.K = K
        self.m = m
        self.T = T
        self.bidirectional = bidirectional
        self.criterion = nn.CrossEntropyLoss()
        
        # Freeze pre-trained encoders (optional, can be fine-tuned)
        if freeze_lightcurve:
            self._freeze_encoder(spectra_encoder)
        if freeze_spectra:
            self._freeze_encoder(lightcurve_encoder)
        
        # Query path: Add projection heads to pre-trained encoders
        self.spectra_encoder_q = spectra_encoder
        self.lightcurve_encoder_q = lightcurve_encoder
        
        # Get the output dimensions of your pre-trained encoders
        with torch.no_grad():
            # You'll need to adjust these based on your encoder architectures
            spectra_out_dim = spectra_encoder.output_dim
            lightcurve_out_dim = lightcurve_encoder.output_dim
        
        # Projection heads for query encoders
        # self.spectra_proj_q = projection_MLP(spectra_out_dim, hidden_dim=hidden_dim, out_dim=projection_dim)
        # self.lightcurve_proj_q = projection_MLP(lightcurve_out_dim, hidden_dim=hidden_dim, out_dim=projection_dim)
        self.spectra_proj_q = self._build_projector(spectra_out_dim, hidden_dim,
                                                     projection_dim, num_layers,
                                                      transformer=transformer)
        self.lightcurve_proj_q = self._build_projector(lightcurve_out_dim, hidden_dim,
                                                         projection_dim, num_layers,
                                                          transformer=transformer)
        
        # Key path: Create momentum encoders and projectors
        self.spectra_encoder_k = copy.deepcopy(spectra_encoder)
        self.lightcurve_encoder_k = copy.deepcopy(lightcurve_encoder)
        self.spectra_proj_k = copy.deepcopy(self.spectra_proj_q)
        self.lightcurve_proj_k = copy.deepcopy(self.lightcurve_proj_q)
        
        # Freeze key encoders and projectors
        self._freeze_encoder(self.spectra_encoder_k)
        self._freeze_encoder(self.lightcurve_encoder_k)
        self._freeze_encoder(self.spectra_proj_k)
        self._freeze_encoder(self.lightcurve_proj_k)
        
        # Initialize queues for both directions if bidirectional
        self.register_buffer("lightcurve_queue", torch.randn(projection_dim, K))
        self.lightcurve_queue = F.normalize(self.lightcurve_queue, dim=0)
        self.register_buffer("lightcurve_queue_ptr", torch.zeros(1, dtype=torch.long))
        
        if bidirectional:
            self.register_buffer("spectra_queue", torch.randn(projection_dim, K))
            self.spectra_queue = F.normalize(self.spectra_queue, dim=0)
            self.register_buffer("spectra_queue_ptr", torch.zeros(1, dtype=torch.long))

    def weighted_contrastive_loss(self, q, k, sample_properties):
        # Normalize query and key vectors
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        
        # Compute similarity matrix
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T

        if torch.isnan(logits).sum() > 0:
            print("nans in logits: ", torch.isnan(logits).sum())
        
        # Compute L2 distance between sample properties
        # Assumes sample_properties is a tensor of shape (batch_size, n_features)
        if sample_properties is None:
            distance_weights = torch.zeros(logits.shape[0], logits.shape[0], device=logits.device)
        else:
            prop_distances = torch.cdist(sample_properties, sample_properties, p=2.0)
            distance_weights = (prop_distances - prop_distances.min()) / (prop_distances.max() - prop_distances.min())
        
        # Normalize distances to use as weights
        # You might want to adjust this normalization strategy
        
        if torch.isnan(distance_weights).sum() > 0:
            print("nans in distance weights: ", torch.isnan(distance_weights).sum())
        
        # Create a mask for negative pairs (non-diagonal elements)
        N = logits.shape[0]
        positive_mask = torch.eye(N, dtype=torch.bool, device=logits.device)
        negative_mask = ~positive_mask
        
        # Apply weights to negative pairs
        weighted_logits = logits.clone()
        weighted_logits[negative_mask] *= (1 + distance_weights[negative_mask])
        
        # Prepare labels (positive pair on diagonal)
        labels = torch.arange(N, dtype=torch.long, device=logits.device)

        if torch.isnan(labels).sum() > 0:
            print("nans in labels: ", torch.isnan(labels).sum())
        
        if torch.isnan(weighted_logits).sum() > 0:
            print("nans in weighted logits: ", torch.isnan(weighted_logits).sum())
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(weighted_logits, labels)

        if torch.isnan(loss).sum() > 0:
            print("nans in loss: ", torch.isnan(loss).sum())
        
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
        # Update spectra encoder and projector
        self._momentum_update_encoder(
            self.spectra_encoder_q, self.spectra_encoder_k,
            self.spectra_proj_q, self.spectra_proj_k
        )
        # Update lightcurve encoder and projector
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
        
        # Check if we'll exceed queue size
        if ptr + batch_size > self.K:
            # Split the operation into two parts
            first_part = self.K - ptr  # number of slots until end of queue
            queue[:, ptr:] = keys.T[:, :first_part]  # fill until end
            remaining = batch_size - first_part
            if remaining > 0:  # if we still have keys to insert
                queue[:, :remaining] = keys.T[:, first_part:]  # wrap around to start
            ptr = remaining if remaining > 0 else 0
        else:
            # Normal case - enough space in queue
            queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K  # move pointer
        
        queue_ptr[0] = ptr
        return queue, queue_ptr
    
    def forward(self,lightcurves, spectra, w=None, w_threshold=0.5):
        """
        Forward pass computing contrastive loss in both directions
        Args:
            spectra: batch of spectra
            lightcurves: batch of light curves
        Returns:
            losses and logits for both directions
        """
        start = time.time()
        spectra = self.spectra_encoder_q(spectra)
        if isinstance(spectra, tuple):
            spectra = spectra[0]
        lightcurves = self.lightcurve_encoder_q(lightcurves)
        if isinstance(lightcurves, tuple):
            lightcurves = lightcurves[0]


        if torch.isnan(spectra).sum() > 0:
            print("nans in spectra: ", torch.isnan(spectra).sum())
        if torch.isnan(lightcurves).sum() > 0:
            print("nans in lightcurves: ", torch.isnan(lightcurves).sum())

        q_s = self.spectra_proj_q(spectra)
        q_l = self.lightcurve_proj_q(lightcurves)

        if torch.isnan(q_s).sum() > 0:
            print("nans in q_s: ", torch.isnan(q_s).sum())
        if torch.isnan(q_l).sum() > 0:
            print("nans in q_l: ", torch.isnan(q_l).sum())
        
        # Normalize features
        q_s = F.normalize(q_s, dim=1)
        q_l = F.normalize(q_l, dim=1)

        if torch.isnan(q_s).sum() > 0:
            print("nans in q_s after norm: ", torch.isnan(q_s).sum())
        if torch.isnan(q_l).sum() > 0:
            print("nans in q_l after norm: ", torch.isnan(q_l).sum())
        
        # Compute key features

        with torch.no_grad():
            self._momentum_update()
            spectra = self.spectra_encoder_k(spectra)
            if isinstance(spectra, tuple):
                spectra = spectra[0]
            lightcurves = self.lightcurve_encoder_k(lightcurves)
            if isinstance(lightcurves, tuple):
                lightcurves = lightcurves[0]
            k_s = self.spectra_proj_k(spectra)
            k_l = self.lightcurve_proj_k(lightcurves)
            
            k_s = F.normalize(k_s, dim=1)
            k_l = F.normalize(k_l, dim=1)
        
        # Compute logits for spectra->lightcurve direction
        # l_pos_s = torch.einsum('nc,nc->n', [q_s, k_l]).unsqueeze(-1)
        # l_neg_s = torch.einsum('nc,ck->nk', [q_s, self.lightcurve_queue.clone().detach()])
        # logits_s = torch.cat([l_pos_s, l_neg_s], dim=1)
        # logits_s /= self.T

        # logits_time = time.time()- start - keys_time
        # print("time taken for logits: ", logits_time)
        
        # Update spectra->lightcurve queue
        self.lightcurve_queue, self.lightcurve_queue_ptr = self._dequeue_and_enqueue(
            k_l, self.lightcurve_queue, self.lightcurve_queue_ptr
        )
        # enqueue_time = time.time()- start - logits_time
        # print("time taken for enqueue: ", enqueue_time)

        loss_s, logits_s, labels = self.weighted_contrastive_loss(q_l, k_s, w)            
    
        if self.bidirectional:
            loss_l, logits_l, labels_l = self.weighted_contrastive_loss(q_s, k_l, w) 
            loss = (loss_s + loss_l) / 2
            logits = logits_s + logits_l
            q = q_l + q_s
            k = k_l + k_s
        else:
            loss = loss_s
            logits = logits_s
            q = q_l
            k = k_s
        
        return {'loss': loss, 'logits': logits , 'labels': labels, 'q': q, 'k': k}

