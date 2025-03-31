        # Check for NaN in loss earl0
# code modified from https://github.com/facebookresearch/moco/blob/main/moco/builder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from nn.simsiam import projection_MLP, prediction_MLP
from nn.models import Transformer
from nn.Modules.conformer import ConformerEncoder
import time
import numpy as np
from collections import deque
from copy import deepcopy
import torch.distributed as dist




class MultimodalMoCo(nn.Module):
    """
    Multimodal MoCo model with shared encoder for light curves and spectra.
    """
    def __init__(
        self,
        spectra_encoder,  # pre-trained spectra encoder
        lightcurve_encoder,  # pre-trained light curve encoder
        projection_args,
        combined_encoder=None,
        projection_dim=128,  # Final projection dimension
        hidden_dim=512,  # Hidden dimension of projection MLP
        num_layers=8,
        shared_dim=128,
        K=65536,  # Queue size
        m=0.999,  # Momentum coefficient
        T=0.07,  # Temperature
        freeze_lightcurve=True,  # Whether to freeze light curve encoder
        freeze_spectra=True,  # Whether to freeze spectra encoder
        freeze_combined=True, # Whether to freeze combined encoder
        bidirectional=True,  # Whether to train in both directions
        transformer=False,
        calc_loss=True
    ):
        super(MultimodalMoCo, self).__init__()
        
        self.K = K
        self.m = m
        self.T = T
        self.shared_dim = shared_dim
        self.bidirectional = bidirectional
        self.criterion = nn.CrossEntropyLoss()
        self.calc_loss = calc_loss

        if freeze_lightcurve:
            self._freeze_encoder(lightcurve_encoder)
        if freeze_spectra:
            self._freeze_encoder(spectra_encoder)
        self.combined_encoder = combined_encoder
        if self.combined_encoder is not None and freeze_combined:
            self._freeze_encoder(self.combined_encoder)

        self.spectra_encoder_q = spectra_encoder
        self.lightcurve_encoder_q = lightcurve_encoder
        
        # with torch.no_grad():
        #     spectra_out_dim = spectra_encoder.output_dim
        #     lightcurve_out_dim = lightcurve_encoder.output_dim
       
        # self.spectra_proj_q = nn.Linear(spectra_encoder.output_dim, projection_dim)
        # self.lightcurve_proj_q = nn.Linear(lightcurve_encoder.output_dim // 2, projection_dim)
        
        self.shared_encoder_q = Transformer(projection_args)
        
        self.shared_encoder_k = copy.deepcopy(self.shared_encoder_q)
        
        # self.spectra_proj_k = copy.deepcopy(self.spectra_proj_q)
        # self.lightcurve_proj_k = copy.deepcopy(self.lightcurve_proj_q)
        
        self._freeze_encoder(self.shared_encoder_k)
        
        # self._freeze_encoder(self.spectra_proj_k)
        # self._freeze_encoder(self.lightcurve_proj_k)

        self.register_buffer("lightcurve_queue", torch.randn(projection_dim, K))
        self.lightcurve_queue = F.normalize(self.lightcurve_queue, dim=0)
        self.register_buffer("lightcurve_queue_ptr", torch.zeros(1, dtype=torch.long))
        
        if bidirectional:
            self.register_buffer("spectra_queue", torch.randn(projection_dim, K))
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
        if self.combined_encoder is not None:
            spectra = torch.nn.functional.pad(spectra, (0, lightcurves.shape[-1] - spectra.shape[-1], 0,0))
            combined_input = torch.cat((lightcurves, spectra.unsqueeze(1)),dim=1)
            combined_embed = self.combined_encoder(combined_input)
            spectra_feat = torch.cat((spectra_feat, combined_embed),dim=-1)
            lightcurve_feat = torch.cat((lightcurve_feat, combined_embed), dim=-1)
        q_s, _ = self.shared_encoder_q(spectra_feat.unsqueeze(-1))
        q_l, _ = self.shared_encoder_q(lightcurve_feat.unsqueeze(-1))

        # q_s = self.spectra_proj_q(spectra_feat)
        # q_l = self.lightcurve_proj_q(lightcurve_feat)

        if not self.calc_loss:
            return {
                    'q': torch.cat((q_l, q_s),dim=-1)
                    }

        with torch.no_grad():
            # k_s, _ = self.shared_encoder_k(spectra_feat.unsqueeze(-1))
            # k_l, _ = self.shared_encoder_k(lightcurve_feat.unsqueeze(-1))

            k_s = self.spectra_proj_k(spectra_feat)
            k_l = self.lightcurve_proj_k(lightcurve_feat)

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

class PredictiveMoco(MultimodalMoCo):
    """
    Predictive MoCo model with shared encoder for light curves and spectra.
    """
    def __init__(
        self,
        spectra_encoder,  # pre-trained spectra encoder
        lightcurve_encoder,  # pre-trained light curve encoder
        projection_args,
        predictor_args,
        loss_args,
        **kwargs
    ):
        super(PredictiveMoco, self).__init__(
            spectra_encoder, lightcurve_encoder, projection_args, **kwargs
        )
        
        self.vicreg_predictor = Predictor(**predictor_args)
        # self.vicreg_predictor = Transformer(projection_args)
        predictor_args['w_dim'] = 0
        # self.moco_predictor = Predictor(**predictor_args)
        self.loss_args = loss_args
        self.pool = nn.AdaptiveAvgPool1d(self.shared_dim)
    
    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def vicreg_loss(self, x,y):
        # https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py#L239

        repr_loss = F.mse_loss(x, y)

        x = torch.cat(FullGatherLayer.apply(x), dim=0)
        y = torch.cat(FullGatherLayer.apply(y), dim=0)
        batch_size, num_features = x.shape
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2


        cov_x = (x.T @ x) / (batch_size - 1)
        cov_y = (y.T @ y) / (batch_size - 1)
        cov_loss = self.off_diagonal(cov_x).pow_(2).sum().div(
            num_features
        ) + self.off_diagonal(cov_y).pow_(2).sum().div(num_features)
        loss = (
            self.loss_args.sim_coeff * repr_loss
            + self.loss_args.std_coeff * std_loss
            + self.loss_args.cov_coeff * cov_loss
        )
        loss = loss.nan_to_num(0)
        return loss
    
    def forward(self, lightcurves, spectra, w=None, pred_coeff=1):
 
         spectra_feat = self.spectra_encoder_q(spectra)
         if isinstance(spectra_feat, tuple):
             spectra_feat = spectra_feat[0]
         lightcurve_feat = self.lightcurve_encoder_q(lightcurves)
         if isinstance(lightcurve_feat, tuple):
             lightcurve_feat = lightcurve_feat[0]
 
         if self.combined_encoder is not None:
             spectra = torch.nn.functional.pad(spectra, (0, lightcurves.shape[-1] - spectra.shape[-1], 0,0))
             combined_input = torch.cat((lightcurves, spectra.unsqueeze(1)),dim=1)
             combined_embed = self.combined_encoder(combined_input)
             spectra_feat = torch.cat((spectra_feat, combined_embed),dim=-1)
             lightcurve_feat = torch.cat((lightcurve_feat, combined_embed), dim=-1)
 
         q_s, _ = self.shared_encoder_q(spectra_feat.unsqueeze(-1))
         q_l, _ = self.shared_encoder_q(lightcurve_feat.unsqueeze(-1))
 
        #  q_s = self.spectra_proj_q(spectra_feat)
        #  q_l = self.lightcurve_proj_q(lightcurve_feat)
 
         q_s = q_s.nan_to_num(0)
         q_l = q_l.nan_to_num(0)
         if not self.calc_loss:
             return {
                     'q': torch.cat((q_l, q_s),dim=-1)
                     }
 
         q_s_vicreg = self.vicreg_predictor(q_s, w=w)
         q_l_vicreg = self.vicreg_predictor(q_l, w=w)
        #  q_s_moco = self.moco_predictor(q_s)
        #  q_l_moco = self.moco_predictor(q_l)
 
         with torch.no_grad():
             k_s = self.shared_encoder_k(spectra_feat.unsqueeze(-1))[0]
             k_l = self.shared_encoder_k(lightcurve_feat.unsqueeze(-1))[0]
 
            #  k_s = self.spectra_proj_k(spectra_feat)
            #  k_l = self.lightcurve_proj_k(lightcurve_feat)
 
         loss_s_pred = self.vicreg_loss(q_s_vicreg, q_l_vicreg)
 
         loss_s, logits_s, labels = self.contrastive_loss(
             q_s, k_l, self.lightcurve_queue
         )
        #  print("loss_s", loss_s, "loss_s_pred", loss_s_pred)
         cont_loss = loss_s
 
         loss_s = pred_coeff * loss_s_pred + (1 - pred_coeff) * loss_s
 
 
         self.lightcurve_queue, self.lightcurve_queue_ptr = self._dequeue_and_enqueue(
             k_l, self.lightcurve_queue, self.lightcurve_queue_ptr
         )
 
         if self.bidirectional:
             loss_l_pred = self.vicreg_loss(q_l_vicreg, q_s_vicreg)
             loss_l, logits_l, labels_l = self.contrastive_loss(
                 q_l, k_s, self.spectra_queue
             )
            #  print("loss_l", loss_l, "loss_l_pred", loss_l_pred)
 
             cont_loss = (cont_loss + loss_l) / 2
 
             loss_l = pred_coeff * loss_l_pred + (1 - pred_coeff) * loss_l
 
             self.spectra_queue, self.spectra_queue_ptr = self._dequeue_and_enqueue(
                 k_s, self.spectra_queue, self.spectra_queue_ptr
             )
 
             loss = (loss_s + loss_l) / 2
             loss_pred = (loss_s_pred + loss_l_pred) / 2
             logits = torch.cat((logits_l, logits_s), dim=-1)
             q = torch.cat((q_l, q_s), dim=-1)
             k = torch.cat((k_l, k_s), dim=-1)
         else:
             loss = loss_s
             loss_l = None
             logits = logits_s
             loss_pred = loss_s_pred
             q = q_l
             k = k_s
         loss = loss.nan_to_num(0)
         return {
             'loss': loss,
             'logits': logits,
             'loss_pred': loss_pred,
             'loss_contrastive': cont_loss,
             'loss_s': loss_s,
             'loss_l': loss_l,
             'loss_l_pred': loss_l_pred,
             'loss_s_pred': loss_s_pred,
             'labels': labels,
             'q': q,
             'k': k
         }


class MultiTaskMoCo(nn.Module):
    """
    Multitask MoCo model with shared encoder for light curves and spectra.
    """
    def __init__(
        self,
        moco_model,
        predictor_args,
    ):
        super(MultiTaskMoCo, self).__init__()
        self.moco_model = moco_model
        self.predictor = Predictor(**predictor_args)
    def forward(self, lightcurves, spectra, w=None, pred_coeff=1):
        moco_out = self.moco_model(lightcurves, spectra,
                                      w=w, pred_coeff=pred_coeff)
        features = moco_out['q']
        # features = torch.cat([moco_out['lightcurve_feat'], moco_out['spectra_feat']], dim=-1)
        preds = self.predictor(features)
        moco_out['preds'] = preds
        return moco_out
       

class Predictor(nn.Module):
    """
    Predictive model for MoCo.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, w_dim=0):
        super(Predictor, self).__init__()
        in_dim += w_dim
        
        # Mark the first layer for proper initialization
        self.input_layer = nn.Linear(in_dim, hidden_dim)
        self.input_layer.is_first_layer = True
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.activation1 = nn.SiLU()
        
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.activation2 = nn.SiLU()
        
        self.output_layer = nn.Linear(hidden_dim, out_dim)
        
        self.predictor = nn.Sequential(
                self.input_layer,
                self.norm1,
                self.activation1,
                self.hidden_layer,
                self.norm2,
                self.activation2,
                self.output_layer
            )
    
    def forward(self, x, w=None):
        if w is not None:
            w = w.nan_to_num(0)
            x = torch.cat((x, w), dim=1)
        return self.predictor(x)

class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]

class MocoTuner(nn.Module):
    """
    class to fine tune a moco model
    """
    def __init__(self, moco_model, tune_args, freeze_moco=False):
        super(MocoTuner, self).__init__()
        self.moco_model = moco_model
        if freeze_moco:
            for name, parameter in self.moco_model.named_parameters():
                parameter.requires_grad = False
        self.tune_args = tune_args
        # self.predictor = Predictor(**tune_args)
        self.pred_layer = nn.Sequential(
        nn.Linear(tune_args['in_dim'], tune_args['hidden_dim']),
        nn.GELU(),
        nn.Dropout(p=0.3),
        nn.Linear(tune_args['hidden_dim'], tune_args['out_dim'])
        )


    def forward(self, lightcurves, spectra, w=None, w_tune=None, pred_coeff=1):
        moco_out = self.moco_model(lightcurves, spectra, w=w, pred_coeff=pred_coeff)
        q = moco_out['q']
        preds = self.pred_layer(q)
        return preds

