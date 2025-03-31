import torch
import torch.nn as nn
import torch.nn.functional as F

def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception



class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=64):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.output_dim = out_dim
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        self.num_layers = 3
    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        # print("projection_MLP: ", x.shape)
        if isinstance(x, tuple):
            x = x[0]
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x 


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=64, hidden_dim=32, out_dim=64): # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 

def weighted_info_nce_loss(features, sample_properties, t):
    # Calculate cosine similarity
    cos_sim = F.cosine_similarity(features[:, None, :], features[None, :, :], dim=-1)
    # print("cos_sim: ", cos_sim.shape, cos_sim.max(), cos_sim.min())
    
    # Mask out cosine similarity to itself
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
    cos_sim.masked_fill_(self_mask, -9e15)
    
    # Find positive pairs 
    batch_size = cos_sim.shape[0] // 2
    pos_mask = torch.zeros_like(cos_sim, dtype=torch.bool)
    pos_mask[torch.arange(batch_size), torch.arange(batch_size) + batch_size] = True
    pos_mask[torch.arange(batch_size) + batch_size, torch.arange(batch_size)] = True
    
    # Compute L2 distance between sample properties
    prop_distances = torch.cdist(sample_properties, sample_properties, p=2.0)
    # print("prop_distances: ", prop_distances.shape, prop_distances.max(), prop_distances.min())
    
    # Normalize distances to use as weights
    distance_weights = (prop_distances - prop_distances.min()) / (prop_distances.max() - prop_distances.min())
    # distance_weights = prop_distances 
    
    # Create a mask for negative pairs (non-diagonal, non-positive elements)
    negative_mask = ~(self_mask | pos_mask)
    
    # Apply temperature scaling
    cos_sim = cos_sim / t
    
    # Create a copy of cos_sim to modify for weighted loss
    weighted_cos_sim = cos_sim.clone()
    
    # Scale negative similarities by their distance weights
    weighted_cos_sim[negative_mask] *= (1 + distance_weights[negative_mask])
    
    # Compute negative log-likelihood with weighted similarities
    nll = -cos_sim[pos_mask] + torch.logsumexp(weighted_cos_sim, dim=-1)
    nll = nll.mean()
    
    return nll

def info_nce_loss(features, t):
    # Calculate cosine similarity
    cos_sim = F.cosine_similarity(features[:, None, :], features[None, :, :], dim=-1)
    # Mask out cosine similarity to itself
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
    cos_sim.masked_fill_(self_mask, -9e15)
    # Find positive example -> batch_size//2 away from the original example
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
    # InfoNCE loss
    cos_sim = cos_sim / t
    print("info nce loss cos_sim: ", cos_sim.shape)
    nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
    nll = nll.mean()
    return nll

class SimCLR(nn.Module):
    def __init__(self, backbone, args):
        super(SimCLR, self).__init__()
        self.backbone = backbone
        self.projector = projection_MLP(backbone.output_idm,
                                        hidden_dim=args.hidden_dim, out_dim=args.hidden_dim)
        self.predictor = prediction_MLP(in_dim=args.hidden_dim,
                                        hidden_dim=args.hidden_dim//2, out_dim=args.hidden_dim)
    def forward(self, x, temperature=1.0):
        z = self.backbone(x)
        p = self.projector(z)
        p = self.predictor(p)
        # print("p: ", p.shape, "z: ", z.shape)
        L = info_nce_loss(p, t=temperature)
        return {'loss': L, 'features': z, 'predictions': p}

class SimSiam(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        
        self.backbone = backbone
        self.projector = projection_MLP(backbone.output_dim, backbone.output_dim // 4, backbone.output_dim // 4 )

        self.encoder = nn.Sequential( # f encoder
            self.backbone,
            self.projector
        )
        self.encoder.output_dim = backbone.output_dim // 4
        self.output_dim = backbone.output_dim // 4
        self.predictor = prediction_MLP(backbone.output_dim // 4, backbone.output_dim // 4, backbone.output_dim // 4)
    
    def forward(self, x1, x2):
        f, h = self.encoder, self.predictor
        z1, z2 = f(x1), f(x2)
        p1, p2 = h(z1), h(z2)
        L = D(p1, z2) / 2 + D(p2, z1) / 2
        return {'loss': L, 'z1': z1, 'z2': z2}

class MultiModalSimCLR(nn.Module):
    def __init__(self, backbone,
                  lightcurve_backbone,
                    spectra_backbone,
                    args):
        super().__init__()
        
        self.lightcurve_backbone = lightcurve_backbone
        self.spectra_backbone = spectra_backbone
        if args.freeze_backbone:
            self.__freeze_backbone()
        self.backbone = backbone
        self.projector = projection_MLP(backbone.output_dim,
                                        hidden_dim=args.hidden_dim, out_dim=args.hidden_dim)
        self.predictor = prediction_MLP(in_dim=args.hidden_dim,
                                        hidden_dim=args.hidden_dim//2, out_dim=args.hidden_dim)
        # print all trainable parameters
    
    def __freeze_backbone(self):
        for param in self.lightcurve_backbone.parameters():
            param.requires_grad = False
        for param in self.spectra_backbone.parameters():
            param.requires_grad = False
    
    def forward(self, lightcurve, spectra, w, temperature=1.0):
        x1 = self.lightcurve_backbone(lightcurve)
        x2 = self.spectra_backbone(spectra)
        x = torch.cat((x1, x2), dim=0)
        w = torch.cat((w, w), dim=0)
        z = self.backbone(x)
        p = self.projector(z)
        p = self.predictor(p)
        # print("p: ", p.shape, "z: ", z.shape)
        L = weighted_info_nce_loss(p, w, t=temperature)
        return {'loss': L, 'features': z, 'predictions': p}

class MultiModalSimSiam(nn.Module):
    def __init__(self, backbone,
                  lightcurve_backbone,
                    spectra_backbone,
                    args):
        super().__init__()
        
        self.lightcurve_backbone = lightcurve_backbone
        self.spectra_backbone = spectra_backbone
        if args.freeze_backbone:
            self.__freeze_backbone()
        self.backbone = backbone
        self.projector = projection_MLP(backbone.output_dim, hidden_dim=args.hidden_dim, out_dim=args.projection_dim)

        self.encoder = nn.Sequential( # f encoder
            self.backbone,
            self.projector
        )
        self.predictor = prediction_MLP(in_dim=args.projection_dim, hidden_dim=args.projection_dim //2 ,
                                     out_dim=args.output_dim)
    
    def __freeze_backbone(self):
        for param in self.lightcurve_backbone.parameters():
            param.requires_grad = False
        for param in self.spectra_backbone.parameters():
            param.requires_grad = False
    
    def forward(self, lightcurve, spectra, w=None):
        x1 = self.lightcurve_backbone(lightcurve)
        if isinstance(x1, tuple):
            x1 = x1[0]
        x2 = self.spectra_backbone(spectra)
        if isinstance(x2, tuple):
            x2 = x2[0]
        f, h = self.encoder, self.predictor
        z1, z2 = f(x1), f(x2)
        p1, p2 = h(z1), h(z2)
        L = D(p1, z2) / 2 + D(p2, z1) / 2
        return {'loss': L, 'logits1': z1, 'logits2': z2}
