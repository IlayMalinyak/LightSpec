import torch
import torch.nn as nn
import torch.nn.functional as F

class AdversarialAlignment(nn.Module):
    def __init__(
        self, 
        spectra_encoder,  # pre-trained spectra encoder
        lightcurve_encoder,  # pre-trained light curve encoder
        shared_dim=64,   # Dimension of shared embedding space
        gen_dims=[1024, 128, 64],  # Hidden layer dimensions for generator
        disc_dims=[64, 128, 512],  # Hidden layer dimensions for discriminator
        freeze_encoders=True  # Freeze encoder parameters
    ):
        super(AdversarialAlignment, self).__init__()

        if freeze_encoders:
            self._freeze_encoder(spectra_encoder)
            self._freeze_encoder(lightcurve_encoder)
        
        self.spectra_encoder = spectra_encoder
        self.lightcurve_encoder = lightcurve_encoder
        
        # Generator Network: Creates shared embedding from two modalities
        self.generator = nn.Sequential(
            nn.Linear(gen_dims[0], gen_dims[1]),
            nn.BatchNorm1d(gen_dims[1]),
            nn.LeakyReLU(0.2),
            nn.Linear(gen_dims[1], gen_dims[2]),
            nn.BatchNorm1d(gen_dims[2]),
            nn.LeakyReLU(0.2),
            nn.Linear(gen_dims[2], shared_dim)
        )
        
        # Discriminator Network: Predicts similarity/proximity
        self.discriminator = nn.Sequential(
            nn.Linear(shared_dim, disc_dims[1]),
            nn.BatchNorm1d(disc_dims[1]),
            nn.LeakyReLU(0.2),
            nn.Linear(disc_dims[1], disc_dims[0]),
            nn.BatchNorm1d(disc_dims[0]),
            nn.LeakyReLU(0.2),
            nn.Linear(disc_dims[0], 1)  # Continuous output for proximity prediction
        )

    def _freeze_encoder(self, encoder):
        """Freeze encoder parameters"""
        for name, param in encoder.named_parameters():
            param.requires_grad = False
        
    def forward(self, spectra, lightcurves, w=None):
        """
        Forward pass for adversarial multimodal alignment
        Args:
            spectra: Spectra representations
            lightcurves: Lightcurve representations
            w: Proximity weights (optional)
        """
        spectra_embedding = self.spectra_encoder(spectra)
        lightcurve_embedding = self.lightcurve_encoder(lightcurves)
        # Concatenate inputs for generator
        print('spectra_embedding', spectra_embedding.shape, 'lightcurve_embedding', lightcurve_embedding.shape)
        combined_input = torch.cat([spectra_embedding, lightcurve_embedding], dim=1)
        print('combined_input', combined_input.shape)
        
        # Generate shared embedding
        shared_embedding = self.generator(combined_input)
        
        # Discriminator prediction of proximity
        proximity_pred = self.discriminator(shared_embedding)
        
        return {
            'shared_embedding': shared_embedding,
            'proximity_pred': proximity_pred
        }
    
    def generator_loss(self, spectra, lightcurves, w):
        """
        Generator loss that aims to fool the discriminator
        """
        # Forward pass to get shared embedding
        output = self.forward(spectra, lightcurves, w)
        shared_embedding = output['shared_embedding']
        proximity_pred = output['proximity_pred']
        
        batch_size = shared_embedding.size(0)
        positive_mask = torch.eye(batch_size, dtype=torch.bool, device=spectra.device)
        negative_mask = ~positive_mask
        
        # Split predictions and weights
        pos_pred = proximity_pred[positive_mask]
        neg_pred = proximity_pred[negative_mask]
        pos_w = w[positive_mask]
        neg_w = w[negative_mask]
        
        # Compute component losses
        pos_gen_loss = torch.mean(torch.log(pos_w) * pos_pred)
        neg_gen_loss = torch.mean(-torch.log(neg_w) * neg_pred)
        
        gen_loss = pos_gen_loss + neg_gen_loss
        
        return gen_loss
    
    def discriminator_loss(self, spectra, lightcurves, w):
        """
        Discriminator loss that tries to accurately predict proximity
        """
        # Forward pass to get shared embedding
        output = self.forward(spectra, lightcurves, w)
        proximity_pred = output['proximity_pred']
        
        # Compute component losses
        disc_loss = F.mse_loss(proximity_pred, torch.log(w))
                
        return disc_loss