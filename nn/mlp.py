
import torch
import torch.nn as nn
import torch.nn.functional as F


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