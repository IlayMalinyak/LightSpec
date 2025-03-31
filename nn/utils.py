import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from collections import OrderedDict
from .Modules.mhsa_pro import MHA_rotary
from .Modules.cnn import ConvBlock
from nn.models import *
from nn.moco import MultimodalMoCo
from nn.simsiam import SimCLR, SimSiam, MultiModalSimSiam
from nn.astroconf import Astroconformer, AstroEncoderDecoder
from nn.scheduler import WarmupScheduler
from util.utils import Container
import yaml
import os
os.system('pip install torchinfo')
import torchinfo

models = {'Astroconformer': Astroconformer, 'CNNEncoder': CNNEncoder, 'SimCLR': SimCLR, 'SimSiam': SimSiam,
           'MultiModalSimSiam': MultiModalSimSiam, 'MultimodalMoCo': MultimodalMoCo,
             'AstroEncoderDecoder': AstroEncoderDecoder,
               'CNNEncoderDecoder': CNNEncoderDecoder,}

schedulers = {'WarmupScheduler': WarmupScheduler, 'OneCycleLR': OneCycleLR,
 'CosineAnnealingLR': CosineAnnealingLR, 'none': None}

def load_checkpoints_ddp(model, checkpoint_path, prefix='', load_backbone=False):
  print(f"****Loading  checkpoint - {checkpoint_path}****")
  state_dict = torch.load(f'{checkpoint_path}', map_location=torch.device('cpu'))
  new_state_dict = OrderedDict()
  for key, value in state_dict.items():
    # print(key)
    while key.startswith('module.'):
        key = key[7:]
    if load_backbone:
        if key.startswith('backbone.'):
            key = key[9:]
        else:
            continue
    key = prefix + key
    # print(key, value.shape)
    new_state_dict[key] = value
  state_dict = new_state_dict
  
  missing, unexpected = model.load_state_dict(state_dict, strict=False)
  print("number of keys in state dict and model: ", len(state_dict), len(model.state_dict()))
  print("number of missing keys: ", len(missing))
  print("number of unexpected keys: ", len(unexpected))
  print("missing keys: ", missing)
  print("unexpected keys: ", unexpected)
  return model

def compare_model_architectures(model1, model2):
    """
    Compare two PyTorch model architectures
    
    Args:
        model1 (torch.nn.Module): First PyTorch model
        model2 (torch.nn.Module): Second PyTorch model
    """
    print("Model 1 Architecture:")
    print(model1)
    print("\n" + "="*50 + "\n")
    
    print("Model 2 Architecture:")
    print(model2)
    print("\n" + "="*50 + "\n")
    
    # Detailed summary using torchinfo
    print("Detailed Model 1 Summary:")
    torchinfo.summary(model1, input_size=(1, input_shape))
    print("\n" + "="*50 + "\n")
    
    print("Detailed Model 2 Summary:")
    torchinfo.summary(model2, input_size=(1, input_shape))
    
    # Compare number of parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\nParameter Comparison:")
    print(f"Model 1 Total Trainable Parameters: {count_parameters(model1):,}")
    print(f"Model 2 Total Trainable Parameters: {count_parameters(model2):,}")
    
    # Compare layer types
    def get_layer_types(model):
        layer_types = {}
        for name, module in model.named_modules():
            layer_type = type(module).__name__
            layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
        return layer_types
    
    print("\nLayer Type Comparison:")
    print("Model 1 Layer Types:")
    for layer, count in get_layer_types(model1).items():
        print(f"  {layer}: {count}")
    
    print("\nModel 2 Layer Types:")
    for layer, count in get_layer_types(model2).items():
        print(f"  {layer}: {count}")

def init_model(model, model_args, prefix='', load_backbone=False):
  if model_args.load_checkpoint:
        model = load_checkpoints_ddp(model, model_args.checkpoint_path, prefix=prefix, load_backbone=load_backbone)
  else:
      print("****applying deepnorm initialization****")
      # Apply DeepNorm initialization for transformer-based models
      if hasattr(model, 'encoder') and any(
          'transformer' in str(type(module)).lower() 
          or 'conformer' in str(type(module)).lower()
          or 'mhsa' in str(type(module)).lower() 
          for name, module in model.named_modules()):
          deepnorm_init(model, model_args)
  return model

def deepnorm_init(model, args):
  """
  Apply DeepNorm initialization to transformer-based models.
  This helps with stability and training of deep transformer networks.
  
  Args:
      model (nn.Module): Model to initialize
      args: Configuration arguments with beta parameter
  """
  # Handle shared encoders in MoCo models (Transformer instances)
  from nn.models import Transformer
  if isinstance(model, Transformer):
    print(f"Applying DeepNorm initialization to Transformer with {len(model.layers)} layers")
    beta = getattr(args, 'beta', 1)
    
    # Initialize encoder (first layer)
    if hasattr(model, 'encoder'):
      nn.init.xavier_normal_(model.encoder.weight, gain=1)
      if model.encoder.bias is not None:
        nn.init.zeros_(model.encoder.bias)
        
    # Initialize all attention blocks
    for i, layer in enumerate(model.layers):
      if hasattr(layer, 'attn'):
        # Initialize attention components
        if hasattr(layer.attn, 'query_proj'):
          nn.init.xavier_normal_(layer.attn.query_proj.weight, gain=1)
          nn.init.xavier_normal_(layer.attn.key_proj.weight, gain=1)
          nn.init.xavier_normal_(layer.attn.value_proj.weight, gain=beta)
          nn.init.xavier_normal_(layer.attn.output_proj.weight, gain=beta)
          
          if hasattr(layer.attn.query_proj, 'bias') and layer.attn.query_proj.bias is not None:
            nn.init.zeros_(layer.attn.query_proj.bias)
            nn.init.zeros_(layer.attn.key_proj.bias)
            nn.init.zeros_(layer.attn.value_proj.bias)
            nn.init.zeros_(layer.attn.output_proj.bias)
      
      # Initialize FFN components
      if hasattr(layer, 'ffn'):
        nn.init.xavier_normal_(layer.ffn.linear1.weight, gain=beta)
        nn.init.xavier_normal_(layer.ffn.linear2.weight, gain=beta)
        if hasattr(layer.ffn.linear1, 'bias') and layer.ffn.linear1.bias is not None:
          nn.init.zeros_(layer.ffn.linear1.bias)
          nn.init.zeros_(layer.ffn.linear2.bias)
    
    # Initialize output projection
    if hasattr(model, 'head'):
      if hasattr(model.head, 'linear1'):
        nn.init.xavier_normal_(model.head.linear1.weight, gain=beta)
      if hasattr(model.head, 'linear2'):
        nn.init.xavier_normal_(model.head.linear2.weight, gain=beta)
    
    return
  
  # For other model types, use the apply method
  def init_func(m):
    beta = getattr(args, 'beta', 1)
    
    # Handle MHA_rotary for Conformer/Astroconformer models
    if isinstance(m, MHA_rotary):
      nn.init.xavier_normal_(m.query.weight, gain=1)
      nn.init.xavier_normal_(m.key.weight, gain=1)
      nn.init.xavier_normal_(m.value.weight, gain=beta)
      nn.init.xavier_normal_(m.output.weight, gain=beta)

      nn.init.zeros_(m.query.bias)
      nn.init.zeros_(m.key.bias)
      nn.init.zeros_(m.value.bias)
      nn.init.zeros_(m.output.bias)
      if getattr(m, 'ffn', None) is not None:
        nn.init.xavier_normal_(m.ffn.linear1.weight, gain=beta)
        nn.init.xavier_normal_(m.ffn.linear2.weight, gain=beta)
        nn.init.zeros_(m.ffn.linear1.bias)
        nn.init.zeros_(m.ffn.linear2.bias)
    
    # Handle Flash_Mha for regular Transformer models
    elif hasattr(m, 'attn') and hasattr(m.attn, 'query_proj'):
      nn.init.xavier_normal_(m.attn.query_proj.weight, gain=1)
      nn.init.xavier_normal_(m.attn.key_proj.weight, gain=1)
      nn.init.xavier_normal_(m.attn.value_proj.weight, gain=beta)
      nn.init.xavier_normal_(m.attn.output_proj.weight, gain=beta)
      
      if hasattr(m.attn.query_proj, 'bias') and m.attn.query_proj.bias is not None:
        nn.init.zeros_(m.attn.query_proj.bias)
        nn.init.zeros_(m.attn.key_proj.bias)
        nn.init.zeros_(m.attn.value_proj.bias)
        nn.init.zeros_(m.attn.output_proj.bias)
    
    # Handle Block's FFN
    elif hasattr(m, 'ffn') and hasattr(m.ffn, 'linear1'):
      nn.init.xavier_normal_(m.ffn.linear1.weight, gain=beta)
      nn.init.xavier_normal_(m.ffn.linear2.weight, gain=beta)
      if hasattr(m.ffn.linear1, 'bias') and m.ffn.linear1.bias is not None:
        nn.init.zeros_(m.ffn.linear1.bias)
        nn.init.zeros_(m.ffn.linear2.bias)
    
    # Handle general MLP layers
    elif isinstance(m, nn.Linear):
      # For the first layer in the sequence
      if getattr(m, 'is_first_layer', False):
        nn.init.xavier_normal_(m.weight, gain=1)
      else:
        nn.init.xavier_normal_(m.weight, gain=beta)
      if m.bias is not None:
        nn.init.zeros_(m.bias)

  model.apply(init_func)


def get_lightPred_model(seq_len):
    args = Container(**yaml.safe_load(open(f'/data/lightPred/Astroconf/default_config.yaml', 'r')))
    args.load_dict(yaml.safe_load(open(f'/data/lightPred/Astroconf/model_config.yaml', 'r'))[args.model])
    args.output_dim = 1
    if args.deepnorm and args.num_layers >= 10:
        layer_coeff = args.num_layers/5.0
        args.alpha, args.beta = layer_coeff**(0.5), layer_coeff**(-0.5)
    
    model = Astroconformer(args)
    deepnorm_init(model, args)
    lstm_params = {
        'dropout': 0.35,
        'hidden_size': 64,
        'image': False,
        'in_channels': 1,
        'kernel_size': 4,
        'num_classes': 2,
        'num_layers': 5,
        'seq_len': seq_len,
        'stride': 4}
    model.pred_layer = nn.Identity()
    model = LSTM_DUAL_LEGACY(model, encoder_dims=args.encoder_dim, lstm_args=lstm_params, num_classes=2)
    return model


def load_scheduler(optimizer, train_dataloader, world_size, optim_args, data_args):
    """
    Dynamically load and configure a learning rate scheduler.
    
    Args:
        optimizer (torch.optim.Optimizer): The optimizer to apply the scheduler to
        train_dataloader (torch.utils.data.DataLoader): Training dataloader for calculating steps
        world_size (int): Number of distributed processes
        optim_args (Container): Optimization arguments from configuration
        data_args (Container): Data arguments from configuration
    
    Returns:
        torch.optim.lr_scheduler._LRScheduler or None: Configured scheduler
    """
    schedulers = {
        'OneCycleLR':OneCycleLR,
        'CosineAnnealingLR': CosineAnnealingLR, 
        'WarmupScheduler': WarmupScheduler,  # Assuming this is defined elsewhere
        'none': None
    }
    
    # If no scheduler specified, return None
    if optim_args.scheduler == 'none':
        return None
    
    try:
        # Get the scheduler class
        scheduler_class = schedulers.get(optim_args.scheduler)
        if scheduler_class is None:
            print(f"Warning: Scheduler {optim_args.scheduler} not found.")
            return None
        
        # Create a copy of scheduler arguments
        scheduler_args = dict(optim_args.scheduler_args.get(optim_args.scheduler, {}))
        
        # Convert string values to appropriate numeric types
        numeric_keys = [
            'max_lr', 'epochs', 'steps_per_epoch', 'pct_start', 
            'base_momentum', 'max_momentum', 'div_factor', 'final_div_factor',
             'eta_min', 'T_max'
        ]
        
        for key in numeric_keys:
            if key in scheduler_args:
                try:
                    scheduler_args[key] = float(scheduler_args[key])
                except (ValueError, TypeError):
                    print(f"Warning: Could not convert {key} to float")
        
        # Always set the optimizer
        scheduler_args['optimizer'] = optimizer
        
        # Dynamically adjust steps_per_epoch if needed
        if 'steps_per_epoch' in scheduler_args:
            scheduler_args['steps_per_epoch'] = len(train_dataloader) * world_size
        
        # Dynamically adjust epochs
        if 'epochs' in scheduler_args:
            scheduler_args['epochs'] = int(data_args.num_epochs)
        
        # For OneCycleLR, ensure required arguments are present
        if optim_args.scheduler == 'OneCycleLR':
            if 'max_lr' not in scheduler_args:
                scheduler_args['max_lr'] = float(optim_args.max_lr)
            if 'steps_per_epoch' not in scheduler_args:
                scheduler_args['steps_per_epoch'] = len(train_dataloader) * world_size
            if 'epochs' not in scheduler_args:
                scheduler_args['epochs'] = int(data_args.num_epochs)
        elif optim_args.scheduler == 'CosineAnnealingLR':
            if 'T_max' not in scheduler_args:
                scheduler_args['T_max'] = int(len(train_dataloader) * world_size)
        
        # Create the scheduler
        scheduler = scheduler_class(**scheduler_args)
        print(f"Scheduler {optim_args.scheduler} initialized successfully")
        return scheduler
    
    except Exception as e:
        print(f"Error initializing scheduler {optim_args.scheduler}: {e}")
        return None

