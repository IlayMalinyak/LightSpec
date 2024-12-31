import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from collections import OrderedDict
from .Modules.mhsa_pro import MHA_rotary
from .Modules.cnn import ConvBlock
from nn.cnn import CNNEncoderDecoder, CNNEncoder
from nn.moco import MultimodalMoCo, LightCurveSpectraMoCo
from nn.simsiam import SimCLR, SimSiam, MultiModalSimSiam
from nn.astroconf import Astroconformer, AstroEncoderDecoder
from nn.scheduler import WarmupScheduler


models = {'Astroconformer': Astroconformer, 'CNNEncoder': CNNEncoder, 'SimCLR': SimCLR, 'SimSiam': SimSiam,
           'MultiModalSimSiam': MultiModalSimSiam, 'MultimodalMoCo': MultimodalMoCo,
             'LightCurveSpectraMoCo': LightCurveSpectraMoCo, 'AstroEncoderDecoder': AstroEncoderDecoder,
               'CNNEncoderDecoder': CNNEncoderDecoder,}

schedulers = {'WarmupScheduler': WarmupScheduler, 'OneCycleLR': OneCycleLR,
 'CosineAnnealingLR': CosineAnnealingLR, 'none': None}

def load_checkpoints_ddp(model, checkpoint_path, add_prefix=False):
  print(f"****Loading  checkpoint - {checkpoint_path}****")
  state_dict = torch.load(f'{checkpoint_path}', map_location=torch.device('cpu'))
  new_state_dict = OrderedDict()
  for key, value in state_dict.items():
      while key.startswith('module.'):
          key = key[7:]
      if add_prefix:
        if key.startswith('backbone.') or key.startswith('pe.') or key.startswith('encoder.'):
            key = key = 'encoder.' + key
      new_state_dict[key] = value
  state_dict = new_state_dict
  missing, unexpected = model.load_state_dict(state_dict, strict=False)
  print("number of keys in state dict and model: ", len(state_dict), len(model.state_dict()))
  print("number of missing keys: ", len(missing))
  print("number of unexpected keys: ", len(unexpected))
  print("missing keys: ", missing)
  print("unexpected keys: ", unexpected)
  return model

def init_model(model, model_args):
  if model_args.load_checkpoint:
      model = load_checkpoints_ddp(model, model_args.checkpoint_path)
  else:
      print("****deepnorm init****")
      deepnorm_init(model, model_args)
  return model

def deepnorm_init(model, args):

  def init_func(m):
    beta = getattr(args, 'beta', 1)
    if isinstance(m, MHA_rotary):  # adjust as necessary for your use case
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
        

  model.apply(init_func)


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
