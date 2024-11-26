import os
import gc
os.system('pip install -q wandb')
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.cuda.amp import GradScaler
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import wandb
import yaml
from tqdm import tqdm
import argparse


# Import your existing modules
from transforms.transforms import *
from dataset.dataset import LightSpecDataset
from nn.astroconf import Astroconformer, AstroEncoderDecoder
from nn.moco import MultimodalMoCo, LightCurveSpectraMoCo
from nn.utils import deepnorm_init
from util.utils import *
from nn.train import ContrastiveTrainer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_optimizer(config, parameters):
    """Create optimizer based on config parameters"""
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2)
        )
    elif config.optimizer == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=config.nesterov
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")

def setup_model(config, local_rank):
    """Setup model with given hyperparameters"""
    args_dir = '/data/lightSpec/nn/config_lightspec_ssl.yaml'
    model_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Astroconformer'])
    spec_model_args = Container(**yaml.safe_load(open(args_dir, 'r'))['AstroEncoderDecoder'])
    
    light_backbone = Astroconformer(model_args)
    light_backbone.pred_layer = torch.nn.Identity()
    spec_model = AstroEncoderDecoder(spec_model_args)
    
    model = LightCurveSpectraMoCo(
        spec_model.encoder, 
        light_backbone,
        projection_dim=config.projection_dim,
        K=config.K,
        m=config.m,
        T=config.T,
        freeze_lightcurve=config.freeze_lightcurve,
        freeze_spectra=config.freeze_spectra
    ).to(local_rank)
    
    return model

def setup_data():
    """Setup data loaders"""
    data_args = Container(**yaml.safe_load(open('/data/lightSpec/nn/config_lightspec_ssl.yaml', 'r'))['Data'])
    
    light_transforms = Compose([
        RandomCrop(int(data_args.max_days_lc/data_args.lc_freq)),
        Normalize('std'),
        ToTensor(),
    ])
    
    spec_transforms = Compose([
        MovingAvg(7),
        Normalize("minmax", axis=0),
        AvgDetrend(kernel_size=100),
        ToTensor(),
    ])
    
    kepler_df = get_all_samples_df(num_qs=None, read_from_csv=True)
    lamost_kepler_df = pd.read_csv('/data/lamost/lamost_dr8_gaia_dr3_kepler_ids.csv')
    lamost_kepler_df = lamost_kepler_df[~lamost_kepler_df['KID'].isna()]
    lamost_kepler_df['KID'] = lamost_kepler_df['KID'].astype(int)
    lamost_kepler_df = lamost_kepler_df.merge(kepler_df[['KID']], on='KID', how='inner')

    try:
        # Try to get a unique integer from the run name/ID
        seed = int(wandb.run.name.split('-')[-1]) if wandb.run.name else hash(wandb.run.id)
    except:
        # Fallback to a default seed
        seed = 42
    
    # Shuffle the DataFrame
    lamost_kepler_df = lamost_kepler_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Log dataset info to wandb
    wandb.log({
        "dataset_size": len(lamost_kepler_df),
        "unique_kids": lamost_kepler_df['KID'].nunique(),
        "shuffle_seed": seed
    })
    
    dataset = LightSpecDataset(
        df=lamost_kepler_df,
        light_transforms=light_transforms,
        spec_transforms=spec_transforms,
        npy_path='/data/lightPred/data/npy',
        spec_path=data_args.spectra_dir,
        light_seq_len=int(data_args.max_days_lc/data_args.lc_freq),
        spec_seq_len=int(data_args.max_len_spectra)
    )
    
    indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    
    train_loader = DataLoader(
        train_subset,
        batch_size=int(data_args.batch_size),
        num_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", None)),
        collate_fn=kepler_collate_fn,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=int(data_args.batch_size),
        num_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", None)),
        collate_fn=kepler_collate_fn,
        shuffle=True
    )
    
    return train_loader, val_loader

def cleanup_memory():
    """Clean up GPU memory"""
    torch.cuda.empty_cache()
    gc.collect()

def setup_ddp():
    """Setup DDP for distributed training"""
    world_size    = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return local_rank, world_size

def train():
    """Training function for W&B sweep"""
    try:
        cleanup_memory()
        
        # Initialize wandb
        wandb.init()

        local_rank=device
        
        # Setup model and data
        model = setup_model(wandb.config, device)
        train_loader, val_loader = setup_data()
        
        # Setup optimizer and loss
        optimizer = get_optimizer(
            wandb.config,
            filter(lambda p: p.requires_grad, model.parameters())
        )
        criterion = torch.nn.CrossEntropyLoss()

        trainer = ContrastiveTrainer(model=model, optimizer=optimizer,
                        criterion=criterion, output_dim=1, scaler=None,
                       scheduler=None, train_dataloader=train_loader,
                       val_dataloader=val_loader, device=local_rank,
                           exp_num=0, log_path='/data/lightSpec/logs/tune', range_update=None,
                           accumulation_step=1, max_iter=wandb.config.max_iterations, wandb_log=True,
                        exp_name="lightspec_ssl") 
        fit_res = trainer.fit(num_epochs=1, device=local_rank,
                                early_stopping=40, only_p=False, best='loss', conf=True) 
        # Log both detailed arrays and mean values
        wandb.log({
            # Mean values
            "mean_val_loss": np.mean(fit_res['val_loss']),
            "mean_train_loss": np.mean(fit_res['train_loss']),
            
            # Log as line plots
            "val_loss_curve": wandb.plot.line_series(
                xs=list(range(len(fit_res['val_loss']))),
                ys=[fit_res['val_loss']],
                keys=["Validation Loss"],
                title="Validation Loss Over Time",
                xname="Step"
            ),
            "train_loss_curve": wandb.plot.line_series(
                xs=list(range(len(fit_res['train_loss']))),
                ys=[fit_res['train_loss']],
                keys=["Training Loss"],
                title="Training Loss Over Time",
                xname="Step"
            ),
            
            # Store raw values for later analysis
            "val_loss_values": fit_res['val_loss'],
            "train_loss_values": fit_res['train_loss']
        })
        
        return np.mean(fit_res['val_loss'])
    
    finally:
        cleanup_memory()
        wandb.finish()

def main():
    # Load W&B API key
    key_path = '/data/lightSpec/keys/wandb.txt'
    with open(key_path, 'r') as file:
        api_key = file.read().strip()
    
    # Login to W&B
    wandb.login(key=api_key)
    
    # Define sweep configuration
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'loss',
            'goal': 'minimize'
        },
        'parameters': {
            'projection_dim': {
                'values': [64, 64, 512]
            },
            'K': {
                'values': [512, 1024, 2048]
            },
            'T': {
                'distribution': 'log_uniform',
                'min': np.log(0.01),
                'max': np.log(0.1)
            },
            'm': {
                'distribution': 'uniform',
                'min': 0.9,
                'max': 0.999
            },
            'lr': {
                'distribution': 'log_uniform',
                'min': np.log(1e-5),
                'max': np.log(1e-3)
            },
            'weight_decay': {
                'distribution': 'log_uniform',
                'min': np.log(1e-6),
                'max': np.log(1e-4)
            },
            # Optimizer selection and parameters
            'optimizer': {
                'values': ['adamw', 'sgd']
            },
            # Common optimizer parameters
            'lr': {
                'distribution': 'log_uniform',
                'min': np.log(1e-5),
                'max': np.log(1e-3)
            },
            'weight_decay': {
                'distribution': 'log_uniform',
                'min': np.log(1e-6),
                'max': np.log(1e-4)
            },
            # AdamW specific parameters
            'beta1': {
                'value': 0.9
            },
            'beta2': {
                'value': 0.999
            },
            # SGD specific parameters
            'momentum': {
                'distribution': 'uniform',
                'min': 0.8,
                'max': 0.99
            },
            'nesterov': {
                'values': [True, False]
            },
            'max_iterations': {
                'value': 1000
            },
            'freeze_lightcurve': {
                'values': [True, False]
            },
            'freeze_spectra': {
                'values': [True, False]
            }
        }
    }
    
    # Initialize sweep
    sweep_id = wandb.sweep(
        sweep_config,
        project="moco_spcetra_hyperparam_search"
    )
    
    # Start sweep
    wandb.agent(sweep_id, function=train, count=100)

if __name__ == "__main__":
    main()