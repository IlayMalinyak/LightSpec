import os
# os.system("pip install mamba-ssm[causal-conv1d]")
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Subset
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from astropy.io import fits
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl
import yaml
import json
from collections import OrderedDict
import datetime

import sys
from os import path
ROOT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print("running from ", ROOT_DIR) 

from transforms.transforms import *
from dataset.dataset import SpectraDataset
from nn.astroconf import AstroEncoderDecoder
from nn.models import CNNEncoderDecoder, CNNRegressor, MultiTaskRegressor
# from nn.mamba import MambaSeq2Seq
from nn.train import *
from nn.utils import deepnorm_init, load_checkpoints_ddp, load_scheduler
from nn.scheduler import WarmupScheduler
from nn.optim import QuantileLoss, CQR
from util.utils import Container, plot_fit, plot_lr_schedule, kepler_collate_fn, save_predictions_to_dataframe
from features import create_umap
import generator



os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
torch.cuda.empty_cache()

models = {'CNNEncoderDecoder': CNNEncoderDecoder, 'AstroEncoderDecoder': AstroEncoderDecoder,
            'CNNRegressor': CNNRegressor, 'MultiTaskRegressor': MultiTaskRegressor}

prediction_labels = ['teff', 'logg', 'feh']
          
# schedulers = {'WarmupScheduler': WarmupScheduler, 'OneCycleLR': OneCycleLR,
#  'CosineAnnealingLR': CosineAnnealingLR, 'none': None}

current_date = datetime.date.today().strftime("%Y-%m-%d")
datetime_dir = f"spec_decode2_{current_date}"

def plot_joint_prob(lamost_catalog):
    # Create a multi-panel figure to show different views of the distribution
    fig = plt.figure(figsize=(18, 15))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    # 1. Pairwise 2D plots - using scatter plots with color indicating the joint probability
    ax1 = fig.add_subplot(2, 2, 1)
    scatter = ax1.scatter(lamost_catalog['combined_teff'], 
                          lamost_catalog['combined_logg'], 
                          c=lamost_catalog['joint_prob'], 
                          s=3, alpha=0.7, 
                          cmap='viridis_r')  # viridis_r to have rare (low prob) be yellow
    ax1.set_xlabel('Effective Temperature (K)')
    ax1.set_ylabel('Surface Gravity (log g)')
    ax1.set_title('Joint Probability: Teff vs logg')
    plt.colorbar(scatter, ax=ax1, label='Probability')
    
    ax2 = fig.add_subplot(2, 2, 2)
    scatter = ax2.scatter(lamost_catalog['combined_teff'], 
                          lamost_catalog['combined_feh'], 
                          c=lamost_catalog['joint_prob'], 
                          s=3, alpha=0.7, 
                          cmap='viridis_r')
    ax2.set_xlabel('Effective Temperature (K)')
    ax2.set_ylabel('Metallicity [Fe/H]')
    ax2.set_title('Joint Probability: Teff vs [Fe/H]')
    plt.colorbar(scatter, ax=ax2, label='Probability')
    
    ax3 = fig.add_subplot(2, 2, 3)
    scatter = ax3.scatter(lamost_catalog['combined_logg'], 
                          lamost_catalog['combined_feh'], 
                          c=lamost_catalog['joint_prob'], 
                          s=3, alpha=0.7, 
                          cmap='viridis_r')
    ax3.set_xlabel('Surface Gravity (log g)')
    ax3.set_ylabel('Metallicity [Fe/H]')
    ax3.set_title('Joint Probability: logg vs [Fe/H]')
    plt.colorbar(scatter, ax=ax3, label='Probability')
    
    # 2. 3D scatter plot with color indicating probability
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    scatter = ax4.scatter(lamost_catalog['combined_teff'], 
                          lamost_catalog['combined_logg'], 
                          lamost_catalog['combined_feh'], 
                          c=lamost_catalog['joint_prob'], 
                          s=3, alpha=0.7, 
                          cmap='viridis_r')
    ax4.set_xlabel('Teff (K)')
    ax4.set_ylabel('log g')
    ax4.set_zlabel('[Fe/H]')
    ax4.set_title('3D Joint Probability Distribution')
    
    plt.suptitle('Joint Probability Distribution of Stellar Parameters', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    plt.savefig('/data/lightSpec/images/lamost_joint_prob.png', dpi=300, bbox_inches='tight')
    print("Joint probability visualization saved to /data/lightSpec/images/lamost_joint_prob.png")
    
    # Create a second visualization showing the weights (inverse probabilities)
    fig2 = plt.figure(figsize=(18, 15))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    ax1 = fig2.add_subplot(2, 2, 1)
    scatter = ax1.scatter(lamost_catalog['combined_teff'], 
                          lamost_catalog['combined_logg'], 
                          c=lamost_catalog['joint_weight'], 
                          s=3, alpha=0.7, 
                          cmap='plasma',
                          norm=mpl.colors.LogNorm())  # Log scale for weights
    ax1.set_xlabel('Effective Temperature (K)')
    ax1.set_ylabel('Surface Gravity (log g)')
    ax1.set_title('Joint Weights: Teff vs logg')
    plt.colorbar(scatter, ax=ax1, label='Weight (1/prob)')
    
    ax2 = fig2.add_subplot(2, 2, 2)
    scatter = ax2.scatter(lamost_catalog['combined_teff'], 
                          lamost_catalog['combined_feh'], 
                          c=lamost_catalog['joint_weight'], 
                          s=3, alpha=0.7, 
                          cmap='plasma',
                          norm=mpl.colors.LogNorm())
    ax2.set_xlabel('Effective Temperature (K)')
    ax2.set_ylabel('Metallicity [Fe/H]')
    ax2.set_title('Joint Weights: Teff vs [Fe/H]')
    plt.colorbar(scatter, ax=ax2, label='Weight (1/prob)')
    
    ax3 = fig2.add_subplot(2, 2, 3)
    scatter = ax3.scatter(lamost_catalog['combined_logg'], 
                          lamost_catalog['combined_feh'], 
                          c=lamost_catalog['joint_weight'], 
                          s=3, alpha=0.7, 
                          cmap='plasma',
                          norm=mpl.colors.LogNorm())
    ax3.set_xlabel('Surface Gravity (log g)')
    ax3.set_ylabel('Metallicity [Fe/H]')
    ax3.set_title('Joint Weights: logg vs [Fe/H]')
    plt.colorbar(scatter, ax=ax3, label='Weight (1/prob)')
    
    # 3D scatter with weights
    ax4 = fig2.add_subplot(2, 2, 4, projection='3d')
    scatter = ax4.scatter(lamost_catalog['combined_teff'], 
                          lamost_catalog['combined_logg'], 
                          lamost_catalog['combined_feh'], 
                          c=lamost_catalog['joint_weight'], 
                          s=3, alpha=0.7, 
                          cmap='plasma',
                          norm=mpl.colors.LogNorm())
    ax4.set_xlabel('Teff (K)')
    ax4.set_ylabel('log g')
    ax4.set_zlabel('[Fe/H]')
    ax4.set_title('3D Joint Weight Distribution')
    
    plt.suptitle('Joint Weight Distribution of Stellar Parameters', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    plt.savefig('/data/lightSpec/images/lamost_joint_weight.png', dpi=300, bbox_inches='tight')
    print("Joint weight visualization saved to /data/lightSpec/images/lamost_joint_weight.png")
    
    # Also create a 2D histogram of the most important pairwise relationships
    fig3, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Teff vs logg
    h = axes[0].hist2d(lamost_catalog['combined_teff'], 
                  lamost_catalog['combined_logg'], 
                  bins=(20, 20), 
                  cmap='viridis')
    axes[0].set_xlabel('Effective Temperature (K)')
    axes[0].set_ylabel('Surface Gravity (log g)')
    axes[0].set_title('Density: Teff vs logg')
    plt.colorbar(h[3], ax=axes[0], label='Count')
    
    # Teff vs FeH
    h = axes[1].hist2d(lamost_catalog['combined_teff'], 
                  lamost_catalog['combined_feh'], 
                  bins=(20, 20), 
                  cmap='viridis')
    axes[1].set_xlabel('Effective Temperature (K)')
    axes[1].set_ylabel('Metallicity [Fe/H]')
    axes[1].set_title('Density: Teff vs [Fe/H]')
    plt.colorbar(h[3], ax=axes[1], label='Count')
    
    # logg vs FeH
    h = axes[2].hist2d(lamost_catalog['combined_logg'], 
                  lamost_catalog['combined_feh'], 
                  bins=(20, 20), 
                  cmap='viridis')
    axes[2].set_xlabel('Surface Gravity (log g)')
    axes[2].set_ylabel('Metallicity [Fe/H]')
    axes[2].set_title('Density: logg vs [Fe/H]')
    plt.colorbar(h[3], ax=axes[2], label='Count')
    
    plt.suptitle('2D Histograms of Stellar Parameters', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig('/data/lightSpec/images/lamost_2d_histograms.png', dpi=300, bbox_inches='tight')
    print("2D histograms saved to /data/lightSpec/images/lamost_2d_histograms.png")
    
    plt.close('all')  # Close all figures to free memory

def test_dataset_samples(dataset, num_iters=10):
    start_time = time.time()
    for i in range(num_iters):
        x_masked, x, y, mask, info, _ = dataset[i]
        print('y: ', len(y))
        if 'rv2' in info.keys():
            print(info['snrg'], info['snri'], info['snrr'], info['snrz'])
    print(f"Time taken for {num_iters} iterations: {time.time() - start_time:.2f} seconds." \
        f"avg per iteration: {(time.time() - start_time)/num_iters:.2f} seconds")

def create_train_test_dfs(meta_columns):
    lamost_catalog = pd.read_csv('/data/lamost/lamost_afgkm_teff_3000_7500_catalog_updated.csv')
    lamost_catalog = lamost_catalog.drop_duplicates(subset=['combined_obsid'])
    lamost_catalog = lamost_catalog[lamost_catalog['combined_snrg'] > 0]
    lamost_catalog = lamost_catalog.dropna(subset=['combined_teff', 'combined_logg', 'combined_feh'])
    print("values ranges: ")
    for c in ['combined_teff', 'combined_logg', 'combined_feh']:
        print(c, lamost_catalog[c].min(), lamost_catalog[c].max())
    
    # Calculate joint probability distribution for target variables
    # First, bin the continuous values to create discrete categories
    num_bins = 200  # Adjust number of bins as needed
    
    # Create bins for each variable
    teff_bins = pd.cut(lamost_catalog['combined_teff'], bins=num_bins, labels=False)
    logg_bins = pd.cut(lamost_catalog['combined_logg'], bins=num_bins, labels=False)
    feh_bins = pd.cut(lamost_catalog['combined_feh'], bins=num_bins, labels=False)
    
    # Combine bins to create joint categories
    lamost_catalog['joint_bin'] = teff_bins.astype(str) + '_' + logg_bins.astype(str) + '_' + feh_bins.astype(str)
    
    # Calculate joint probability
    joint_counts = lamost_catalog['joint_bin'].value_counts()
    total_samples = len(lamost_catalog)
    joint_probs = joint_counts / total_samples
    
    # Assign probabilities back to the dataframe
    lamost_catalog['joint_prob'] = lamost_catalog['joint_bin'].map(joint_probs)
    
    # Calculate inverse probability for weighting (rare combinations get higher weights)
    lamost_catalog['joint_weight'] = 1.0 / lamost_catalog['joint_prob']
    
    # Normalize weights to have mean = 1.0
    mean_weight = lamost_catalog['joint_weight'].mean()
    lamost_catalog['joint_weight'] = lamost_catalog['joint_weight'] / mean_weight
    
    # Print some statistics about the joint distribution
    print(f"Number of unique joint bins: {len(joint_probs)}")
    print(f"Min joint probability: {lamost_catalog['joint_prob'].min():.6f}")
    print(f"Max joint probability: {lamost_catalog['joint_prob'].max():.6f}")
    print(f"Min weight: {lamost_catalog['joint_weight'].min():.2f}")
    print(f"Max weight: {lamost_catalog['joint_weight'].max():.2f}")

    
    train_df, test_df = train_test_split(lamost_catalog, test_size=0.2, random_state=1234)
    return train_df, test_df

def setup():
    world_size    = int(os.environ["WORLD_SIZE"])
    rank          = int(os.environ["SLURM_PROCID"])
    jobid         = int(os.environ["SLURM_JOBID"])
    gpus_per_node = torch.cuda.device_count()
    print('jobid ', jobid)
    print('gpus per node ', gpus_per_node)
    print(f"Hello from rank {rank} of {world_size} where there are" \
          f" {gpus_per_node} allocated GPUs per node. ", flush=True)

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)
    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)
    print(f"rank: {rank}, local_rank: {local_rank}")
    return local_rank, world_size, gpus_per_node


local_rank, world_size, gpus_per_node = setup()
args_dir = '/data/lightSpec/nn/full_config.yaml'
data_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Data'])
exp_num = data_args.exp_num
model_name = data_args.spec_model_name
os.makedirs(f"{data_args.log_dir}/{datetime_dir}", exist_ok=True)

# if data_args.test_run:
#     datetime_dir = f"test_{current_date}"


train_dataset, val_dataset, test_dataset, complete_config = generator.get_data(data_args,
 data_generation_fn=create_train_test_dfs, dataset_name='Spectra')


train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
train_dataloader = DataLoader(train_dataset,
                              batch_size=int(data_args.batch_size) * 6, \
                              num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
                              collate_fn=kepler_collate_fn,
                              sampler=train_sampler)


val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank)
val_dataloader = DataLoader(val_dataset,
                            batch_size=int(data_args.batch_size) * 8,
                            collate_fn=kepler_collate_fn,
                            sampler=val_sampler, \
                            num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]))

test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=world_size, rank=local_rank)
test_dataloader = DataLoader(test_dataset,
                            batch_size=int(data_args.batch_size) * 8,
                            collate_fn=kepler_collate_fn,
                            sampler=test_sampler, \
                            num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]))

_, optim_args, complete_config, _, model = generator.get_model(data_args, args_dir, complete_config, local_rank)

model = model.to(local_rank)

model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters: {num_params}")

if data_args.create_umap:
    umap_df = create_umap(model.module.encoder, val_dataloader, local_rank, use_w=False, dual=False, max_iter=800)
    print("umap created: ", umap_df.shape)
    umap_df.to_csv(f"{data_args.log_dir}/{datetime_dir}/umap_{exp_num}_ssl.csv", index=False)
    print(f"umap saved at {data_args.log_dir}/{datetime_dir}/umap_{exp_num}_ssl.csv")
    exit()
# loss_fn = torch.nn.L1Loss(reduction='none')
loss_fn = CQR(quantiles=optim_args.quantiles, reduction='none')
ssl_loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=float(optim_args.max_lr), weight_decay=float(optim_args.weight_decay))

trainer = MaskedRegressorTrainer(model=model, optimizer=optimizer,
                        criterion=loss_fn, ssl_criterion=ssl_loss_fn,
                         output_dim=len(data_args.prediction_labels_spec), scaler=None,
                       scheduler=None, train_dataloader=train_dataloader,
                       val_dataloader=val_dataloader, device=local_rank, num_quantiles=len(optim_args.quantiles),
                           exp_num=datetime_dir, log_path=data_args.log_dir, range_update=None,
                                 accumulation_step=1, max_iter=np.inf, w_name='snrg',
                           w_init_val=1,  exp_name=f"spectra_decode_{exp_num}") 

complete_config.update(
    {"trainer": trainer.__dict__,
    "loss_fn": str(loss_fn),
    "optimizer": str(optimizer)}
)
   
config_save_path = f"{data_args.log_dir}/{datetime_dir}/spec_decode_{exp_num}_complete_config.yaml"
with open(config_save_path, "w") as config_file:
    json.dump(complete_config, config_file, indent=2, default=str)

# print(f"Configuration (with model structure) saved at {config_save_path}.")

fit_res = trainer.fit(num_epochs=data_args.num_epochs, device=local_rank,
                       early_stopping=40, best='loss', conf=True) 
output_filename = f'{data_args.log_dir}/{datetime_dir}/{model_name}_spectra_decode_{exp_num}.json'
with open(output_filename, "w") as f:
    json.dump(fit_res, f, indent=2)
fig, axes = plot_fit(fit_res, legend=exp_num, train_test_overlay=True)
plt.savefig(f"{data_args.log_dir}/{datetime_dir}/fit_{model_name}_spectra_decode_{exp_num}.png")
plt.clf()


preds_val, targets_val, info = trainer.predict(val_dataloader, device=local_rank)

preds, targets, info = trainer.predict(test_dataloader, device=local_rank)

low_q = preds[:, :, 0] 
high_q = preds[:, :, -1]
coverage = np.mean((targets >= low_q) & (targets <= high_q))
print('coverage: ', coverage)

cqr_errs = loss_fn.calibrate(preds_val, targets_val)
print(targets.shape, preds.shape)
preds_cqr = loss_fn.predict(preds, cqr_errs)

low_q = preds_cqr[:, :, 0]
high_q = preds_cqr[:, :, -1]
coverage = np.mean((targets >= low_q) & (targets <= high_q))
print('coverage after calibration: ', coverage)
df = save_predictions_to_dataframe(preds, targets, info, prediction_labels, optim_args.quantiles)
df.to_csv(f"{data_args.log_dir}/{datetime_dir}/{model_name}_spectra_decode2_{exp_num}.csv", index=False)
print('predictions saved in', f"{data_args.log_dir}/{datetime_dir}/{model_name}_spectra_decode2_{exp_num}.csv") 
df_cqr = save_predictions_to_dataframe(preds_cqr, targets, info, prediction_labels, optim_args.quantiles)
df_cqr.to_csv(f"{data_args.log_dir}/{datetime_dir}/{model_name}_spectra_decode2_{exp_num}_cqr.csv", index=False)
print('predictions saved in', f"{data_args.log_dir}/{datetime_dir}/{model_name}_spectra_decode2_{exp_num}_cqr.csv")

umap_df = create_umap(model.module.encoder, test_dataloader, local_rank, use_w=False, dual=False)
print("umap created: ", umap_df.shape)
umap_df.to_csv(f"{data_args.log_dir}/{datetime_dir}/umap_{exp_num}_ssl.csv", index=False)
print(f"umap saved at {data_args.log_dir}/{datetime_dir}/umap_{exp_num}_ssl.csv")
exit(0)
