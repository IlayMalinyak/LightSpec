import json
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import pandas as pd
from tqdm import tqdm
import numpy as np

import sys
from os import path
ROOT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print("running from ", ROOT_DIR)    

from dataset.dataset import *
from nn.train import *
from util.utils import *
from transforms.transforms import *

root_data_folder = "/data/lightPred/data"
b_size = 64
dur = 1440
cad = 30
DAY2MIN = 24*60
log_path = "/data/lightPred/logs/activity_proxy"



def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def compute_moments(x):
    """
    Compute standard deviation, third moment, and skewness for batched time series data.
    
    Args:
        x: PyTorch tensor with shape (batch_size, time_steps)
            where each row represents a separate time series
    
    Returns:
        std_dev: Standard deviation for each batch (shape: batch_size)
        third_moment: Raw third central moment for each batch (shape: batch_size)
        skewness: Standardized third moment (third_moment / std_dev^3) for each batch
    """
    # Calculate mean for each batch
    means = torch.mean(x, dim=1, keepdim=True)
    
    # Calculate centered data (x - μ)
    centered_data = x - means
    
    # Standard deviation calculation
    # Variance is the mean of squared deviations
    variance = torch.mean(centered_data**2, dim=1)
    std_dev = torch.sqrt(variance)
    
    # Third central moment calculation
    # E[(x - μ)³]
    third_moment_raw = torch.mean(centered_data**3, dim=1)
    
    # Standardized third moment (skewness)
    # skewness = E[(x - μ)³] / σ³
    skewness = third_moment_raw / (std_dev**3)
    
    return std_dev, skewness

def calculate_percentile_differences_batch(time_series, window_size, step_size):
    # Initialize a list to store the differences for each batch
    # differences = []

    num_windows = (time_series.size(1) - window_size) // step_size + 1
    
    # Use unfold to create sliding windows
    windows = time_series.unfold(dimension=1, size=window_size, step=step_size)


    p5 = torch.quantile(windows, 0.05, dim=2)
    p95 = torch.quantile(windows, 0.95, dim=2)

    # # Iterate over the time series with the given step size
    # for i in range(0, time_series.size(1) - segment_size + 1, step_size):
    #     segment = time_series[:, i:i + segment_size]
    #     p5 = torch.quantile(segment, 0.05, dim=1)
    #     p95 = torch.quantile(segment, 0.95, dim=1)
    #     diff = p95 - p5
    #     differences.append(diff)

    # Stack the differences into a single tensor
    return p95 - p5

def moving_window_std_pytorch(time_series, window_sizes, step_sizes):
    # time_series shape: (B, L)
    # window_sizes shape: (B,)
    # step_sizes shape: (B,)

    if len(time_series.shape) == 1:
        time_series = time_series.unsqueeze(0)

    device = time_series.device
    batch_size, seq_length = time_series.shape  # B, L
    window_sizes = window_sizes.to(device)
    step_sizes = step_sizes.to(device)

    # Calculate the number of windows for each sequence
    num_windows = ((seq_length - window_sizes).float() / step_sizes).floor().long() + 1  # shape: (B,)
    max_num_windows = num_windows.max().item()
    max_window_size = window_sizes.max().item()

    # Create indices for the start of each window
    # shape: (B, max_num_windows)
    window_starts = torch.arange(max_num_windows, device=device).unsqueeze(0) * step_sizes.unsqueeze(1)

    # Create a tensor of all possible window indices
    # shape: (B, max_num_windows, max_window_size)
    all_indices = window_starts.unsqueeze(2) + torch.arange(max_window_size, device=device).unsqueeze(0).unsqueeze(0)

    # Create a mask for valid indices
    # shape: (B, max_num_windows, max_window_size)
    valid_mask = (all_indices < seq_length) & (all_indices < (num_windows.unsqueeze(1) * step_sizes.unsqueeze(1)).unsqueeze(2))
    valid_mask = valid_mask & (all_indices < window_sizes.unsqueeze(1).unsqueeze(2))

    # Expand time_series and gather values
    # shape: (B, L) -> (B, 1, L) -> (B, max_num_windows, L)
    expanded_time_series = time_series.unsqueeze(1).expand(-1, max_num_windows, -1)
    
    # shape: (B, max_num_windows, max_window_size)
    gathered = expanded_time_series.gather(2, all_indices.clamp(0, seq_length - 1).long()).detach()

    # Calculate mean
    # shape: (B, max_num_windows)
    valid_count = valid_mask.float().sum(dim=2).clamp(min=1)
    mean = (gathered * valid_mask.float()).sum(dim=2) / valid_count

    # Calculate variance
    # shape: (B, max_num_windows)
    variance = ((gathered - mean.unsqueeze(2)) ** 2 * valid_mask.float()).sum(dim=2) / (valid_count - 1).clamp(min=1)
    # Calculate stand ard deviation
    # shape: (B, max_num_windows)
    std = torch.sqrt(variance)
    torch.cuda.empty_cache()

    return std.mean(dim=1)

def moving_window_std(time_series, window_size, step_size):
    # Get the number of windows
    num_windows = (time_series.size(1) - window_size) // step_size + 1
    
    # Use unfold to create sliding windows
    windows = time_series.unfold(dimension=1, size=window_size, step=step_size)
    
    # Calculate the standard deviation along the window dimension
    std_devs = windows.std(dim=2)
    
    return std_devs.mean(dim=1)

def moving_window_std_numpy(time_series, window_size, step_size):
    # Convert input to numpy array if it's not already
    time_series = np.asarray(time_series)
    
    # Ensure window_size and step_size are integers
    window_size = int(window_size)
    step_size = int(step_size)
    
    # Calculate the number of windows
    num_windows = max(0, (len(time_series) - window_size) // step_size + 1)
    
    # If there are no valid windows, return an empty array
    if num_windows == 0:
        return np.array([])
    
    # Create an array of indices for the start of each window
    start_indices = np.arange(num_windows) * step_size
    
    # Create a 2D array where each row is a window
    windows = np.lib.stride_tricks.as_strided(
        time_series,
        shape=(num_windows, window_size),
        strides=(time_series.strides[0] * step_size, time_series.strides[0])
    )
    
    # Calculate the standard deviation for each window
    return np.std(windows, axis=1)


if __name__ == "__main__":
    world_size    = int(os.environ["WORLD_SIZE"])
    rank          = int(os.environ["SLURM_PROCID"])
    jobid         = int(os.environ["SLURM_JOBID"])

    #gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    # gpus_per_node = 4
    gpus_per_node = torch.cuda.device_count()
    print('jobid ', jobid)
    print('gpus per node ', gpus_per_node)
    print(f"Hello from rank {rank} of {world_size} where there are" \
            f" {gpus_per_node} allocated GPUs per node.", flush=True)

    setup(rank, world_size)

    if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)
    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)
    num_workers = int(os.environ["SLURM_CPUS_PER_TASK"])
    print(f"rank: {rank}, local_rank: {local_rank}")

    print("logdir ", f'{log_path}')

    kepler_df = get_all_samples_df(num_qs=None)

    period_df = pd.read_csv('/data/lightPred/tables/kepler_predictions_clean.csv')

    berger_df = pd.read_csv('/data/lightPred/tables/berger_catalog_full.csv')

    kepler_meta = pd.read_csv('/data/lightPred/tables/kepler_dr25_meta_data.csv')


    kepler_df = kepler_df.merge(kepler_meta[['KID', 'KMAG']], on='KID', how='left').merge(period_df[['KID', 'predicted period']], on='KID', how='left').merge(berger_df, on='KID', how='left')

    kepler_df['kmag_abs'] = kepler_df['KMAG'] - 5 * np.log10(kepler_df['Dist']) + 5


    transforms = Compose([MovingAvg(13), Normalize('dist'), Normalize('mag'), ToTensor()]) 

    full_dataset = KeplerDataset(df=kepler_df, transforms=transforms,
                                target_transforms=None,
                                npy_path = '/data/lightPred/data/raw_npy',
                                seq_len=62000,
                                masked_transforms = False,
                                use_acf=False,
                                scale_flux=False,
                                labels=[],
                                )
    sampler = torch.utils.data.distributed.DistributedSampler(full_dataset, num_replicas=world_size, rank=rank)

    full_dataloader = DataLoader(full_dataset, batch_size=b_size, \
                                    num_workers=num_workers,
                                    collate_fn=kepler_collate_fn,
                                        pin_memory=True, sampler=sampler)

    print("len dataframes ", len(full_dataset), len(full_dataloader))

    teff = []
    kmag = []
    R = []
    logg = []
    kid = []
    r_var_vals = []
    s_ph = []
    pbar = tqdm(full_dataloader)
    for i, data in enumerate(pbar):
        x,_,y,_,info,_ = data
        x = x.squeeze(1)
        x = x.to(local_rank)
        ks = [i['KID'] for i in info]
        ps = torch.tensor([i['predicted period'] for i in info]).to(local_rank)
        ps = ps.nan_to_num(0)
        steps = ps*48
        ps = ps*5*48
        stds = moving_window_std_pytorch(x.squeeze(), ps, steps)
        s_ph.extend(stds.detach().cpu().tolist())
        kid.extend(ks)
        torch.cuda.empty_cache()
        # print(len(s_ph), len(kid))
        # print(len(s_ph))
        # for b in range(x.size(0)):
        #     p = info[b]['period']
        #     s = moving_window_std_numpy(x[b].squeeze().numpy(), p*5*48, p*48)
        #     print(p, s.mean())
        #     s_ph.append(s)
        r_var = calculate_percentile_differences_batch(x.squeeze(), 90*48, 30*48)
        r_var_range = torch.max(r_var, dim=1).values - torch.min(r_var, dim=1).values
        r_var = list(r_var_range.cpu().numpy())
        r_var_vals.extend(r_var)
        # teff.extend([i['Teff'] for i in info])
        # kmag.extend([i['kmag'] for i in info])
        # R.extend([i['R'] for i in info])
        # logg.extend([i['logg'] for i in info])
        # kid.extend([i['KID'] for i in info])
        # if i > 10:
        #     break

        
    res_df = pd.DataFrame({'KID': kid,  's_ph': s_ph, 'r_var': r_var_vals})
    res_df.to_csv('/data/logs/activity_proxies/proxies_full_dist_mag.csv', index=False)
    print(res_df.head())
