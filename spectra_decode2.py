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
from nn.cnn import CNNEncoderDecoder, CNNRegressor, MultiTaskRegressor
# from nn.mamba import MambaSeq2Seq
from nn.train import *
from nn.utils import deepnorm_init, load_checkpoints_ddp, load_scheduler
from nn.scheduler import WarmupScheduler
from util.utils import Container, plot_fit, plot_lr_schedule, kepler_collate_fn


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
torch.cuda.empty_cache()

models = {'CNNEncoderDecoder': CNNEncoderDecoder, 'AstroEncoderDecoder': AstroEncoderDecoder,
            'CNNRegressor': CNNRegressor, 'MultiTaskRegressor': MultiTaskRegressor}
          
# schedulers = {'WarmupScheduler': WarmupScheduler, 'OneCycleLR': OneCycleLR,
#  'CosineAnnealingLR': CosineAnnealingLR, 'none': None}

current_date = datetime.date.today().strftime("%Y-%m-%d")
datetime_dir = f"spec_decode2_{current_date}"

def test_dataset(dataset, num_iters=10):
    start_time = time.time()
    for i in range(num_iters):
        x_masked, x, mask, _, info, _ = dataset[i]
        if 'rv2' in info.keys():
            print(info['snrg'], info['snri'], info['snrr'], info['snrz'])
    print(f"Time taken for {num_iters} iterations: {time.time() - start_time:.2f} seconds." \
        f"avg per iteration: {(time.time() - start_time)/num_iters:.2f} seconds")


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
args_dir = '/data/lightSpec/nn/config_spectra_decode2.yaml'
data_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Data'])
exp_num = data_args.exp_num
model_name = data_args.model_name
if data_args.test_run:
    datetime_dir = f"test_{current_date}"
model_args = Container(**yaml.safe_load(open(args_dir, 'r'))[model_name])
conformer_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Conformer'])
optim_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Optimization'])
if not os.path.exists(f"{data_args.log_dir}/{datetime_dir}"):
    os.makedirs(f"{data_args.log_dir}/{datetime_dir}")

transforms = Compose([LAMOSTSpectrumPreprocessor(continuum_norm=False, plot_steps=False),
                        ToTensor(),
                         ])

lamost_catalog = pd.read_csv('/data/lamost/lamost_afgkm_teff_3000_7500_catalog.csv', sep='|')
lamost_catalog = lamost_catalog.drop_duplicates(subset=['combined_obsid'])
lamost_catalog = lamost_catalog[lamost_catalog['combined_snrg'] > 0]
lamost_catalog = lamost_catalog.dropna(subset=['combined_teff', 'combined_logg', 'combined_feh'])
print(lamost_catalog['combined_snrg'].describe())
plt.hist(lamost_catalog['combined_snrg'] / 1000, bins=100)
plt.savefig(f"/data/lightSpec/images/lamost_snrg_hist.png")
plt.close()
print("number of samples: ", len(lamost_catalog), lamost_catalog.head)
train_dataset = SpectraDataset(data_args.data_dir, transforms=transforms, df=lamost_catalog,
                                 max_len=int(data_args.max_len_spectra))
indices = list(range(len(train_dataset)))
train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
train_subset = Subset(train_dataset, train_indices)
val_subset = Subset(train_dataset, val_indices)

train_sampler = torch.utils.data.distributed.DistributedSampler(train_subset, num_replicas=world_size, rank=local_rank)
train_dataloader = DataLoader(train_subset,
                              batch_size=int(data_args.batch_size), \
                              num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
                              collate_fn=kepler_collate_fn,
                              sampler=train_sampler)


val_sampler = torch.utils.data.distributed.DistributedSampler(val_subset, num_replicas=world_size, rank=local_rank)
val_dataloader = DataLoader(val_subset,
                            batch_size=int(data_args.batch_size),
                            collate_fn=kepler_collate_fn,
                            sampler=val_sampler, \
                            num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]))

test_dataset(train_dataset, num_iters=10)

model = models[model_name](model_args, conformer_args=conformer_args)
model = model.to(local_rank)

print("model: ", model)

checkpoint_num = int(model_args.checkpoint_num)
if model_args.load_checkpoint:
    datetime_dir = os.path.basename(os.path.dirname(model_args.checkpoint_path))
    checkpoint_num = os.path.basename(model_args.checkpoint_path).split('.')[0].split('_')[-1]
    print(datetime_dir)
    print("loading checkpoint from: ", model_args.checkpoint_path)
    model = load_checkpoints_ddp(model, model_args.checkpoint_path, add_prefix=True)
    print("loaded checkpoint from: ", model_args.checkpoint_path)
else:
    deepnorm_init(model, model_args)

model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters: {num_params}")

loss_fn = torch.nn.L1Loss(reduction='none')
ssl_loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=float(optim_args.max_lr), weight_decay=float(optim_args.weight_decay))

config_save_path = f"{data_args.log_dir}/{datetime_dir}/{model_name}_spectra__decode_{checkpoint_num}_complete_config.yaml"

complete_config = {
    "model_name": model_name,
    "data_args": data_args.__dict__,
    "model_args": model_args.__dict__,
    "optim_args": optim_args.__dict__,
    "num_params": num_params,
    "model_structure": str(model),  # Add the model structure to the configuration
    "transforms": str(transforms)
}

# Save the complete configuration to a YAML file
with open(config_save_path, "w") as config_file:
    yaml.dump(complete_config, config_file, default_flow_style=False)

print(f"Configuration (with model structure) saved at {config_save_path}.")

trainer = MaskedRegressorTrainer(model=model, optimizer=optimizer,
                        criterion=loss_fn, ssl_criterion=ssl_loss_fn, output_dim=model_args.output_dim, scaler=None,
                       scheduler=None, train_dataloader=train_dataloader,
                       val_dataloader=val_dataloader, device=local_rank,
                           exp_num=datetime_dir, log_path=data_args.log_dir, range_update=None,
                           accumulation_step=1, max_iter=np.inf, w_name='snrg',
                           w_init_val=1,  exp_name=f"{model_name}_spectra_decode_{checkpoint_num}") 
fit_res = trainer.fit(num_epochs=data_args.num_epochs, device=local_rank,
                        early_stopping=40, only_p=False, best='acc', conf=True) 
output_filename = f'{data_args.log_dir}/{datetime_dir}/{model_name}_spectra__decode_{checkpoint_num}.json'
with open(output_filename, "w") as f:
    json.dump(fit_res, f, indent=2)
fig, axes = plot_fit(fit_res, legend=exp_num, train_test_overlay=True)
plt.savefig(f"{data_args.log_dir}/{datetime_dir}/fit_{model_name}_spectra__decode_{checkpoint_num}.png")
plt.clf()