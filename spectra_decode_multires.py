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
from nn.models import CNNEncoderDecoder, CNNRegressor, MultiTaskRegressor, MultiResRegressor, Transformer
from nn.optim import QuantileLoss, CQR
# from nn.mamba import MambaSeq2Seq
from nn.train import *
from nn.utils import deepnorm_init, load_checkpoints_ddp, load_scheduler
from nn.scheduler import WarmupScheduler
from util.utils import Container, plot_fit, plot_lr_schedule, kepler_collate_fn, save_predictions_to_dataframe
from features import create_umap



os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
torch.cuda.empty_cache()

models = {'CNNEncoderDecoder': CNNEncoderDecoder, 'AstroEncoderDecoder': AstroEncoderDecoder,
            'CNNRegressor': CNNRegressor, 'MultiTaskRegressor': MultiTaskRegressor, 'Transformer': Transformer,
            'MultiResRegressor': MultiResRegressor}

prediction_labels = ['vsini', 'teff', 'logg', 'feh']
          
# schedulers = {'WarmupScheduler': WarmupScheduler, 'OneCycleLR': OneCycleLR,
#  'CosineAnnealingLR': CosineAnnealingLR, 'none': None}

current_date = datetime.date.today().strftime("%Y-%m-%d")
datetime_dir = f"spec_decode_multires_{current_date}"

def test_dataset(dataset, num_iters=10):
    vsinis = []
    start_time = time.time()
    for i in range(num_iters):
        x_masked, x, mask, _, info, _ = dataset[i]
        print(x.shape, info.keys())
        plt.clf()

        
    print(f"Time taken for {num_iters} iterations: {time.time() - start_time:.2f} seconds." \
        f"avg per iteration: {(time.time() - start_time)/num_iters:.2f} seconds")
    plt.hist(vsinis, bins=40)
    plt.xlabel('vsini $(km s^{-1})$')
    plt.savefig('/data/lightSpec/images/apogee_lamost_vsini_hist.png')


def create_datasets(catalog, transforms, data_args, hr=False):
    if not hr:
        train_dataset = SpectraDataset(data_args.data_dir, transforms=transforms, df=catalog, 
                                    max_len=int(data_args.max_len_spectra))
    else:
        train_dataset = SpectraDataset(data_args.data_dir_hr, transforms=transforms, df=catalog, 
                                    max_len=int(data_args.max_len_spectra_hr),id='APOGEE_ID')
    indices = list(range(len(train_dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=0.1, random_state=42)
    val_indices, test_indices = train_test_split(val_indices, test_size=0.5, random_state=42)
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)
    test_subset = Subset(train_dataset, test_indices)
    return train_dataset, train_subset, val_subset, test_subset


def create_dataloaders(train_subset, val_subset, test_subset, b_size=None):

    batch_size = b_size if b_size is not None else int(data_args.batch_size)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_subset, num_replicas=world_size, rank=local_rank)
    train_dataloader = DataLoader(train_subset,
                                  batch_size=batch_size, 
                                  num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
                                  collate_fn=kepler_collate_fn,
                                  sampler=train_sampler)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_subset, num_replicas=world_size, rank=local_rank)
    val_dataloader = DataLoader(val_subset,
                                batch_size=batch_size,
                                collate_fn=kepler_collate_fn,
                                sampler=val_sampler, 
                                num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]))

    test_sampler = torch.utils.data.distributed.DistributedSampler(test_subset, num_replicas=world_size, rank=local_rank)
    test_dataloader = DataLoader(test_subset,
                                batch_size=batch_size,
                                collate_fn=kepler_collate_fn,
                                sampler=test_sampler, 
                                num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]))

    return train_dataloader, val_dataloader, test_dataloader


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
args_dir = '/data/lightSpec/nn/config_spectra_multires.yaml'
data_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Data'])
exp_num = data_args.exp_num
model_name = data_args.model_name
if data_args.test_run:
    datetime_dir = f"test_{current_date}"
model_args = Container(**yaml.safe_load(open(args_dir, 'r'))[model_name])
conformer_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Conformer'])
transformer_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Transformer'])
optim_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Optimization'])
model_args.num_quantiles = len(optim_args.quantiles)
if not os.path.exists(f"{data_args.log_dir}/{datetime_dir}"):
    os.makedirs(f"{data_args.log_dir}/{datetime_dir}")

transforms = Compose([LAMOSTSpectrumPreprocessor(continuum_norm=data_args.continuum_norm, plot_steps=False),
                        ToTensor(),
                         ])

lamost_catalog = pd.read_csv('/data/lamost/lamost_afgkm_teff_3000_7500_catalog.csv', sep='|')
lamost_catalog = lamost_catalog.drop_duplicates(subset=['combined_obsid'])
lamost_catalog = lamost_catalog[lamost_catalog['combined_snrg'] > 0]
lamost_catalog = lamost_catalog.dropna(subset=['combined_teff', 'combined_logg', 'combined_feh'])

ds_full, train_subset, val_subset, test_subset = create_datasets(lamost_catalog, transforms, data_args)

print("number of samples: ", len(lamost_catalog))
print("train samples: ", len(train_subset))
print("val samples: ", len(val_subset))
print("test samples: ", len(test_subset))

# lamost_apogee = pd.read_csv('/data/lamost/crossmatched_catalog.csv')
# lamost_apogee = lamost_apogee[lamost_apogee['APOGEE_SNR'] > 0]
# lamost_apogee = lamost_apogee.dropna(subset=['APOGEE_VSINI','APOGEE_TEFF', 'APOGEE_LOGG', 'APOGEE_FE_H'])
# ds_full_hr, train_subset_hr, val_subset_hr, test_subset_hr = create_datasets(lamost_apogee)

transforms_hr = Compose([ToTensor()])

apogee = pd.read_csv('/data/apogee/allStar-dr17-synspec_rev1.csv')
apogee['APOGEE_ID'] = apogee['APOGEE_ID'].apply(lambda x: x.lstrip('b')[1:-1]
 if isinstance(x, str) else x)
apogee = apogee[apogee['SNR'] > 0]
apogee = apogee.dropna(subset=['VSINI','TEFF', 'LOGG', 'FE_H'])
ds_full_hr, train_subset_hr, val_subset_hr, test_subset_hr = create_datasets(apogee,
                                                 transforms_hr, data_args, hr=True)
# test_dataset(train_subset_hr, num_iters=10)
# exit()

ds_ratio = len(ds_full_hr) / len(ds_full)

train_dl, val_dl, test_dl = create_dataloaders(train_subset, val_subset, test_subset)
train_dl_hr, val_dl_hr, test_dl_hr = create_dataloaders(train_subset_hr, val_subset_hr, test_subset_hr)

print("dataframe lengths: ", len(lamost_catalog), len(apogee))
print("lamost dataset: ", len(train_subset), len(val_subset), len(test_subset))
print("apogee dataset: ", len(train_subset_hr), len(val_subset_hr), len(test_subset_hr))

# start_time = time.time()
# for i, batch in enumerate(train_dl):
#     print(i)
#     # print(batch[0].shape, batch[1].shape, batch[2].shape)
#     if i == 100:
#         break
# print(f"Time taken for 100 iterations: {time.time() - start_time:.2f} seconds." \
#     f"avg per iteration: {(time.time() - start_time)/100:.2f} seconds")

# start_time = time.time()
# for i, batch in enumerate(train_dl_hr):
#     print(i)
#     # print(batch[0].shape, batch[1].shape, batch[2].shape)
#     if i == 100:
#         break
# print(f"Time taken for 100 iterations high resoltion: {time.time() - start_time:.2f} seconds." \
#     f"avg per iteration: {(time.time() - start_time)/100:.2f} seconds")

# exit()
# test_dataset(train_subset, num_iters=100)
# test_dataset(train_subset_hr, num_iters=1000)

model = models[model_name](model_args, conformer_args=conformer_args)
# model =Transformer(transformer_args)
model = model.to(local_rank)

print("model: ", model)

checkpoint_num = int(model_args.checkpoint_num)
if model_args.load_checkpoint:
    try:
        datetime_dir = os.path.basename(os.path.dirname(model_args.checkpoint_path))
        checkpoint_num = os.path.basename(model_args.checkpoint_path).split('.')[0].split('_')[-1]
        print(datetime_dir)
        print("loading checkpoint from: ", model_args.checkpoint_path)
        model = load_checkpoints_ddp(model, model_args.checkpoint_path)
        print("loaded checkpoint from: ", model_args.checkpoint_path)
    except Exception as e:
        print("error loading checkpoint: ", e)
        deepnorm_init(model, model_args)
else:
    deepnorm_init(model, model_args)

model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters: {num_params}")

if data_args.create_umap:
    umap_df = create_umap(model.module.encoder, val_dl, local_rank, use_w=False, dual=False, max_iter=800)
    print("umap created: ", umap_df.shape)
    umap_df.to_csv(f"{data_args.log_dir}/{datetime_dir}/umap_{exp_num}_ssl.csv", index=False)
    print(f"umap saved at {data_args.log_dir}/{datetime_dir}/umap_{exp_num}_ssl.csv")
    exit()

# loss_fn = torch.nn.L1Loss(reduction='none')
loss_fn = CQR(quantiles=optim_args.quantiles, reduction='none')
ssl_loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=float(optim_args.max_lr), weight_decay=float(optim_args.weight_decay))

config_save_path = f"{data_args.log_dir}/{datetime_dir}/{model_name}_spectra__decode_{checkpoint_num}_complete_config.yaml"

complete_config = {
    "model_name": model_name,
    "data_args": data_args.__dict__,
    "model_args": model_args.__dict__,
    "conformer_args": conformer_args.__dict__,
    "optim_args": optim_args.__dict__,
    "num_params": num_params,
    "model_structure": str(model),  # Add the model structure to the configuration
    "transforms": str(transforms)
}

# Save the complete configuration to a YAML file
with open(config_save_path, "w") as config_file:
    yaml.dump(complete_config, config_file, default_flow_style=False)

print(f"Configuration (with model structure) saved at {config_save_path}.")

trainer = MultiResolutionTrainer(model=model, optimizer=optimizer,
                        criterion=loss_fn, ssl_criterion=ssl_loss_fn, output_dim=model_args.output_dim, scaler=None,
                       scheduler=None, train_dataloader=train_dl, high_res_train=train_dl_hr,
                        num_quantiles=model_args.num_quantiles,
                       val_dataloader=val_dl, high_res_val=val_dl_hr, lambda_high_res=1.5, device=local_rank,
                           exp_num=datetime_dir, log_path=data_args.log_dir, range_update=None,
                           accumulation_step=1, max_iter=np.inf, w_name='snrg',
                           w_init_val=1,  exp_name=f"{model_name}_spectra_decode_multires_{checkpoint_num}") 


fit_res = trainer.fit(num_epochs=data_args.num_epochs, device=local_rank,
                        early_stopping=10, best='loss', conf=True) 
output_filename = f'{data_args.log_dir}/{datetime_dir}/{model_name}_spectra_decode_multires_{checkpoint_num}.json'
with open(output_filename, "w") as f:
    json.dump(fit_res, f, indent=2)
fig, axes = plot_fit(fit_res, legend=exp_num, train_test_overlay=True)
plt.savefig(f"{data_args.log_dir}/{datetime_dir}/fit_{model_name}_spectra_decode_multires_{checkpoint_num}.png")
plt.clf()

preds_val, targets_val, info = trainer.predict(val_dl_hr, device=local_rank)
preds_val = np.swapaxes(preds_val, 1, 2)
cqr_errs = loss_fn.calibrate(preds_val, targets_val)

preds, targets, info = trainer.predict(test_dl_hr, device=local_rank)
preds = np.swapaxes(preds, 1, 2)
print(targets.shape, preds.shape)
preds = loss_fn.predict(preds, cqr_errs)
preds = np.swapaxes(preds, 1, 2)
print('calibrated preds: ', preds.shape)

df = save_predictions_to_dataframe(preds, targets, info, prediction_labels, optim_args.quantiles)
df.to_csv(f"{data_args.log_dir}/{datetime_dir}/{model_name}_spectra_decode_multires_{checkpoint_num}_predictions_low_res.csv", index=False)

preds, targets, info = trainer.predict(test_dl, device=local_rank)
preds = np.swapaxes(preds, 1, 2)
print(targets.shape, preds.shape)
print(len(info['Teff']))
preds = loss_fn.predict(preds, cqr_errs)
preds = np.swapaxes(preds, 1, 2)

df = save_predictions_to_dataframe(preds, targets, info, prediction_labels, optim_args.quantiles)
df.to_csv(f"{data_args.log_dir}/{datetime_dir}/{model_name}_spectra_decode_multires_{checkpoint_num}_predictions_low_res.csv", index=False)
