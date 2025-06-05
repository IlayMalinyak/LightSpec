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


def test_dataset_samples(dataset, num_iters=100):
    start_time = time.time()
    for i in range(num_iters):
        x_masked, x, y, mask, _, info = dataset[i]
        print(info['spectra_type'], x_masked.shape, x.shape, y.shape, mask.shape)
        # print('y: ', len(y))
        # if 'rv2' in info.keys():
        #     print(info['snrg'], info['snri'], info['snrr'], info['snrz'])
        # if i % 20 == 0:
        #     plt.plot(info['wavelength'], x_masked.cpu().numpy(), label='masked')
        #     plt.plot(info['wavelength'], x.cpu().numpy(), label='original')
        #     plt.title(f"{info['spectra_type']} {info['id']}")
        #     plt.xlabel('Wavelength')
        #     plt.ylabel('Flux')
        #     plt.savefig(f"/data/lightSpec/images/lamost_apogee_sample_{i}.png")
        #     plt.clf()
    print(f"Time taken for {num_iters} iterations: {time.time() - start_time:.4f} seconds." \
        f"avg per iteration: {(time.time() - start_time)/num_iters:.6f} seconds")

def create_train_test_dfs(meta_columns):
    lamost_apogee_catalog = pd.read_csv('/data/lamost/lamost_afgkm_teff_3000_7500_apogee_all.csv')
    lamost_apogee_catalog = lamost_apogee_catalog.drop_duplicates(subset=['combined_obsid', 'APOGEE_ID'])
    lamost_apogee_catalog['APOGEE_ID'] = lamost_apogee_catalog['APOGEE_ID'].apply(
    lambda x: x[2:-1] if isinstance(x, str) and x.startswith("b'") and x.endswith("'") else x)
    print("lamost_apogee_catalog shape: ", lamost_apogee_catalog.shape,
     'apogee sampels: ', lamost_apogee_catalog['APOGEE_ID'].nunique())
    
    
    train_df, test_df = train_test_split(lamost_apogee_catalog, test_size=0.2, random_state=1234)
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
args_dir = '/data/lightSpec/nn/full_config_lamost_apogee.yaml'
data_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Data'])
exp_num = data_args.exp_num
model_name = data_args.spec_model_name
os.makedirs(f"{data_args.log_dir}/{datetime_dir}", exist_ok=True)

# if data_args.test_run:
#     datetime_dir = f"test_{current_date}"


train_dataset, val_dataset, test_dataset, complete_config = generator.get_data(data_args,
 data_generation_fn=create_train_test_dfs, dataset_name='Spectra')

# test_dataset_samples(train_dataset, num_iters=100)
# exit()

train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
train_dataloader = DataLoader(train_dataset,
                              batch_size=int(data_args.batch_size) * 4, \
                              num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
                              collate_fn=kepler_collate_fn,
                              sampler=train_sampler)


val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank)
val_dataloader = DataLoader(val_dataset,
                            batch_size=int(data_args.batch_size) * 4,
                            collate_fn=kepler_collate_fn,
                            sampler=val_sampler, \
                            num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]))

test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=world_size, rank=local_rank)
test_dataloader = DataLoader(test_dataset,
                            batch_size=int(data_args.batch_size) * 4,
                            collate_fn=kepler_collate_fn,
                            sampler=test_sampler, \
                            num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]))

_, optim_args, tuner_args, complete_config, _, model = generator.get_model(data_args, args_dir, complete_config, local_rank)

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
                                 accumulation_step=1, max_iter=np.inf, w_name='snrg', add_wv=data_args.add_wv,
                           w_init_val=1,  exp_name=f"spectra_decode_{exp_num}") 

complete_config.update(
    {"trainer": trainer.__dict__,
    "loss_fn": str(loss_fn),
    "optimizer": str(optimizer)}
)
   
config_save_path = f"{data_args.log_dir}/{datetime_dir}/spec_decode_{exp_num}_complete_config.yaml"
with open(config_save_path, "w") as config_file:
    json.dump(complete_config, config_file, indent=2, default=str)

print(f"Configuration (with model structure) saved at {config_save_path}.")

fit_res = trainer.fit(num_epochs=data_args.num_epochs, device=local_rank,
                       early_stopping=40, best='loss', conf=True) 
output_filename = f'{data_args.log_dir}/{datetime_dir}/{model_name}_spectra_decode_{exp_num}.json'
with open(output_filename, "w") as f:
    json.dump(fit_res, f, indent=2)
fig, axes = plot_fit(fit_res, legend=exp_num, train_test_overlay=True)
plt.savefig(f"{data_args.log_dir}/{datetime_dir}/fit_{model_name}_spectra_decode_{exp_num}.png")
plt.clf()

# predict_results(trainer, val_dataloader, test_dataloader, loss_fn,
#                  data_args.prediction_labels_lc,
#                  data_args, optim_args, 'light', exp_num,
#                   datetime_dir, local_rank, world_size)

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
coverage = np.mean((targets >= low_q) & (targets <= high_q), axis=0)

obsids = info['obsid']
df = pd.DataFrame({
    'obsid': obsids,
})
for i, label in enumerate(prediction_labels):
    df[f'target_{label}'] = targets[:, i]
    for j, q in enumerate(optim_args.quantiles):
        print(label, q)
        df[f'pred_{label}_q{q:.3f}'] = preds_cqr[:, i, j]
print(df.head)
print('coverage after calibration: ', coverage)
df.to_csv(f"{data_args.log_dir}/{datetime_dir}/{model_name}_spectra_decode2_{exp_num}_cqr.csv", index=False)


# df = save_predictions_to_dataframe(preds, targets, info, prediction_labels, optim_args.quantiles)
# df.to_csv(f"{data_args.log_dir}/{datetime_dir}/{model_name}_spectra_decode2_{exp_num}.csv", index=False)
# print('predictions saved in', f"{data_args.log_dir}/{datetime_dir}/{model_name}_spectra_decode2_{exp_num}.csv") 
# df_cqr = save_predictions_to_dataframe(preds_cqr, targets, info, prediction_labels, optim_args.quantiles)
# df_cqr.to_csv(f"{data_args.log_dir}/{datetime_dir}/{model_name}_spectra_decode2_{exp_num}_cqr.csv", index=False)
# print('predictions saved in', f"{data_args.log_dir}/{datetime_dir}/{model_name}_spectra_decode2_{exp_num}_cqr.csv")

# umap_df = create_umap(model.module.encoder, test_dataloader, local_rank, use_w=False, dual=False)
# print("umap created: ", umap_df.shape)
# umap_df.to_csv(f"{data_args.log_dir}/{datetime_dir}/umap_{exp_num}_ssl.csv", index=False)
# print(f"umap saved at {data_args.log_dir}/{datetime_dir}/umap_{exp_num}_ssl.csv")
# exit(0)
