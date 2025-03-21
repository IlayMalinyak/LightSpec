
import os
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Subset
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import OneCycleLR
from astropy.io import fits
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl
import yaml
import json
from collections import OrderedDict
import warnings
import datetime
warnings.filterwarnings("ignore")

import sys
from os import path
ROOT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print("running from ", ROOT_DIR) 

from transforms.transforms import *
from dataset.dataset import KeplerDataset
from nn.astroconf import Astroconformer
from nn.models import CNNEncoder, MultiEncoder, CNNEncoderDecoder, CNNRegressor, MultiTaskSimSiam
# from nn.mamba import MambaEncoder
from nn.simsiam import SimSiam
from nn.train import ContrastiveTrainer, MaskedSSLTrainer
from nn.utils import deepnorm_init, load_checkpoints_ddp
from nn.optim import CQR
from util.utils import *
from features import create_umap

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
torch.cuda.empty_cache()

models = {'Astroconformer': Astroconformer, 'CNNEncoder': CNNEncoder, 'MultiEncoder': MultiEncoder,
            "CNNEncoderDecoder": CNNEncoderDecoder, 'CNNRegressor': CNNRegressor}

current_date = datetime.date.today().strftime("%Y-%m-%d")
datetime_dir = f"light_{current_date}"

local_rank, world_size, gpus_per_node = setup()
args_dir = '/data/lightSpec/nn/config_lc_ssl.yaml'
data_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Data'])
model_name = data_args.model_name
conformer_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Conformer'])
exp_num = data_args.exp_num
model_args = Container(**yaml.safe_load(open(args_dir, 'r'))[model_name])
optim_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Optimization SSL'])
os.makedirs(f"{data_args.log_dir}/{datetime_dir}", exist_ok=True)

model_args.num_quantiles = len(optim_args.quantiles)
model_args.output_dim = len(data_args.labels)

transforms = Compose([ RandomCrop(int(data_args.max_len_lc)),
                        MovingAvg(13),
                        ACF(max_lag_day=None, max_len=int(data_args.max_len_lc)),
                        Normalize(['none', 'std']),
                        # RandomMasking(0.1, mask_value=-999),
                        ToTensor(), ])


kepler_df = get_all_samples_df(num_qs=None, read_from_csv=True)
berger_cat = pd.read_csv('/data/lightPred/tables/berger_catalog_full.csv')
kepler_meta = pd.read_csv('/data/lightPred/tables/kepler_dr25_meta_data.csv')
activity_cat = pd.read_csv('/data/logs/activity_proxies/proxies_full_dist.csv')
kepler_df = kepler_df.merge(berger_cat, on='KID').merge(kepler_meta[['KID', 'KMAG']], on='KID', how='left').merge(activity_cat, on='KID', how='left')
kepler_df['kmag_abs'] = kepler_df['KMAG'] - 5 * np.log10(kepler_df['Dist']) + 5
kepler_df.dropna(subset=data_args.labels, inplace=True)
print("number of samples: ", len(kepler_df))
train_dataset = KeplerDataset(df=kepler_df, transforms=transforms,
                                target_transforms=transforms,
                                npy_path = '/data/lightPred/data/raw_npy',
                                seq_len=int(data_args.max_len_lc),
                                masked_transforms = data_args.masked_transform,
                                use_acf=data_args.use_acf,
                                scale_flux=data_args.scale_flux,
                                labels=data_args.labels,
                                dims=model_args.in_channels,
                                )
start = time.time()
for i in range(100):
    x1,x2,y,_,info1,info2 = train_dataset[i]
    print(x1.shape, x1.max(), x1.min(), x2.max(), x2.min(),x1[:, x1[0]==-999].shape, x2[:, x2[0]==-999].shape)
    if i % 10 == 0:
        fig, axes = plt.subplots(x1.shape[0],1)
        for j in range(x2.shape[0]):
            axes[j].plot(x1[j, :1800].numpy())
        plt.savefig(f"/data/lightSpec/images/lc_ssl_{i}.png")
print("average time taken per iteration: ", (time.time()-start)/100)
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


# backbone = Astroconformer(model_args)
# backbone.pred_layer = torch.nn.Identity()
# backbone = CNNEncoder(model_args)
# backbone = MambaEncoder(model_args)
backbone = models[model_name](model_args, conformer_args=conformer_args)
model = SimSiam(backbone)
model = MultiTaskSimSiam(model_args, conformer_args=conformer_args)
model = model.to(local_rank)

model_suffix = 0
checkpoint_dir = datetime_dir
checkpoint_exp = exp_num
if model_args.load_checkpoint:
    checkpoint_dir = os.path.basename(os.path.dirname(model_args.checkpoint_path))
    checkpoint_exp = os.path.basename(model_args.checkpoint_path).split('.')[0].split('_')[-1]
    print(datetime_dir)
    print("loading checkpoint from: ", model_args.checkpoint_path)
    model = load_checkpoints_ddp(model, model_args.checkpoint_path, prefix='', load_backbone=False)
    print("loaded checkpoint from: ", model_args.checkpoint_path)
else:
    deepnorm_init(model, model_args)

model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters: {num_params}")

if data_args.create_umap:
    umap_df = create_umap(model.module.encoder, val_dataloader, local_rank, use_w=False, dual=False)
    print("umap created: ", umap_df.shape)
    umap_df.to_csv(f"{data_args.log_dir}/{checkpoint_dir}/umap_{checkpoint_exp}_ssl.csv", index=False)
    print(f"umap saved at {data_args.log_dir}/{checkpoint_dir}/umap_{checkpoint_exp}_ssl.csv")
    exit()

    
loss_fn = CQR(quantiles=optim_args.quantiles)
ssl_loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=float(optim_args.max_lr), weight_decay=float(optim_args.weight_decay))
scaler = GradScaler()
total_steps = int(data_args.num_epochs) * len(train_dataloader)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                    T_max=total_steps,
                                                    eta_min=float(optim_args.max_lr)/10)
# scheduler = OneCycleLR(
#         optimizer,
#         max_lr=float(optim_args.max_lr),
#         epochs= int(data_args.num_epochs),
#         steps_per_epoch = len(train_dataloader),
#         pct_start=float(optim_args.warmup_pct),
#         anneal_strategy='cos',
#         cycle_momentum=True,
#         base_momentum=0.85,
#         max_momentum=0.95,
#         div_factor=10.0,
#         final_div_factor=100.0
#     )

# Save the complete configuration to a YAML file


if data_args.masked_transform:
    trainer = MaskedRegressorTrainer(model=model, optimizer=optimizer,
                        criterion=loss_fn, ssl_criterion=ssl_loss_fn, output_dim=1, scaler=scaler,
                       scheduler=scheduler, train_dataloader=train_dataloader,
                       val_dataloader=val_dataloader, device=local_rank,
                           exp_num=datetime_dir, log_path=data_args.log_dir, range_update=None,
                           accumulation_step=1, max_iter=np.inf,
                        exp_name=f"{model_name}_lc_{exp_num}")  
else:
    trainer = ContrastiveTrainer(model=model, optimizer=optimizer,
                            criterion=loss_fn, output_dim=1, scaler=scaler,
                        scheduler=scheduler, train_dataloader=train_dataloader,
                        val_dataloader=val_dataloader, device=local_rank, ssl_weight=data_args.ssl_weight, num_quantiles=len(optim_args.quantiles),
                            exp_num=datetime_dir, log_path=data_args.log_dir, range_update=None,
                            accumulation_step=1, max_iter=np.inf,
                            exp_name=f"{model_name}_lc_{exp_num}") 


complete_config = {
    "model_name": model_name,
    "data_args": data_args.__dict__,
    "model_args": model_args.__dict__,
    "conformer_args": conformer_args.__dict__,
    "optim_args": optim_args.__dict__,
    "num_params": num_params,
    "model_structure": str(model),  # Add the model structure to the configuration
    "transforms": str(transforms),
    'trainer': trainer.__dict__
}
config_save_path = f"{data_args.log_dir}/{datetime_dir}/{model_name}_lc_{exp_num}_complete_config.json"
with open(config_save_path, "w") as config_file:
     json.dump(complete_config, config_file, indent=2, default=str)
print(f"Configuration (with model structure) saved at {config_save_path}.")
# fig, axes = plot_lr_schedule(scheduler, optim_args.steps_per_epoch, data_args.num_epochs)
# plt.savefig(f"{data_args.log_dir}/{datetime_dir}/lr_schedule.png")

fit_res = trainer.fit(num_epochs=data_args.num_epochs, device=local_rank,
                        early_stopping=40, best='loss', conf=True) 
output_filename = f'{data_args.log_dir}/{datetime_dir}/{model_name}_lc_{exp_num}.json'
with open(output_filename, "w") as f:
    json.dump(fit_res, f, indent=2)
fig, axes = plot_fit(fit_res, legend=exp_num, train_test_overlay=True)
plt.savefig(f"{data_args.log_dir}/{datetime_dir}/fit_{model_name}_lc_{exp_num}.png")
plt.clf()

umap_df = create_umap(model.module.encoder, val_dataloader, local_rank, use_w=False, dual=False)
print("umap created: ", umap_df.shape)
umap_df.to_csv(f"{data_args.log_dir}/{datetime_dir}/umap_{exp_num}_ssl.csv", index=False)
print(f"umap saved at {data_args.log_dir}/{datetime_dir}/umap_{exp_num}_ssl.csv")
