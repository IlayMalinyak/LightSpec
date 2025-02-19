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
from nn.models import CNNEncoderDecoder, CNNRegressor
from nn.optim import CQR
# from nn.mamba import MambaEncoder
from nn.simsiam import SimSiam
from nn.train import RegressorTrainer
from nn.utils import deepnorm_init, load_checkpoints_ddp
from util.utils import *
from features import create_umap

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
torch.cuda.empty_cache()

models = {'Astroconformer': Astroconformer, 'CNNEncoderDecoder': CNNEncoderDecoder,
            'CNNRegressor': CNNRegressor}

current_date = datetime.date.today().strftime("%Y-%m-%d")
datetime_dir = f"light__deocde_{current_date}"

local_rank, world_size, gpus_per_node = setup()
args_dir = '/data/lightSpec/nn/config_lc_decode.yaml'
data_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Data'])
model_name = data_args.model_name
exp_num = data_args.exp_num
model_args = Container(**yaml.safe_load(open(args_dir, 'r'))[model_name])
conformer_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Conformer'])
optim_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Optimization'])
if not os.path.exists(f"{data_args.log_dir}/{datetime_dir}"):
    os.makedirs(f"{data_args.log_dir}/{datetime_dir}")

transforms = Compose([RandomCrop(int(data_args.max_days_lc/data_args.lc_freq)),
                        MovingAvg(13),
                        Normalize('std'),
                        ToTensor(), ])

kepler_df = get_all_samples_df(num_qs=None, read_from_csv=True)

lightpred_df = pd.read_csv('/data/lightPred/tables/reinhold2015.csv')
kepler_df = kepler_df.merge(lightpred_df[['KIC', 'Pmax', 'Pmin', 'Rvar']], left_on='KID', right_on='KIC')
print("number of samples: ", len(kepler_df))
train_dataset = KeplerDataset(df=kepler_df, transforms=transforms,
                                target_transforms=None,
                                npy_path = '/data/lightPred/data/npy',
                                seq_len=int(data_args.max_days_lc/data_args.lc_freq)
                                )
start = time.time()
for i in range(100):
    x,y,_,_,info1,info2 = train_dataset[i]
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
                            
model_args.num_quantiles = len(optim_args.quantiles)
model = models[model_name](model_args, conformer_args=conformer_args)
model = model.to(local_rank)
checkpoint_num = int(model_args.checkpoint_num)
if model_args.load_checkpoint:
    datetime_dir = os.path.basename(os.path.dirname(model_args.checkpoint_path))
    # exp_num = os.path.basename(model_args.checkpoint_path).split('.')[0].split('_')[-1]
    print(datetime_dir)
    print("loading checkpoint from: ", model_args.checkpoint_path)
    model = load_checkpoints_ddp(model, model_args.checkpoint_path, prefix='encoder.', load_backbone=True)
    print("loaded checkpoint from: ", model_args.checkpoint_path)
else:
    deepnorm_init(model, model_args)

model = DDP(model, device_ids=[local_rank])

print('model: ', model)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters: {num_params}")

if data_args.create_umap:
    model.module.return_logits = True
    umap_df = create_umap(model, val_dataloader, local_rank, use_w=False, dual=False)
    print("umap created: ", umap_df.shape)
    umap_df.to_csv(f"{data_args.log_dir}/{datetime_dir}/umap_{exp_num}.csv", index=False)
    print(f"umap saved at {data_args.log_dir}/{datetime_dir}/umap_{exp_num}.csv")
    exit()


# loss_fn = torch.nn.L1Loss(reduction='none')
loss_fn = CQR(quantiles=optim_args.quantiles, reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=float(optim_args.max_lr), weight_decay=float(optim_args.weight_decay))

config_save_path = f"{data_args.log_dir}/{datetime_dir}/{model_name}_lc__decode_{checkpoint_num}_complete_config.yaml"

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

trainer = RegressorTrainer(model=model, optimizer=optimizer,
                        criterion=loss_fn, output_dim=1, scaler=None,
                       scheduler=None, train_dataloader=train_dataloader,
                       val_dataloader=val_dataloader, device=local_rank,
                           exp_num=datetime_dir, log_path=data_args.log_dir, range_update=None,
                           accumulation_step=1, max_iter=np.inf, w_name=None,
                           w_init_val=1, num_quantiles=len(optim_args.quantiles),
                             exp_name=f"{model_name}_lc_decode_{checkpoint_num}") 
fit_res = trainer.fit(num_epochs=data_args.num_epochs, device=local_rank,
                        early_stopping=60, only_p=False, best='loss', conf=True) 
output_filename = f'{data_args.log_dir}/{datetime_dir}/{model_name}_lc__decode_{checkpoint_num}.json'
with open(output_filename, "w") as f:
    json.dump(fit_res, f, indent=2)
fig, axes = plot_fit(fit_res, legend=exp_num, train_test_overlay=True)
plt.savefig(f"{data_args.log_dir}/{datetime_dir}/fit_{model_name}_lc__decode_{checkpoint_num}.png")
plt.clf()