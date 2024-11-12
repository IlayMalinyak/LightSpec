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
warnings.filterwarnings("ignore")

import sys
from os import path
ROOT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print("running from ", ROOT_DIR) 

from transforms.transforms import *
from dataset.dataset import KeplerDataset
from nn.astroconf import Astroconformer
from nn.simsiam import SimSiam
from nn.train import ContrastiveTrainer
from nn.utils import deepnorm_init
from util.utils import *

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
torch.cuda.empty_cache()

local_rank, world_size, gpus_per_node = setup()
args_dir = '/data/lightSpec/nn/config_lc_ssl.yaml'
model_name = 'Astroconformer'
model_args = Container(**yaml.safe_load(open(args_dir, 'r'))[model_name])
data_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Data'])
optim_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Optimization SSL'])
exp_num = data_args.exp_num
if not os.path.exists(f"{data_args.log_dir}/exp{exp_num}"):
    os.makedirs(f"{data_args.log_dir}/exp{exp_num}")

transforms = Compose([RandomCrop(int(data_args.max_days_lc/data_args.lc_freq)),
                    #   FillNans(interpolate=True),
                        MovingAvg(13),
                        RandomTransform([AddGaussianNoise(sigma=0.0001),
                                        RandomMasking(mask_prob=0.05),
                                        Shuffle(segment_len=270/data_args.lc_freq),
                                        Identity()]),
                        Normalize('std'),
                        ToTensor(), ])


kepler_df = get_all_samples_df(num_qs=None, read_from_csv=True)
print("number of samples: ", len(kepler_df))
train_dataset = KeplerDataset(df=kepler_df, transforms=transforms,
                                target_transforms=transforms,
                                npy_path = '/data/lightPred/data/npy',
                                seq_len=int(data_args.max_days_lc/data_args.lc_freq)
                                )
start = time.time()
for i in range(100):
    x1,x2,_,_,info1,info2 = train_dataset[i]
    print(x1.shape, x2.shape)
print("average time taken per iteration: ", (time.time()-start)/100)


indices = list(range(len(train_dataset)))
train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
train_subset = Subset(train_dataset, train_indices)
val_subset = Subset(train_dataset, val_indices)


backbone = Astroconformer(model_args)
backbone.pred_layer = torch.nn.Identity()
model = SimSiam(backbone)
model = model.to(local_rank)

model_suffix = 0
checkpoint_num = int(model_args.checkpoint_num)
if model_args.load_checkpoint:
    prev_checkpoint_num = checkpoint_num - 1
    print("****Loading checkpoint******")
    state_dict = torch.load(f'{data_args.log_dir}/exp{exp_num}/astroconf_lc_{prev_checkpoint_num}.pth', map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        while key.startswith('module.'):
            key = key[7:]
        new_state_dict[key] = value
    state_dict = new_state_dict
    model.load_state_dict(state_dict)
else:
    deepnorm_init(model, model_args)

model = DDP(model, device_ids=[local_rank])

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters: {num_params}")

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
    
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=float(optim_args.weight_decay))
scaler = GradScaler()
scheduler = OneCycleLR(
        optimizer,
        max_lr=float(optim_args.max_lr),
        epochs= int(data_args.num_epochs),
        steps_per_epoch = len(train_dataloader),
        pct_start=float(optim_args.warmup_pct),
        anneal_strategy='cos',
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=10.0,
        final_div_factor=100.0
    )
# fig, axes = plot_lr_schedule(scheduler, optim_args.steps_per_epoch, data_args.num_epochs)
# plt.savefig(f"{data_args.log_dir}/exp{exp_num}/lr_schedule.png")
trainer = ContrastiveTrainer(model=model, optimizer=optimizer,
                        criterion=loss_fn, output_dim=1, scaler=scaler,
                       scheduler=None, train_dataloader=train_dataloader,
                       val_dataloader=val_dataloader, device=local_rank,
                           exp_num=exp_num, log_path=data_args.log_dir, range_update=None,
                           accumulation_step=1, max_iter=np.inf,
                        exp_name=f"astroconf_lc_{checkpoint_num}") 
fit_res = trainer.fit(num_epochs=data_args.num_epochs, device=local_rank,
                        early_stopping=40, only_p=False, best='loss', conf=True) 
output_filename = f'{data_args.log_dir}/exp{exp_num}/astroconf_lc_{checkpoint_num}.json'
with open(output_filename, "w") as f:
    json.dump(fit_res, f, indent=2)
fig, axes = plot_fit(fit_res, legend=exp_num, train_test_overlay=True)
plt.savefig(f"{data_args.log_dir}/exp{exp_num}/fit_lc_{checkpoint_num}.png")
plt.clf()