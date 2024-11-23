import os
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Subset
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import OneCycleLR
from astropy.io import fits
import pandas as pd
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
from dataset.dataset import LightSpecDataset
from dataset.sampler import DistinctParameterSampler
from nn.astroconf import Astroconformer, AstroEncoderDecoder
from nn.moco import MoCo, MultimodalMoCo, LightCurveSpectraMoCo
from nn.simsiam import SimSiam
from nn.utils import deepnorm_init
from util.utils import *
from nn.train import ContrastiveTrainer

META_COLUMNS = ['KID', 'Teff', 'logg', 'FeH', 'Rstar', 'Mstar']

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
torch.cuda.empty_cache()

local_rank, world_size, gpus_per_node = setup()
args_dir = '/data/lightSpec/nn/config_lightspec_ssl.yaml'
light_model_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Astroconformer'])
spec_model_args = Container(**yaml.safe_load(open(args_dir, 'r'))['AstroEncoderDecoder'])
moco_args = Container(**yaml.safe_load(open(args_dir, 'r'))['MoCo'])
data_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Data'])
optim_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Optimization SSL'])
exp_num = data_args.exp_num
if not os.path.exists(f"{data_args.log_dir}/exp{exp_num}"):
    os.makedirs(f"{data_args.log_dir}/exp{exp_num}")

light_transforms = Compose([RandomCrop(int(data_args.max_days_lc/data_args.lc_freq)),
                        Normalize('std'),
                        ToTensor(),
                         ])
spec_transforms = Compose([MovingAvg(7),
                           Normalize("minmax", axis=0),
                           AvgDetrend(kernel_size=100),
                           ToTensor(),
                           ])

kepler_df = get_all_samples_df(num_qs=None, read_from_csv=True)
kepler_meta = pd.read_csv('/data/lightPred/tables/berger_catalog.csv')
kepler_df = kepler_df.merge(kepler_meta, on='KID', how='left')
lamost_kepler_df = pd.read_csv('/data/lamost/lamost_dr8_gaia_dr3_kepler_ids.csv')
lamost_kepler_df = lamost_kepler_df[~lamost_kepler_df['KID'].isna()]
lamost_kepler_df['KID'] = lamost_kepler_df['KID'].astype(int)
lamost_kepler_df = lamost_kepler_df.merge(kepler_df[META_COLUMNS], on='KID', how='inner')

# all_spectra_files = os.listdir(spectra_dir)
# spectra_kids = np.array([int(os.path.splitext(filename)[0]) for filename in all_spectra_files if filename.endswith('.fits')])
# print(spectra_kids[:10])
# print(kepler_df['KID'].iloc[:10])
plt.hist(kepler_df['KID'].values, bins=100, density=True, alpha=0.5, label='kepler')
plt.hist(lamost_kepler_df['KID'].values, bins=100, alpha=0.5, density=True, label='spectra')
plt.legend()
plt.savefig('/data/lightSpec/spectra_kepler_hist.png')
plt.close()
kepler_df_filtered = kepler_df[kepler_df['KID'].isin(lamost_kepler_df['KID'].values)]
# kepler_df_filtered = kepler_df_filtered.merge(lamost_kepler_df[['KID', 'ObsID']], on='KID')

print("number of samples: ", len(kepler_df), len(lamost_kepler_df), 'filtered: ', len(kepler_df_filtered))
train_dataset = LightSpecDataset(df=lamost_kepler_df, light_transforms=light_transforms,
                                spec_transforms=spec_transforms,
                                npy_path = '/data/lightPred/data/npy',
                                spec_path = data_args.spectra_dir,
                                light_seq_len=int(data_args.max_days_lc/data_args.lc_freq),
                                spec_seq_len=int(data_args.max_len_spectra)
                                )
                        
indices = list(range(len(train_dataset)))
train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
train_subset = Subset(train_dataset, train_indices)
val_subset = Subset(train_dataset, val_indices)

start = time.time()
no_founds = 0
for i in range(100):
    light, spec, _,_, info,_ = train_dataset[i]
    if light.sum() == 0:
        no_founds += 1
    # print(light.shape, spec.shape)
print("average time taken per iteration: ", (time.time()-start)/100)
print("no_founds: ", no_founds)

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

light_backbone = Astroconformer(light_model_args)
light_backbone.pred_layer = torch.nn.Identity()
light_model = SimSiam(light_backbone)

if light_model_args.load_light_checkpoint:
    print("****Loading light checkpoint****")
    state_dict = torch.load(f'{light_model_args.light_checkpoint_path}', map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        while key.startswith('module.'):
            key = key[7:]
        # key = key.replace('backbone.', '')
        new_state_dict[key] = value
    state_dict = new_state_dict
    missing, unexpected = light_model.load_state_dict(state_dict, strict=False)
    print("missing keys: ", missing)
    print("unexpected keys: ", unexpected)
    
else:
    print("****deepnorm init for lightcurve****")
    deepnorm_init(light_backbone, light_model_args)

spec_model = AstroEncoderDecoder(spec_model_args)

if spec_model_args.load_spec_checkpoint:
    print("****Loading spectra checkpoint******")
    state_dict = torch.load(f'{spec_model_args.spec_checkpoint_path}', map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        while key.startswith('module.'):
            key = key[7:]
        new_state_dict[key] = value
    state_dict = new_state_dict
    missing, unexpected = spec_model.load_state_dict(state_dict, strict=False)
    print("missing keys: ", missing)
    print("unexpected keys: ", unexpected)
else:
    print("****deepnorm init for spectra****")
    deepnorm_init(spec_model, spec_model_args)

# model = MultimodalMoCo(spec_model.encoder, light_model.backbone,  **moco_args.get_dict()).to(local_rank)
model = LightCurveSpectraMoCo(spec_model.encoder, light_model.backbone,  **moco_args.get_dict()).to(local_rank)
model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters: {num_params}")
print("number of training samples: ", len(train_subset), len(val_subset))
print("model: \n", model)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=float(optim_args.weight_decay))
scaler = GradScaler()

# print all the trainable layers in the model
print("Trainable layers in the model: ")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.shape)

# save all containers in log directory
all_args = yaml.safe_load(open(args_dir, 'r'))
yaml.dump(all_args, open(f"{data_args.log_dir}/exp{exp_num}/config.yaml", 'w'))

print("config saved in ", f"{data_args.log_dir}/exp{exp_num}/config.yaml")

trainer = ContrastiveTrainer(model=model, optimizer=optimizer,
                        criterion=loss_fn, output_dim=1, scaler=None,
                       scheduler=None, train_dataloader=train_dataloader,
                       val_dataloader=val_dataloader, device=local_rank,
                           exp_num=exp_num, log_path=data_args.log_dir, range_update=None,
                           accumulation_step=1, max_iter=np.inf,
                        exp_name="lightspec_ssl") 
fit_res = trainer.fit(num_epochs=data_args.num_epochs, device=local_rank,
                        early_stopping=40, only_p=False, best='loss', conf=True) 
output_filename = f'{data_args.log_dir}/exp{exp_num}/lightspec.json'
with open(output_filename, "w") as f:
    json.dump(fit_res, f, indent=2)
fig, axes = plot_fit(fit_res, legend=exp_num, train_test_overlay=True)
plt.savefig(f"{data_args.log_dir}/exp{exp_num}/fit_lightspec.png")
plt.clf()
