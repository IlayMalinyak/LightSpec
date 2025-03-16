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
import datetime

warnings.filterwarnings("ignore")

import sys
from os import path
ROOT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print("running from ", ROOT_DIR) 

from transforms.transforms import *
from dataset.dataset import DualDataset
from nn.astroconf import Astroconformer, AstroEncoderDecoder
from nn.models import CNNEncoder, CNNEncoderDecoder, MultiEncoder
from nn.simsiam import SimSiam
from nn.moco import MoCo, MultimodalMoCo, LightCurveSpectraMoCo
from nn.utils import init_model, load_checkpoints_ddp
from nn.supervised import DualModel
from util.utils import *
from nn.train import Trainer
from nn.optim import QuantileLoss, CQR

DATA_DIR = '/data/simulations/dataset_small'
LABELS_PATH = '/data/simulations/dataset_small/simulation_properties.csv'
LABELS = ['Inclination', 'Shear', 'Period']
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
MODELS = {'Astroconformer': Astroconformer, 'CNNEncoder': CNNEncoder,
           'AstroEncoderDecoder': AstroEncoderDecoder, 'CNNEncoderDecoder': CNNEncoderDecoder,}

def collate_fn(batch:List):
    """
    Collate function for the Kepler dataset.        
    """
    lcs, spectras, ys, info = zip(*batch)

    lcss_tensor = torch.stack(lcs, dim=0)
    spectras_tensor = torch.stack(spectras, dim=0)
    ys_tensor = torch.stack(ys, dim=0)
    return lcss_tensor, spectras_tensor, ys_tensor, info


def get_backbone(light_model_name, light_model_args, spec_model_name, spec_model_args):

    light_backbone = MODELS[light_model_name](light_model_args)
    if light_model_name == 'Astroconformer':
        light_backbone.pred_layer = torch.nn.Identity()
    light_model = SimSiam(light_backbone)

    light_model = init_model(light_model, light_model_args)

    spec_model = MODELS[spec_model_name](spec_model_args)

    spec_model = init_model(spec_model, spec_model_args)

    # model = MultimodalMoCo(spec_model.encoder, light_model.backbone,  **moco_args.get_dict()).to(local_rank)

    return light_model.backbone, spec_model.encoder

torch.cuda.empty_cache()

current_date = datetime.date.today().strftime("%Y-%m-%d")
datetime_dir = f"sbl_{current_date}"
local_rank, world_size, gpus_per_node = setup()

args_dir = '/data/lightSpec/nn/config_lightspec_ssl.yaml'
data_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Data'])
exp_num = data_args.exp_num
light_model_name = data_args.light_model_name
spec_model_name = data_args.spec_model_name
if data_args.test_run:
    datetime_dir = f"test_{current_date}"
light_model_args = Container(**yaml.safe_load(open(args_dir, 'r'))[light_model_name])
spec_model_args = Container(**yaml.safe_load(open(args_dir, 'r'))[spec_model_name])
sims_args = Container(**yaml.safe_load(open(args_dir, 'r'))['SimSiam'])
conv_args = Container(**yaml.safe_load(open(args_dir, 'r'))['conv_args'])
astroconf_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Astroconformer'])
optim_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Optimization SSL'])
if not os.path.exists(f"{data_args.log_dir}/{datetime_dir}"):
    os.makedirs(f"{data_args.log_dir}/{datetime_dir}")

light_transforms = Compose([RandomCrop(int(data_args.max_days_lc/data_args.lc_freq)),
                            MovingAvg(13),
                            ToTensor(),
                         ])
spec_transforms = Compose([LAMOSTSpectrumPreprocessor(continuum_norm=False, plot_steps=False),
                            ToTensor()
                           ])

train_dataset = DualDataset(data_dir=DATA_DIR, labels_names=LABELS, labels_path=LABELS_PATH, lc_transforms=light_transforms,
                            spectra_transforms=spec_transforms)

indices = list(range(len(train_dataset)))
train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
train_subset = Subset(train_dataset, train_indices)
val_subset = Subset(train_dataset, val_indices)

erros = 0
for i in range(len(train_dataset)):
    lc, spec, y, info = train_dataset[i]
    if i % 10000 == 0:
        print(i)
    if not lc.sum():
        erros += 1
print("num erros: ", erros)
exit()

train_sampler = torch.utils.data.distributed.DistributedSampler(train_subset, num_replicas=world_size, rank=local_rank)
train_dataloader = DataLoader(train_subset, batch_size=data_args.batch_size, sampler=train_sampler, \
                                            num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
                                             pin_memory=True, collate_fn=collate_fn)


val_sampler = torch.utils.data.distributed.DistributedSampler(val_subset, num_replicas=world_size, rank=local_rank)
val_dataloader = DataLoader(val_subset, batch_size=data_args.batch_size, sampler=val_sampler, \
                                num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]), 
                                collate_fn=collate_fn, pin_memory=True)
                        

lc_backbone, spec_backbone = get_backbone(light_model_name, light_model_args,
                                           spec_model_name, spec_model_args)

# backbone = load_checkpoints_ddp(backbone, data_args.checkpoint_path)

# for param in backbone.parameters():
#     param.requires_grad = False

model = DualModel(lc_backbone, spec_backbone,
                 hidden_dim=128,
                  output_dim=len(LABELS),
                  backbone_args = conv_args,
                  num_quantiles=len(QUANTILES)).to(local_rank)

model = DDP(model, device_ids=[local_rank])

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters: {num_params}")
print("number of training samples: ", len(train_subset), len(val_subset))
print("model: \n", model)

loss_fn = CQR(quantiles=QUANTILES)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)

trainer = Trainer(model=model, optimizer=optimizer,
                        criterion=loss_fn, output_dim=len(LABELS), scaler=None,
                       scheduler=None, train_dataloader=train_dataloader,
                       val_dataloader=val_dataloader, device=local_rank,
                           exp_num=datetime_dir, log_path=data_args.log_dir, range_update=None,
                           accumulation_step=1, max_iter=np.inf, 
                        exp_name=f"sbl_{exp_num}") 

config_save_path = f"{data_args.log_dir}/{datetime_dir}/sbl_{exp_num}_complete_config.yaml"

complete_config = {
    "data_args": data_args.__dict__,
    "light_model_args": light_model_args.__dict__,
    "spec_model_args": spec_model_args.__dict__,
    "sims_args": sims_args.__dict__,
    "conv_args": conv_args.__dict__,
    "optim_args": optim_args.__dict__,
    "num_params": num_params,
    "model_structure": str(model),
    "light_transforms": str(light_transforms),
    "spec_transforms": str(spec_transforms),
    "trainer": str(trainer)
}

# Save the complete configuration to a YAML file
with open(config_save_path, "w") as config_file:
    yaml.dump(complete_config, config_file, default_flow_style=False)
print(f"Configuration (with model structure) saved at {config_save_path}.")

fit_res = trainer.fit(num_epochs=data_args.num_epochs, device=local_rank,
                        early_stopping=40, only_p=False, best='loss', conf=True) 
output_filename = f'{data_args.log_dir}/{datetime_dir}/sbl_{exp_num}.json'
with open(output_filename, "w") as f:
    json.dump(fit_res, f, indent=2)
fig, axes = plot_fit(fit_res, legend=exp_num, train_test_overlay=True)
plt.savefig(f"{data_args.log_dir}/{datetime_dir}/sbl_fit_{exp_num}.png")
plt.clf()







