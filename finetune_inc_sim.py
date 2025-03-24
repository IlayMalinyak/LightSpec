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
from dataset.dataset import KeplerDataset, DualDataset
from nn.astroconf import Astroconformer
from nn.models import *
# from nn.mamba import MambaEncoder
from nn.simsiam import SimSiam
from nn.train import *
from nn.utils import deepnorm_init, load_checkpoints_ddp, get_lightPred_model
from nn.optim import CQR, TotalEnergyLoss, SumLoss, StephanBoltzmanLoss
from util.utils import *
from util.cgs_consts import *
from features import create_umap
import generator

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
torch.cuda.empty_cache()
DATA_DIR = '/data/butter/data_aigrain2/simulations'
LABELS_PATH = '/data/butter/data_aigrain2/simulation_properties.csv'
LABELS = ['Period']
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]

models = {'Astroconformer': Astroconformer, 'CNNEncoder': CNNEncoder, 'MultiEncoder': MultiEncoder, 'MultiTaskSimSiam': MultiTaskSimSiam,
          "CNNEncoderDecoder": CNNEncoderDecoder, 'CNNRegressor': CNNRegressor, 'MultiTaskRegressor': MultiTaskRegressor}

current_date = datetime.date.today().strftime("%Y-%m-%d")
datetime_dir = f"inc_sim_finetune_{current_date}"

local_rank, world_size, gpus_per_node = setup()
args_dir = '/data/lightSpec/nn/config_finetune_inc.yaml'
data_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Data'])
exp_num = data_args.exp_num
os.makedirs(f"{data_args.log_dir}/{datetime_dir}", exist_ok=True)

train_dataset, val_dataset, test_dataset, complete_config = generator.get_data(data_args,
                                                                                 data_generation_fn=None,
                                                                                 dataset_name='Simulation')

print("number of training samples: ", len(train_dataset))
print("number of validation samples: ", len(val_dataset))
print("number of test samples: ", len(test_dataset))

train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
train_dataloader = DataLoader(train_dataset,
                              batch_size=int(data_args.batch_size), \
                              num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
                              collate_fn=kepler_collate_fn,
                              sampler=train_sampler)


val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank)
val_dataloader = DataLoader(val_dataset,
                            batch_size=int(data_args.batch_size),
                            collate_fn=kepler_collate_fn,
                            sampler=val_sampler, \
                            num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]))

test_dataloader = DataLoader(test_dataset,
                            batch_size=int(data_args.batch_size),
                            collate_fn=kepler_collate_fn,
                            num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]))


for i in range(10):
    light, spec, y, _, _, info = train_dataset[i]
    print("train dataset: ", light.shape, spec.shape, y)

model, optim_args, complete_config = generator.get_model(data_args, args_dir, complete_config, local_rank)

loss_fn = CQR(quantiles=optim_args.quantiles)
optimizer = torch.optim.Adam(model.parameters(), lr=float(optim_args.max_lr), weight_decay=float(optim_args.weight_decay))


trainer = RegressorTrainer(
    model=model,
    optimizer=optimizer,
    criterion=loss_fn,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    device=local_rank,
    output_dim=len(data_args.prediction_labels),
    num_quantiles=len(optim_args.quantiles),
    use_w = False,
    log_path=data_args.log_dir,
    exp_num=datetime_dir,
    exp_name=f"inc_sim_finetune_{exp_num}",
)


complete_config.update(
    {"trainer": trainer.__dict__,
    "loss_fn": str(loss_fn),
    "optimizer": str(optimizer)}
)
   
config_save_path = f"{data_args.log_dir}/{datetime_dir}/finetune_inc_sim{exp_num}_complete_config.yaml"
with open(config_save_path, "w") as config_file:
    json.dump(complete_config, config_file, indent=2, default=str)

print(f"Configuration (with model structure) saved at {config_save_path}.")

fit_res = trainer.fit(num_epochs=data_args.num_epochs, device=local_rank,
                        early_stopping=10, best='loss', conf=True) 
output_filename = f'{data_args.log_dir}/{datetime_dir}/{model_name}_inc_sim_{exp_num}.json'
with open(output_filename, "w") as f:
    json.dump(fit_res, f, indent=2)
fig, axes = plot_fit(fit_res, legend=exp_num, train_test_overlay=True)
plt.savefig(f"{data_args.log_dir}/{datetime_dir}/fit_{model_name}_inc_sim{exp_num}.png")
plt.clf()

test_res = trainer.predict(test_loader, device=local_rank)

y, y_pred = test_res['y'], test_res['y_pred']
print('results shapes: ', y.shape, y_pred.shape)
results_df = pd.DataFrame({'y': y})
for q in len(optim_args.quantiles):
    y_pred_q = y_pred[:,:, q]
    for i, label in enumerate(['vsini', 'Prot', 'sin_inc']):
        results_df[f'{label}_{q}'] = y_pred_q[:, i]
print(results_df.head())
results_df.to_csv(f"{data_args.log_dir}/{datetime_dir}/test_predictions.csv", index=False)
# Access results



