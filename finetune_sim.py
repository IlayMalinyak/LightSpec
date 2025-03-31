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
from dataset.dataset import KeplerDataset, SimulationDataset
from nn.astroconf import Astroconformer
from nn.models import *
# from nn.mamba import MambaEncoder
from nn.simsiam import SimSiam
from nn.train import *
from nn.utils import deepnorm_init, load_checkpoints_ddp, get_lightPred_model, compare_model_architectures
from nn.optim import CQR, TotalEnergyLoss, SumLoss, StephanBoltzmanLoss
from util.utils import *
from util.cgs_consts import *
from features import create_umap
import generator

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
torch.cuda.empty_cache()
DATA_DIR = '/data/simulations/dataset_clean'
LABELS_PATH = '/data/simulations/dataset_clean/simulation_properties.csv'
OLD_DATA_DIR = '/data/butter/data_aigrain2/simulations'
OLD_LABELS_PATH = '/data/butter/data_aigrain2/simulation_properties.csv'
LABELS = ['Period']
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]

models = {'Astroconformer': Astroconformer, 'CNNEncoder': CNNEncoder, 'MultiEncoder': MultiEncoder, 'MultiTaskSimSiam': MultiTaskSimSiam,
          "CNNEncoderDecoder": CNNEncoderDecoder, 'CNNRegressor': CNNRegressor, 'MultiTaskRegressor': MultiTaskRegressor}

def create_train_test_dfs(meta_columns):
    sim_df = pd.read_csv(OLD_LABELS_PATH)
    print("simulation df columns: ", sim_df.columns)
    sim_df['padded_idx'] = sim_df.apply(lambda x:
     f'{x["Simulation Number"]:d}'.zfill(int(np.log10(sim_df['Simulation Number'].max()))+1), axis=1)
    sim_df['sin_inc'] = np.sin(np.radians(sim_df['Inclination']))
    if 'age' in sim_df.columns:
        sim_df['log_age'] = np.log10(sim_df['age'])
    sim_df['lc_data_path'] = sim_df.apply(lambda x: os.path.join(OLD_DATA_DIR, f'lc_{x['padded_idx']}.pqt'), axis=1)
    sim_df['spec_data_path'] = sim_df.apply(lambda x: os.path.join(DATA_DIR, 'lamost', f'{x['Simulation Number']}.pqt'), axis=1)
    sim_df.to_csv('/data/butter/data_aigrain2/simulation_properties_modified.csv', index=False)
    train_df, test_df = train_test_split(sim_df, test_size=0.2, random_state=1234)
    return train_df, test_df

def create_raw_samples(data_args, num_samples=10):
    train_df, test_df = create_train_test_dfs(data_args.meta_columns_lc)
    transforms = Compose([ToTensor()])
    ds = SimulationDataset(df=train_df,
                        light_transforms=transforms,
                        spec_transforms=transforms,
                        npy_path = '/data/lightPred/data/raw_npy',
                        spec_path = data_args.spectra_dir,
                        light_seq_len=int(data_args.max_len_lc),
                        spec_seq_len=int(data_args.max_len_spectra),
                        use_acf=False,
                        use_fft=False,
                        meta_columns=data_args.meta_columns_simulation,
                        scale_flux=False,
                        labels=data_args.prediction_labels_simulation
                        )
    detrender = AvgDetrend(100*48)
    for i in range(num_samples):
        lc, spec, y, _, _, info = ds[i]
        lc_detrended, _, _ = detrender(lc[0].numpy())
        lc = lc[0]
        lc_detrended = lc_detrended[0]
        print(np.isnan(lc_detrended).sum(), np.isnan(lc).sum(), spec.shape)
        t = np.linspace(0, len(lc)/48, len(lc))
        fig, ax = plt.subplots(3,1)
        ax[0].plot(t, lc)
        ax[1].plot(t, lc_detrended)
        ax[2].plot(spec[0])
        fig.suptitle(f"Simulation {i} with period {y.item()*70:.2f}")
        plt.savefig(f'/data/lightSpec/images/simulation_{i}_lightspec.png')
        plt.close()
    
current_date = datetime.date.today().strftime("%Y-%m-%d")
datetime_dir = f"sim_finetune_{current_date}"

local_rank, world_size, gpus_per_node = setup()
args_dir = '/data/lightSpec/nn/config_finetune_sim.yaml'
data_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Data'])
data_args.prediction_labels = LABELS
exp_num = data_args.exp_num
os.makedirs(f"{data_args.log_dir}/{datetime_dir}", exist_ok=True)

# create_raw_samples(data_args, num_samples=4)
# exit()

train_dataset, val_dataset, test_dataset, complete_config = generator.get_data(data_args,
                                                                                 data_generation_fn=create_train_test_dfs,
                                                                                 dataset_name='Simulation')
s = time.time()
for i in range(100):
    lc, spec, y, _, _, info = train_dataset[i]
    # print(spec.shape, lc.shape, y.shape)
print("average time per sample: ", (time.time() - s)/100)

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

# indices = [13907, 5888, 27165, 25070, 41031, 24479, 49592, 22066, 1786, 33482]
# for i in range(10):
#         x,_, y,_,_,info = train_dataset[i]
#         print("simulation number: ", i, x.shape, y)
#         fig, ax = plt.subplots(2,1, figsize=(10,5))
#         ax[0].plot(x[0])
#         ax[1].plot(x[1])
#         fig.suptitle(f"Simulation {i} with period {y.item():.2f}")
#         plt.savefig(f'/data/lightSpec/images/simulation_{i}_lightspec.png')
# exit()
# _, optim_args, complete_config, light_model, spec_model = generator.get_model(data_args,
#                                                                  args_dir, complete_config, local_rank)

optim_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Optimization'])

light_model = get_lightPred_model(int(data_args.max_len_lc))


light_model = light_model.to(local_rank)
model = DDP(light_model, device_ids=[local_rank], find_unused_parameters=True)


optim_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Optimization'])
loss_fn = CQR(quantiles=optim_args.quantiles)
# loss_fn = nn.L1Loss()

optimizer = torch.optim.Adam(model.parameters(), lr=float(optim_args.max_lr), weight_decay=float(optim_args.weight_decay))
scaler = GradScaler()

# trainer = RegressorTrainer(
#     model=model,
#     optimizer=optimizer,
#     criterion=loss_fn,
#     train_dataloader=train_dataloader,
#     val_dataloader=val_dataloader,
#     device=local_rank,
#     output_dim=len(data_args.prediction_labels),
#     num_quantiles=len(optim_args.quantiles),
#     use_w = False,
#     only_lc=True,
#     log_path=data_args.log_dir,
#     exp_num=datetime_dir,
#     max_iter=np.inf,
#     exp_name=f"sim_finetune_{exp_num}",
# )

# trainer = ContrastiveTrainer(model=model, optimizer=optimizer,
#                             criterion=loss_fn, output_dim=1, scaler=scaler, grad_clip=True,
#                         scheduler=None, train_dataloader=train_dataloader,
#                         val_dataloader=val_dataloader, device=local_rank, ssl_weight=0, use_w=False, only_lc=True,
#                                 weight_decay=True, num_quantiles=len(optim_args.quantiles), stack_pairs=False,
#                             exp_num=datetime_dir, log_path=data_args.log_dir, range_update=None,
#                             accumulation_step=1, max_iter=100,
#                             exp_name=f"test_prot_lc_{exp_num}") 

trainer = DoubleInputTrainer2(model=model, optimizer=optimizer, num_classes=1,
                        criterion=loss_fn, output_dim=len(data_args.prediction_labels_simulation),
                         scaler=None, grad_clip=False,  scheduler=None,
                          train_dataloader=train_dataloader, num_quantiles=1,
                           val_dataloader=val_dataloader, device=local_rank,
                           exp_num=datetime_dir, log_path=data_args.log_dir, range_update=None,
                           accumulation_step=1, max_iter=100,
                        exp_name=f"finetune_sim_{exp_num}")
   


complete_config.update(
    {"trainer": trainer.__dict__,
    "loss_fn": str(loss_fn),
    "optimizer": str(optimizer)}
)
   
config_save_path = f"{data_args.log_dir}/{datetime_dir}/finetune_sim{exp_num}_complete_config.yaml"
with open(config_save_path, "w") as config_file:
    json.dump(complete_config, config_file, indent=2, default=str)

print(f"Configuration (with model structure) saved at {config_save_path}.")

fit_res = trainer.fit(num_epochs=data_args.num_epochs, device=local_rank,
                        early_stopping=10, best='loss') 
output_filename = f'{data_args.log_dir}/{datetime_dir}/{model_name}_sim_{exp_num}.json'
with open(output_filename, "w") as f:
    json.dump(fit_res, f, indent=2)
fig, axes = plot_fit(fit_res, legend=exp_num, train_test_overlay=True)
plt.savefig(f"{data_args.log_dir}/{datetime_dir}/fit_{model_name}_sim_{exp_num}.png")
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



