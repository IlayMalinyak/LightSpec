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
from dataset.dataset import LightSpecDataset, create_unique_loader
from dataset.sampler import DistinctParameterSampler
from nn.astroconf import Astroconformer, AstroEncoderDecoder
from nn.models import CNNEncoder, CNNEncoderDecoder, MultiEncoder, MultiTaskRegressor, Transformer
from nn.simsiam import MultiModalSimSiam, MultiModalSimCLR
from nn.moco import MultimodalMoCo, PredictiveMoco
from nn.simsiam import SimSiam, projection_MLP
from nn.utils import init_model, load_checkpoints_ddp
from util.utils import *
from nn.train import LightSpecTrainer
from tests.test_unique_sampler import run_sampler_tests
from features import multimodal_umap, create_umap

META_COLUMNS = ['KID', 'Teff', 'logg', 'FeH', 'Rstar', 'Mstar', 'kmag_abs']

MODELS = {'Astroconformer': Astroconformer, 'CNNEncoder': CNNEncoder, 'MultiEncoder': MultiEncoder, 'MultiTaskRegressor': MultiTaskRegressor,
           'AstroEncoderDecoder': AstroEncoderDecoder, 'CNNEncoderDecoder': CNNEncoderDecoder,}

torch.cuda.empty_cache()

torch.manual_seed(1234)
np.random.seed(1234)


current_date = datetime.date.today().strftime("%Y-%m-%d")
datetime_dir = f"combined_{current_date}"

def create_train_test_dfs(norm_cols=['Teff', 'logg', 'Mstar']):
    kepler_df = get_all_samples_df(num_qs=None, read_from_csv=True)
    kepler_meta = pd.read_csv('/data/lightPred/tables/berger_catalog.csv')
    kmag_df = pd.read_csv('/data/lightPred/tables/kepler_dr25_meta_data.csv')
    kepler_df = kepler_df.merge(kepler_meta, on='KID', how='left').merge(kmag_df[['KID', 'KMAG']], on='KID', how='left')
    kepler_df['kmag_abs'] = kepler_df['KMAG'] - 5 * np.log10(kepler_df['Dist']) + 5
    lamost_kepler_df = pd.read_csv('/data/lamost/lamost_dr8_gaia_dr3_kepler_ids.csv')
    lamost_kepler_df = lamost_kepler_df[~lamost_kepler_df['KID'].isna()]
    lamost_kepler_df['KID'] = lamost_kepler_df['KID'].astype(int)
    lamost_kepler_df = lamost_kepler_df.merge(kepler_df[META_COLUMNS], on='KID', how='inner')
    lamost_kepler_df['main_seq'] = lamost_kepler_df.apply(giant_cond, axis=1)
    lamost_kepler_df = lamost_kepler_df[lamost_kepler_df['main_seq']==True]
    lamost_kepler_df = lamost_kepler_df.dropna(subset=norm_cols)
    for col in norm_cols:
        lamost_kepler_df[col] = (lamost_kepler_df[col] - lamost_kepler_df[col].min()) / \
        (lamost_kepler_df[col].max() - lamost_kepler_df[col].min()) 
    train_df, val_df  = train_test_split(lamost_kepler_df, test_size=0.2, random_state=42)
    print("number of samples kepler: ", len(kepler_df),  " lamost-kepler :", len(lamost_kepler_df))
    return train_df, val_df 

def test_dataset(ds, num_iters=10):
    start = time.time()
    no_founds = 0
    for i in range(num_iters):
        start = time.time()
        light, spec, w,_, info,_ = ds[i]
        ds_time = time.time()-start
        print("time taken for dataset: ", ds_time)
        if light.sum() == 0:
            no_founds += 1
        # print(light.shape, spec.shape, w.shape, info.keys())
        fig, axes = plt.subplots(1,2, figsize=(24,14))
        axes[0].plot(light[0].cpu().numpy())
        axes[1].plot(spec[0].cpu().numpy())
        axes[0].set_title(f"Lightcurve: {info['KID']}")
        axes[1].set_title(f"Spectrum: {info['obsid']}")
        plt.savefig(f'/data/lightSpec/images/lightspec_{i}.png')
        plt.close()
    print("average time taken per iteration: ", (time.time()-start)/num_iters)


def umap_predictions(model, test_dataloader, data_args, datetime_dir, exp_num, local_rank):
    umap_lc = create_umap(model.module.lightcurve_encoder_q, test_dataloader, local_rank, use_w=False, dual=False)
    print("umap created: ", umap_lc.shape)
    umap_lc.to_csv(f"{data_args.log_dir}/{datetime_dir}/umap_lc_{exp_num}_ssl.csv", index=False)
    print("umap_lc saved in ", f"{data_args.log_dir}/{datetime_dir}/umap_lc_{exp_num}_ssl.csv")

    umap_spec = create_umap(model.module.spectra_encoder_q, test_dataloader, local_rank, use_w=False, dual=False)
    print("umap created: ", umap_spec.shape)
    umap_spec.to_csv(f"{data_args.log_dir}/{datetime_dir}/umap_spec_{exp_num}_ssl.csv", index=False)
    print("umap_spec saved in ", f"{data_args.log_dir}/{datetime_dir}/umap_spec_{exp_num}_ssl.csv")

    umap_dual = create_umap(model, test_dataloader, local_rank, use_w=False, dual=True)
    print("umap created: ", umap_dual.shape)
    umap_dual.to_csv(f"{data_args.log_dir}/{datetime_dir}/umap_dual_{exp_num}_ssl.csv", index=False)
    print("umap_dual saved in ", f"{data_args.log_dir}/{datetime_dir}/umap_dual_{exp_num}_ssl.csv")
    
    print(umap_lc['umap_x'] - umap_spec['umap_x'])
    print(umap_dual['umap_x'] - umap_spec['umap_x'])

    return umap_lc, umap_spec, umap_dual



local_rank, world_size, gpus_per_node = setup()
args_dir = '/data/lightSpec/nn/config_combined_embed.yaml'
data_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Data'])
exp_num = data_args.exp_num
model_name = data_args.model_name
if data_args.test_run:
    datetime_dir = f"test_{current_date}"
model_args = Container(**yaml.safe_load(open(args_dir, 'r'))[f'{model_name}'])
conformer_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Conformer'])
optim_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Optimization SSL'])
if not os.path.exists(f"{data_args.log_dir}/{datetime_dir}"):
    os.makedirs(f"{data_args.log_dir}/{datetime_dir}")

light_transforms = Compose([RandomCrop(int(data_args.max_days_lc/data_args.lc_freq)),
                            MovingAvg(13),
                            ACF(max_lag_day=None, max_len=int(data_args.max_len_lc)),
                            RandomMasking(mask_prob=0.15),
                            Normalize('std'),
                            ToTensor(),
                         ])
spec_transforms = Compose([LAMOSTSpectrumPreprocessor(continuum_norm=data_args.continuum_norm, plot_steps=False),
                            ToTensor()
                           ])

train_df, val_df = create_train_test_dfs()
val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=42) 

train_dataset = LightSpecDataset(df=train_df, light_transforms=light_transforms,
                                spec_transforms=spec_transforms,
                                npy_path = '/data/lightPred/data/npy',
                                spec_path = data_args.spectra_dir,
                                light_seq_len=int(data_args.max_len_lc),
                                use_acf=data_args.use_acf,
                                spec_seq_len=int(data_args.max_len_spectra),
                                meta_columns=None
                                )
val_dataset = LightSpecDataset(df=val_df, light_transforms=light_transforms,
                                spec_transforms=spec_transforms,
                                npy_path = '/data/lightPred/data/npy',
                                spec_path = data_args.spectra_dir,
                                light_seq_len=int(data_args.max_len_lc),
                                use_acf=data_args.use_acf,
                                spec_seq_len=int(data_args.max_len_spectra),
                                meta_columns=None
                                )

test_dataset = LightSpecDataset(df=test_df, light_transforms=light_transforms,
                                spec_transforms=spec_transforms,
                                npy_path = '/data/lightPred/data/npy',
                                spec_path = data_args.spectra_dir,
                                light_seq_len=int(data_args.max_len_lc),
                                use_acf=data_args.use_acf,
                                spec_seq_len=int(data_args.max_len_spectra),
                                meta_columns=None
                                )
# test_dataset(train_dataset, num_iters=100)


train_dataloader = create_unique_loader(train_dataset,
                                      batch_size=int(data_args.batch_size), \
                                      num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
                                      collate_fn=kepler_collate_fn )

val_dataloader = create_unique_loader(val_dataset,
                                    batch_size=int(data_args.batch_size),
                                    num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
                                    collate_fn=kepler_collate_fn,
                                    )

test_dataloader = create_unique_loader(test_dataset,
                                    batch_size=int(data_args.batch_size),
                                    num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
                                    collate_fn=kepler_collate_fn,
                                    )

print("len train dataloader ", len(train_dataloader))

backbone = MODELS[model_name](model_args, conformer_args=conformer_args)
model = SimSiam(backbone).to(local_rank)

checkpoint_dir = datetime_dir
checkpoint_exp = exp_num
if data_args.load_checkpoint:
    checkpoint_dir = os.path.basename(os.path.dirname(data_args.checkpoint_path))
    checkpoint_exp = os.path.basename(data_args.checkpoint_path).split('.')[0].split('_')[-1]
    print("loading checkpoint from: ", data_args.checkpoint_path)
    model = load_checkpoints_ddp(model, data_args.checkpoint_path)
    print("loaded checkpoint from: ", data_args.checkpoint_path)

model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainble parameters: {num_params}")
all_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters in model: {all_params}")
print("number of training samples: ", len(train_dataset), len(val_dataset))

if data_args.create_umap:
    umap_df = create_umap(model.module.encoder, val_dataloader, local_rank, use_w=False, dual=False, combine=True)
    print("umap created: ", umap_df.shape)
    umap_df.to_csv(f"{data_args.log_dir}/{checkpoint_dir}/umap_{checkpoint_exp}_ssl.csv", index=False)
    print(f"umap saved at {data_args.log_dir}/{checkpoint_dir}/umap_{checkpoint_exp}_ssl.csv")
    exit()


loss_fn = torch.nn.CrossEntropyLoss()
if optim_args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(),
                                 lr=float(optim_args.max_lr),
                                momentum=float(optim_args.momentum),
                                weight_decay=float(optim_args.weight_decay),
                                nesterov=optim_args.nesterov)
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(optim_args.max_lr),
    weight_decay=float(optim_args.weight_decay))
scaler = GradScaler()

# print all the trainable layers in the model
# print("Trainable layers in the model: ")
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.shape)

trainer = LightSpecTrainer(model=model, optimizer=optimizer,
                        criterion=loss_fn, output_dim=1, scaler=scaler, grad_clip=True,
                       scheduler=None, train_dataloader=train_dataloader,
                       val_dataloader=val_dataloader, device=local_rank,
                           exp_num=datetime_dir, log_path=data_args.log_dir, range_update=None,
                           accumulation_step=1, max_iter=np.inf,
                        exp_name=f"combined_{exp_num}") 

# save all containers in log directory
config_save_path = f"{data_args.log_dir}/{datetime_dir}/lightspec_{exp_num}_complete_config.yaml"

complete_config = {
    "data_args": data_args.__dict__,
    "model_args": model_args.__dict__,
    "conformer_args": conformer_args.__dict__,
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
                        early_stopping=40, best='loss', conf=True) 
output_filename = f'{data_args.log_dir}/{datetime_dir}/combined_{exp_num}.json'
with open(output_filename, "w") as f:
    json.dump(fit_res, f, indent=2)
fig, axes = plot_fit(fit_res, legend=exp_num, train_test_overlay=True)
plt.savefig(f"{data_args.log_dir}/{datetime_dir}/combined_fit_{exp_num}.png")
plt.clf()

umap_lc, umap_spec, umap_dual = umap_predictions(model, test_dataloader, data_args, datetime_dir, exp_num, local_rank)
