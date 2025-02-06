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
from util.utils import *
from dataset.dataset import LightSpecDataset, create_unique_loader, SpectraDataset, KeplerDataset
from dataset.sampler import DistinctParameterSampler
from nn.astroconf import Astroconformer, AstroEncoderDecoder
from nn.models import CNNEncoder, CNNEncoderDecoder, MultiEncoder, MultiTaskRegressor
from nn.mlp import MLPEncoder
from nn.simsiam import MultiModalSimSiam, MultiModalSimCLR
from nn.moco import MoCo, MultimodalMoCo
from nn.simsiam import SimSiam, projection_MLP
from nn.optim import QuantileLoss, CQR
from nn.utils import init_model, load_checkpoints_ddp
from nn.train import ContrastiveTrainer, MultiResolutionTrainer, DualTrainer
from tests.test_unique_sampler import run_sampler_tests
from features import create_umap

META_COLUMNS = ['KID', 'Teff', 'logg', 'FeH', 'Rstar', 'Mstar', 'kmag_abs']

MODELS = {'Astroconformer': Astroconformer, 'CNNEncoder': CNNEncoder, 'MultiEncoder': MultiEncoder, 'MultiTaskRegressor': MultiTaskRegressor,
           'AstroEncoderDecoder': AstroEncoderDecoder, 'CNNEncoderDecoder': CNNEncoderDecoder,}

torch.cuda.empty_cache()

torch.manual_seed(1234)
np.random.seed(1234)


current_date = datetime.date.today().strftime("%Y-%m-%d")
datetime_dir = f"lightspec_{current_date}"

def create_single_datasets(dataset, test_size=0.2):
    indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=test_size, random_state=42)
    val_indices, test_indices = train_test_split(val_indices, test_size=0.5, random_state=42)
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)
    return train_subset, train_subset, val_subset, test_subset

def create_single_dataloaders(train_subset, val_subset, test_subset, b_size=None):

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

def create_lamost_data(data_args, transforms):
    
    lamost_catalog = pd.read_csv('/data/lamost/lamost_afgkm_teff_3000_7500_catalog.csv', sep='|')
    lamost_catalog = lamost_catalog.drop_duplicates(subset=['combined_obsid'])
    lamost_catalog = lamost_catalog[lamost_catalog['combined_snrg'] > 0]
    lamost_catalog = lamost_catalog.dropna(subset=['combined_teff', 'combined_logg', 'combined_feh'])
    lamost_dataset = SpectraDataset(data_args.data_dir, transforms=transforms, df=lamost_catalog, 
                                    max_len=int(data_args.max_len_spectra))

    ds_full, train_subset, val_subset, test_subset = create_single_datasets(lamost_dataset)

    print("number of samples: ", len(lamost_catalog))
    print("train samples: ", len(train_subset))
    print("val samples: ", len(val_subset))
    print("test samples: ", len(test_subset))

    lamost_apogee = pd.read_csv('/data/lamost/crossmatched_catalog.csv')
    lamost_apogee = lamost_apogee[lamost_apogee['APOGEE_SNR'] > 0]
    lamost_apogee = lamost_apogee.dropna(subset=['APOGEE_VSINI','APOGEE_TEFF', 'APOGEE_LOGG', 'APOGEE_FE_H'])
    lamost_apogee_dataset = SpectraDataset(data_args.data_dir, transforms=transforms, df=lamost_apogee, 
                                    max_len=int(data_args.max_len_spectra))
    ds_full_hr, train_subset_hr, val_subset_hr, test_subset_hr = create_single_datasets(lamost_apogee_dataset)

    ds_ratio = len(ds_full_hr) / len(ds_full)
    
    train_dl, val_dl, test_dl = create_single_dataloaders(train_subset, val_subset, test_subset)
    train_dl_hr, val_dl_hr, test_dl_hr = create_single_dataloaders(train_subset_hr, val_subset_hr, test_subset_hr)
    return train_dl, val_dl, test_dl, train_dl_hr, val_dl_hr, test_dl_hr, ds_ratio


def create_kepler_data(data_args, transforms):
    kepler_df = get_all_samples_df(num_qs=None, read_from_csv=True)
    print("number of samples: ", len(kepler_df))
    kepler_dataset = KeplerDataset(df=kepler_df, transforms=transforms,
                                    target_transforms=transforms,
                                    npy_path = '/data/lightPred/data/npy',
                                    seq_len=int(data_args.max_len_lc),
                                    masked_transforms = data_args.masked_transform
                                    )
    ds_full, train_subset, val_subset, test_subset = create_single_datasets(kepler_dataset)

    print("number of samples: ", len(kepler_df))
    print("train samples: ", len(train_subset))
    print("val samples: ", len(val_subset))
    print("test samples: ", len(test_subset))

    train_dl, val_dl, test_dl = create_single_dataloaders(train_subset, val_subset, test_subset)
    return train_dl, val_dl, test_dl



def create_lightspec_data(data_args, spec_transforms, light_transforms, norm_cols=['Teff', 'logg', 'Mstar']):
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

    train_dataset = LightSpecDataset(df=train_df, light_transforms=light_transforms,
                                spec_transforms=spec_transforms,
                                npy_path = '/data/lightPred/data/npy',
                                spec_path = data_args.spectra_dir,
                                light_seq_len=int(data_args.max_days_lc/data_args.lc_freq),
                                spec_seq_len=int(data_args.max_len_spectra),
                                meta_columns=data_args.meta_columns
                                )
    val_dataset = LightSpecDataset(df=val_df, light_transforms=light_transforms,
                                    spec_transforms=spec_transforms,
                                    npy_path = '/data/lightPred/data/npy',
                                    spec_path = data_args.spectra_dir,
                                    light_seq_len=int(data_args.max_days_lc/data_args.lc_freq),
                                    spec_seq_len=int(data_args.max_len_spectra),
                                    meta_columns=data_args.meta_columns
                                    )
    train_dataloader = create_unique_loader(train_dataset,
                                      batch_size=int(data_args.batch_size), \
                                      num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
                                      collate_fn=kepler_collate_fn )

    val_dataloader = create_unique_loader(val_dataset,
                                        batch_size=int(data_args.batch_size),
                                        num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
                                        collate_fn=kepler_collate_fn,
                                        )

    return train_dataset, val_dataset, train_dataloader, val_dataloader

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
        print(light.shape, spec.shape, w.shape, info.keys())
        fig, axes = plt.subplots(1,2, figsize=(24,14))
        axes[0].plot(light[0].cpu().numpy())
        axes[1].plot(spec[0].cpu().numpy())
        axes[0].set_title(f"Lightcurve: {info['KID']}")
        axes[1].set_title(f"Spectrum: {info['obsid']}")
        plt.savefig(f'/data/lightSpec/images/lightspec_{i}.png')
        plt.close()
    print("average time taken per iteration: ", (time.time()-start)/num_iters)



local_rank, world_size, gpus_per_node = setup()
args_dir = '/data/lightSpec/nn/config_lightspec_full.yaml'
data_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Data'])
exp_num = data_args.exp_num
light_model_name = data_args.light_model_name
spec_model_name = data_args.spec_model_name
if data_args.test_run:
    datetime_dir = f"test_{current_date}"
light_model_args = Container(**yaml.safe_load(open(args_dir, 'r'))[f'{light_model_name}_lc'])
spec_model_args = Container(**yaml.safe_load(open(args_dir, 'r'))[f'{spec_model_name}_spec'])
conformer_args_lc = Container(**yaml.safe_load(open(args_dir, 'r'))['Conformer_lc'])
conformer_args_spec = Container(**yaml.safe_load(open(args_dir, 'r'))['Conformer_spec'])
lightspec_args = Container(**yaml.safe_load(open(args_dir, 'r'))['MultiEncoder_lightspec'])
conformer_args_lightspec = Container(**yaml.safe_load(open(args_dir, 'r'))['Conformer_lightspec'])
sims_args = Container(**yaml.safe_load(open(args_dir, 'r'))['MultiModalSimSiam'])
# backbone_args = Container(**yaml.safe_load(open(args_dir, 'r'))['CNNBackbone'])
moco_args = Container(**yaml.safe_load(open(args_dir, 'r'))['MoCo'])
optim_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Optimization'])
if not os.path.exists(f"{data_args.log_dir}/{datetime_dir}"):
    os.makedirs(f"{data_args.log_dir}/{datetime_dir}")

spec_transforms = Compose([LAMOSTSpectrumPreprocessor(continuum_norm=data_args.continuum_norm, plot_steps=False),
                            ToTensor()
                           ])

lc_transforms = Compose([RandomCrop(int(data_args.max_days_lc/data_args.lc_freq)),
                            MovingAvg(13),
                            Normalize('std'),
                            ToTensor(),
                         ])

train_dl_spec, val_dl_spec, test_dl_spec, train_dl_spec_hr, \
val_dl_spec_hr, test_dl_spec_hr, ds_ratio = create_lamost_data(data_args, transforms=spec_transforms)
train_dl_lc, val_dl_lc, test_dl_lc = create_kepler_data(data_args, transforms=lc_transforms)

norm_cols = data_args.meta_columns if not data_args.create_umap else []
 
train_ds, val_ds, train_dl, val_dl = create_lightspec_data(data_args,  spec_transforms=spec_transforms,
                                 light_transforms=lc_transforms, norm_cols=norm_cols )

print("len train dataloader ", len(train_dl), len(val_dl))

light_backbone = MODELS[light_model_name](light_model_args, conformer_args=conformer_args_lc)
if light_model_name == 'Astroconformer':
    light_backbone.pred_layer = torch.nn.Identity()
light_model = SimSiam(light_backbone)

light_model = init_model(light_model, light_model_args)

spec_model = MODELS[spec_model_name](spec_model_args, conformer_args=conformer_args_spec)

spec_model = init_model(spec_model, spec_model_args)

model = MultimodalMoCo(spec_model.encoder, light_model.backbone,  **moco_args.get_dict()).to(local_rank)

# backbone = MultiEncoder(lightspec_args, conformer_args=conformer_args_lightspec)
# model = MultiModalSimSiam(backbone, spec_model.encoder, light_model.backbone, sims_args).to(local_rank)

if data_args.load_checkpoint:
    datetime_dir = os.path.basename(os.path.dirname(data_args.checkpoint_path))
    exp_num = os.path.basename(data_args.checkpoint_path).split('.')[0].split('_')[-1]
    print(datetime_dir)
    print("loading checkpoint from: ", data_args.checkpoint_path)
    model = load_checkpoints_ddp(model, data_args.checkpoint_path)
    print("loaded checkpoint from: ", data_args.checkpoint_path)

model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters: {num_params}")
print("number of training samples: ", len(train_ds), len(val_ds))

if data_args.create_umap:
    umap_df = create_umap(model, val_dl, local_rank)
    print("umap created: ", umap_df.shape)
    umap_df.to_csv(f"{data_args.log_dir}/{datetime_dir}/umap_{exp_num}.csv", index=False)
    exit()

loss_spec = CQR(quantiles=optim_args.quantiles, reduction='none')
loss_spec_ssl = torch.nn.MSELoss()
optimizer_spec = torch.optim.Adam(model.parameters(), lr=float(optim_args.max_lr), weight_decay=float(optim_args.weight_decay))

spec_trainer = MultiResolutionTrainer(model=spec_model, optimizer=optimizer_spec,
                        criterion=loss_spec, ssl_criterion=loss_spec_ssl, output_dim=spec_model_args.output_dim, scaler=None,
                       scheduler=None, train_dataloader=train_dl_spec, high_res_train=train_dl_spec_hr,
                        num_quantiles=spec_model_args.num_quantiles,
                       val_dataloader=val_dl_spec, high_res_val=val_dl_spec_hr, lambda_high_res=1.5, device=local_rank,
                           exp_num=datetime_dir, log_path=data_args.log_dir, range_update=None,
                           accumulation_step=1, max_iter=np.inf, w_name='snrg',
                           w_init_val=1,  exp_name=f"{spec_model_name}_spectra_decode_multires_{exp_num}") 

loss_lc = torch.nn.MSELoss()
optimizer_lc = torch.optim.AdamW(model.parameters(), lr=float(optim_args.max_lr), weight_decay=float(optim_args.weight_decay))
scaler = GradScaler()

light_trainer  = ContrastiveTrainer(model=light_model, optimizer=optimizer_lc,
                            criterion=loss_lc, output_dim=1, scaler=scaler,
                        scheduler=None, train_dataloader=train_dl_lc,
                        val_dataloader=val_dl_lc, device=local_rank,
                            exp_num=datetime_dir, log_path=data_args.log_dir, range_update=None,
                            accumulation_step=1, max_iter=np.inf,
                            exp_name=f"{light_model_name}_lc_{exp_num}")

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

# print all the trainable layers in the model
# print("Trainable layers in the model: ")
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.shape)

trainer = DualTrainer(model=model, trainer_lc=light_trainer, trainer_spec=spec_trainer, optimizer=optimizer,
                        criterion=loss_fn, output_dim=1, scaler=scaler, grad_clip=True,
                       scheduler=None, train_dataloader=train_dl,
                       val_dataloader=val_dl, device=local_rank, lambda_dual=1,
                           exp_num=datetime_dir, log_path=data_args.log_dir, range_update=None,
                           accumulation_step=1, max_iter=np.inf, 
                        exp_name=f"lightspec_{exp_num}") 

# save all containers in log directory
config_save_path = f"{data_args.log_dir}/{datetime_dir}/lightspec_{exp_num}_complete_config.yaml"

complete_config = {
    "data_args": data_args.__dict__,
    "light_model_args": light_model_args.__dict__,
    "spec_model_args": spec_model_args.__dict__,
    "conformer_args_lc": conformer_args_lc.__dict__,
    "conformer_args_spec": conformer_args_spec.__dict__,
    "moco_args": moco_args.__dict__,
    "optim_args": optim_args.__dict__,
    "num_params": num_params,
    "model_structure": str(model),
    "light_transforms": str(lc_transforms),
    "spec_transforms": str(spec_transforms),
    "spec_trainer": spec_trainer.__dict__,
    "light_trainer": light_trainer.__dict__,
    "trainer": str(trainer)
}

# Save the complete configuration to a YAML file
with open(config_save_path, "w") as config_file:
    yaml.dump(complete_config, config_file, default_flow_style=False)
print(f"Configuration (with model structure) saved at {config_save_path}.")

fit_res = trainer.fit(num_epochs=data_args.num_epochs, device=local_rank,
                        early_stopping=40, only_p=False, best='loss', conf=True) 
output_filename = f'{data_args.log_dir}/{datetime_dir}/lightspec_{exp_num}.json'
with open(output_filename, "w") as f:
    json.dump(fit_res, f, indent=2)
fig, axes = plot_fit(fit_res, legend=exp_num, train_test_overlay=True)
plt.savefig(f"{data_args.log_dir}/{datetime_dir}/lightspec_fit_{exp_num}.png")
plt.clf()
