
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
from nn.models import CNNEncoder, CNNEncoderDecoder, MultiEncoder, MultiTaskRegressor, MultiTaskSimSiam
from nn.simsiam import MultiModalSimSiam, MultiModalSimCLR
from nn.moco import MultimodalMoCo, PredictiveMoco, MultiTaskMoCo
from nn.simsiam import SimSiam, projection_MLP
from nn.utils import init_model, load_checkpoints_ddp
from util.utils import *
from nn.optim import CQR
from nn.train import ContrastiveTrainer
from tests.test_unique_sampler import run_sampler_tests
from features import create_umap

META_COLUMNS = ['KID', 'Teff', 'logg', 'FeH', 'Rstar', 'Mstar', 'Dist', 'kmag_abs', 'RUWE']

MODELS = {'Astroconformer': Astroconformer, 'CNNEncoder': CNNEncoder, 'MultiEncoder': MultiEncoder, 'MultiTaskRegressor': MultiTaskRegressor,
          'AstroEncoderDecoder': AstroEncoderDecoder, 'CNNEncoderDecoder': CNNEncoderDecoder, 'MultiTaskSimSiam': MultiTaskSimSiam,}

torch.cuda.empty_cache()

torch.manual_seed(1234)
np.random.seed(1234)


current_date = datetime.date.today().strftime("%Y-%m-%d")
datetime_dir = f"lightspec_{current_date}"

def create_train_test_dfs(norm_cols=['Teff', 'logg', 'Mstar']):
    kepler_df = get_all_samples_df(num_qs=None, read_from_csv=True)
    kepler_meta = pd.read_csv('/data/lightPred/tables/berger_catalog_full.csv')
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
    train_df, val_df  = train_test_split(lamost_kepler_df, test_size=0.2, random_state=42)
    print("number of samples kepler: ", len(kepler_df),  " lamost-kepler :", len(lamost_kepler_df))
    return train_df, val_df 

def test_dataset_samples(ds, num_iters=10):
    start = time.time()
    no_founds = 0
    for i in range(num_iters):
        start = time.time()
        light, spec, y,_, _,info = ds[i]
        ds_time = time.time()-start
        print("labels: ", y, "nans: ", torch.isnan(light).sum(), torch.isnan(spec).sum(), "max: ", light.max(), spec.max(), "min: ", light.min(), spec.min())
        if light.sum() == 0:
            no_founds += 1
        # print(light.shape, spec.shape, w.shape, info.keys())
        if i % 10 == 0:
            fig, axes = plt.subplots(1,2, figsize=(24,14))
            axes[0].plot(light[0].cpu().numpy())
            axes[1].plot(spec.cpu().numpy())
            axes[0].set_title(f"Lightcurve: {info['KID']}")
            axes[1].set_title(f"Spectrum: {info['obsid']}")
            plt.savefig(f'/data/lightSpec/images/lightspec_{i}.png')
            plt.close()
    print("average time taken per iteration: ", (time.time()-start)/num_iters)



local_rank, world_size, gpus_per_node = setup()
args_dir = '/data/lightSpec/nn/config_lightspec_ssl.yaml'
data_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Data'])
exp_num = data_args.exp_num
light_model_name = data_args.light_model_name
spec_model_name = data_args.spec_model_name
combined_model_name = data_args.combined_model_name
if data_args.test_run:
    datetime_dir = f"test_{current_date}"
light_model_args = Container(**yaml.safe_load(open(args_dir, 'r'))[f'{light_model_name}_lc'])
spec_model_args = Container(**yaml.safe_load(open(args_dir, 'r'))[f'{spec_model_name}_spec'])
combined_model_args = Container(**yaml.safe_load(open(args_dir, 'r'))[f'{combined_model_name}_combined'])
conformer_args_lc = Container(**yaml.safe_load(open(args_dir, 'r'))['Conformer_lc'])
conformer_args_spec = Container(**yaml.safe_load(open(args_dir, 'r'))['Conformer_spec'])
conformer_args_combined = Container(**yaml.safe_load(open(args_dir, 'r'))['Conformer_combined'])
lightspec_args = Container(**yaml.safe_load(open(args_dir, 'r'))['MultiEncoder_lightspec'])
transformer_args_lightspec = Container(**yaml.safe_load(open(args_dir, 'r'))['Transformer_lightspec'])
predictor_args = Container(**yaml.safe_load(open(args_dir, 'r'))['predictor'])
moco_pred_args = Container(**yaml.safe_load(open(args_dir, 'r'))['reg_predictor'])
loss_args = Container(**yaml.safe_load(open(args_dir, 'r'))['loss'])
sims_args = Container(**yaml.safe_load(open(args_dir, 'r'))['MultiModalSimSiam'])
# backbone_args = Container(**yaml.safe_load(open(args_dir, 'r'))['CNNBackbone'])
moco_args = Container(**yaml.safe_load(open(args_dir, 'r'))['MoCo'])
optim_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Optimization'])
os.makedirs(f"{data_args.log_dir}/{datetime_dir}", exist_ok=True)

moco_pred_args.out_dim = len(data_args.labels) * len(optim_args.quantiles)

light_transforms = Compose([RandomCrop(int(data_args.max_days_lc/data_args.lc_freq)),
                            MovingAvg(13),
                            ACF(max_lag_day=None, max_len=int(data_args.max_len_lc)),
                            Normalize('dist_median'),
                            ToTensor(),
                         ])
spec_transforms = Compose([LAMOSTSpectrumPreprocessor(continuum_norm=data_args.continuum_norm, plot_steps=False),
                            ToTensor()
                           ])

norm_cols = data_args.meta_columns if not data_args.create_umap else []
train_df, val_df = create_train_test_dfs(norm_cols=norm_cols)
val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=42) 

train_dataset = LightSpecDataset(df=train_df, light_transforms=light_transforms,
                                spec_transforms=spec_transforms,
                                npy_path = '/data/lightPred/data/raw_npy',
                                spec_path = data_args.spectra_dir,
                                light_seq_len=int(data_args.max_len_lc),
                                spec_seq_len=int(data_args.max_len_spectra),
                                meta_columns=data_args.meta_columns,
                                use_acf=data_args.use_acf,
                                 scale_flux=data_args.scale_flux,
                                 labels=data_args.labels
                                )
val_dataset = LightSpecDataset(df=val_df, light_transforms=light_transforms,
                                spec_transforms=spec_transforms,
                                npy_path = '/data/lightPred/data/raw_npy',
                                spec_path = data_args.spectra_dir,
                                light_seq_len=int(data_args.max_len_lc),
                                spec_seq_len=int(data_args.max_len_spectra),
                                meta_columns=data_args.meta_columns,
                                use_acf=data_args.use_acf,
                               scale_flux=data_args.scale_flux,
                               labels=data_args.labels
                                )

test_dataset = LightSpecDataset(df=test_df, light_transforms=light_transforms,
                                spec_transforms=spec_transforms,
                                npy_path = '/data/lightPred/data/raw_npy',
                                spec_path = data_args.spectra_dir,
                                light_seq_len=int(data_args.max_len_lc),
                                spec_seq_len=int(data_args.max_len_spectra),
                                meta_columns=data_args.meta_columns,
                                use_acf=data_args.use_acf,
                                scale_flux=data_args.scale_flux,
                                labels=data_args.labels
                                )
# test_dataset_samples(train_dataset, num_iters=100)

train_dataloader = create_unique_loader(train_dataset,
                                      batch_size=int(data_args.batch_size), \
                                      num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
                                      collate_fn=kepler_collate_fn )

val_dataloader = create_unique_loader(val_dataset,
                                    batch_size=int(data_args.batch_size),
                                    num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
                                    collate_fn=kepler_collate_fn,
                                     drop_last=True
                                    )

test_dataloader = create_unique_loader(test_dataset,
                                    batch_size=int(data_args.batch_size),
                                    num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
                                    collate_fn=kepler_collate_fn,
                                    drop_last=True
                                    )

print("len train dataloader ", len(train_dataloader))

light_model = MODELS[light_model_name](light_model_args, conformer_args=conformer_args_lc)
# light_model = SimSiam(light_backbone)
# light_model = light_backbone

light_model = init_model(light_model, light_model_args)

num_params_lc_all = sum(p.numel() for p in light_model.parameters() if p.requires_grad)
print(f"Number of trainble parameters in light model: {num_params_lc_all}")
num_params_lc = sum(p.numel() for p in light_model.simsiam.encoder.parameters() if p.requires_grad)
print(f"Number of trainble parameters in lc encoder: {num_params_lc}")

spec_model = MODELS[spec_model_name](spec_model_args, conformer_args=conformer_args_spec)

spec_model = init_model(spec_model, spec_model_args)

num_params_spec_all = sum(p.numel() for p in spec_model.parameters() if p.requires_grad)
print(f"Number of trainble parameters in spec model: {num_params_spec_all}")
num_params_spec = sum(p.numel() for p in spec_model.encoder.parameters() if p.requires_grad)
print(f"Number of trainble parameters in spec encoder: {num_params_spec}")


if data_args.combined_embed:

    combined_backbone = MODELS[combined_model_name](combined_model_args, conformer_args=conformer_args_combined)

    combined_model = SimSiam(combined_backbone)

    combined_model = init_model(combined_model, combined_model_args)

    combined_encoder = combined_model.encoder

    num_params_combined = sum(p.numel() for p in combined_encoder.parameters() if p.requires_grad)
    print(f"Number of trainble parameters in combined encoder: {num_params_combined}")

else:
    combined_encoder = None
# backbone = Transformer(transformer_args_lightspec)

# model = MultimodalMoCo(spec_model.encoder, light_model.encoder, transformer_args_lightspec,  **moco_args.get_dict()).to(local_rank)
# model = MultiModalSimSiam(backbone, spec_model.encoder, light_model.backbone, sims_args).to(local_rank)
moco = PredictiveMoco(spec_model.encoder, light_model.simsiam.encoder,
                         transformer_args_lightspec,
                         predictor_args.get_dict(),
                         loss_args,
                        combined_encoder=combined_encoder,
                        **moco_args.get_dict()).to(local_rank)

model = MultiTaskMoCo(moco, moco_pred_args.get_dict()).to(local_rank)

if data_args.load_checkpoint:
    datetime_dir = os.path.basename(os.path.dirname(data_args.checkpoint_path))
    exp_num = os.path.basename(data_args.checkpoint_path).split('.')[0].split('_')[-1]
    print(datetime_dir)
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
    model.module.calc_loss = False
    umap_df = create_umap(model, test_dataloader, local_rank, use_w=True, dual=True, logits_key='q')
    print("umap created: ", umap_df.shape)
    print(umap_df['Teff'].min(), umap_df['Teff'].max())
    umap_df.to_csv(f"{data_args.log_dir}/{datetime_dir}/umap_{exp_num}_ssl.csv", index=False)
    print(f"umap saved at {data_args.log_dir}/{datetime_dir}/umap_{exp_num}_ssl.csv")
    exit()
if data_args.approach == 'ssl':
    loss_fn = None
    num_quantiles = 1
elif data_args.approach == 'multitask':
    loss_fn = CQR(quantiles=optim_args.quantiles)
    num_quantiles = len(optim_args.quantiles)
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
accumulation_step = 1
trainer = ContrastiveTrainer(model=model, optimizer=optimizer,
                        criterion=loss_fn, output_dim=1, scaler=scaler, grad_clip=True,
                       scheduler=None, train_dataloader=train_dataloader,
                       val_dataloader=val_dataloader, device=local_rank, num_quantiles=num_quantiles,
                             exp_num=datetime_dir, log_path=data_args.log_dir, range_update=None,
                             accumulation_step=accumulation_step, max_iter=np.inf, stack_pairs=False, use_w=True,
                           use_pred_coeff=True, pred_coeff_val=data_args.pred_coeff_val,
                        exp_name=f"lightspec_{exp_num}") 
# save all containers in log directory
config_save_path = f"{data_args.log_dir}/{datetime_dir}/lightspec_{exp_num}_complete_config.yaml"

complete_config = {
    "data_args": data_args.__dict__,
    "light_model_args": light_model_args.__dict__,
    "spec_model_args": spec_model_args.__dict__,
    "conformer_args_lc": conformer_args_lc.__dict__,
    "conformer_args_spec": conformer_args_spec.__dict__,
    "transformer_args_lightspec": transformer_args_lightspec.__dict__,
    "moco_args": moco_args.__dict__,
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
                        early_stopping=20, best='loss', conf=True) 
output_filename = f'{data_args.log_dir}/{datetime_dir}/lightspec_{exp_num}_fit_res.json'
with open(output_filename, "w") as f:
    json.dump(fit_res, f, indent=2)
fig, axes = plot_fit(fit_res, legend=exp_num, train_test_overlay=True)
plt.savefig(f"{data_args.log_dir}/{datetime_dir}/lightspec_fit_{exp_num}.png")
plt.clf()


model.module.calc_loss = False
model.eval()
umap_df = create_umap(model, test_dataloader, local_rank, use_w=True, dual=True, logits_key='q')
print("umap created: ", umap_df.shape)
print(umap_df.head())
umap_df.to_csv(f"{data_args.log_dir}/{datetime_dir}/umap_{exp_num}_ssl.csv", index=False)
print(f"umap saved at {data_args.log_dir}/{datetime_dir}/umap_{exp_num}_ssl.csv")
exit()


