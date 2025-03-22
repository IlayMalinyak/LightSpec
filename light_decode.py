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
from nn.models import * 
# from nn.mamba import MambaEncoder
from nn.simsiam import SimSiam
from nn.train import RegressorTrainer, ContrastiveTrainer, MaskedRegressorTrainer, ContrastiveRegressorTrainer, DoubleInputTrainer
from nn.utils import deepnorm_init, load_checkpoints_ddp, get_lightPred_model
from nn.optim import CQR, TotalEnergyLoss, SumLoss, StephanBoltzmanLoss
from util.utils import *
from util.cgs_consts import *
from features import create_umap

set_seed(42)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
torch.cuda.empty_cache()

models = {'Astroconformer': Astroconformer, 'CNNEncoder': CNNEncoder, 'MultiEncoder': MultiEncoder, 'MultiTaskSimSiam': MultiTaskSimSiam,
          "CNNEncoderDecoder": CNNEncoderDecoder, 'CNNRegressor': CNNRegressor, 'MultiTaskRegressor': MultiTaskRegressor}

current_date = datetime.date.today().strftime("%Y-%m-%d")
datetime_dir = f"light_{current_date}"

local_rank, world_size, gpus_per_node = setup()
args_dir = '/data/lightSpec/nn/config_lc_decode2.yaml'
data_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Data'])
model_name = data_args.model_name
conformer_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Conformer'])
exp_num = data_args.exp_num
model_args = Container(**yaml.safe_load(open(args_dir, 'r'))[model_name])
lstm_args = Container(**yaml.safe_load(open(args_dir, 'r'))['LSTMEncoder'])
cnn_args = Container(**yaml.safe_load(open(args_dir, 'r'))['CNNEncoder'])
regressor_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Regressor'])
optim_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Optimization'])
astroconf_args = Container(**yaml.safe_load(open(args_dir, 'r'))['AstroConformer'])
os.makedirs(f"{data_args.log_dir}/{datetime_dir}", exist_ok=True)
regressor_args.output_dim = len(data_args.labels)
regressor_args.num_quantiles = len(optim_args.quantiles)
model_args.num_quantiles = len(optim_args.quantiles)
regressor_args.seq_len = int(data_args.max_len_lc)
lstm_args.seq_len = int(data_args.max_len_lc)
astroconf_args.encoder_only = True

transforms = Compose([ RandomCrop(int(data_args.max_len_lc)),
                        MovingAvg(13),
                        ACF(max_lag_day=None, max_len=int(data_args.max_len_lc)),
                        FFT(seq_len=int(data_args.max_len_lc)),
                        Normalize(['dist_median', 'std']),
                        ToTensor(), ])

target_transforms = transforms if not data_args.masked_transform else None

kepler_df = get_all_samples_df(num_qs=None, read_from_csv=True)
berger_cat = pd.read_csv('/data/lightPred/tables/berger_catalog_full.csv')
kepler_meta = pd.read_csv('/data/lightPred/tables/kepler_dr25_meta_data.csv')
period_df = pd.read_csv('/data/lightPred/tables/kepler_predictions_clean_seg_0_1_2_median.csv')
santos_df = pd.read_csv('/data/lightPred/tables/santos_periods_19_21.csv')
kepler_df = kepler_df.merge(berger_cat, on='KID').merge(kepler_meta[['KID', 'KMAG']], on='KID').merge(period_df[['KID', 'predicted period']], on='KID') 
kepler_df['kmag_abs'] = kepler_df['KMAG'] - 5 * np.log10(kepler_df['Dist']) + 5
kepler_df.dropna(subset=data_args.labels, inplace=True)
print("number of samples: ", len(kepler_df))
print("labels: ", data_args.labels)
for label in data_args.labels:
    print(kepler_df[label].describe())
train_dataset = KeplerDataset(df=kepler_df, transforms=transforms,
                              target_transforms=target_transforms,
                                npy_path = '/data/lightPred/data/raw_npy',
                                seq_len=int(data_args.max_len_lc),
                                masked_transforms = data_args.masked_transform,
                                use_acf=data_args.use_acf,
                                use_fft=data_args.use_fft,
                                scale_flux=data_args.scale_flux,
                                labels=data_args.labels,
                                dims=model_args.in_channels,
                                )
start = time.time()
max_std = 0
for i in range(100):
    x1,x2,y,_,info1,info2 = train_dataset[i]
    std = x1[0].std()
    # if std > max_std:
    #     max_std = std
    # if i % 1000 == 0:
    #     print(i, max_std)
    print(x1.shape, x2.shape, x1[0].max(), x1[0].std())
    # if i % 10 == 0:
    #     fig, ax = plt.subplots(1, 2)
    #     ax[0].plot(x2[0])
    #     ax[1].plot(x2[1])
    #     plt.savefig(f'/data/lightSpec/images/lc_{i}.png')
print("average time taken per iteration: ", (time.time()-start)/100)
indices = list(range(len(train_dataset)))
print("number of samples: ", len(indices))
train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
val_indices, test_indices = train_test_split(val_indices, test_size=0.5, random_state=42)
train_subset = Subset(train_dataset, train_indices)
val_subset = Subset(train_dataset, val_indices)
test_subset = Subset(train_dataset, test_indices)

# plt.hist(kepler_df.loc[train_indices, 'RUWE'], bins=20,label='train', histtype='step')
# plt.hist(kepler_df.loc[val_indices, 'RUWE'], bins=20,label='val', histtype='step')
# plt.hist(kepler_df.loc[test_indices, 'RUWE'], bins=20,label='test', histtype='step')
# plt.legend()
# plt.savefig(f"{data_args.log_dir}/{datetime_dir}/ruwe_split.png")
# plt.close()
#
# plt.hist(kepler_df.loc[train_indices, 'Teff'], bins=20,label='train', histtype='step')
# plt.hist(kepler_df.loc[val_indices, 'Teff'], bins=20,label='val', histtype='step')
# plt.hist(kepler_df.loc[test_indices, 'Teff'], bins=20,label='test', histtype='step')
# plt.legend()
# plt.savefig(f"{data_args.log_dir}/{datetime_dir}/teff_split.png")
# plt.close()
# exit()
train_sampler = torch.utils.data.distributed.DistributedSampler(train_subset, num_replicas=world_size, rank=local_rank)
train_dataloader = DataLoader(train_subset,
                              batch_size=int(data_args.batch_size), \
                              num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
                              collate_fn=kepler_collate_fn,
                              sampler=train_sampler,
                              drop_last=True)


val_sampler = torch.utils.data.distributed.DistributedSampler(val_subset, num_replicas=world_size, rank=local_rank)
val_dataloader = DataLoader(val_subset,
                            batch_size=int(data_args.batch_size),
                            collate_fn=kepler_collate_fn,
                            sampler=val_sampler, \
                            num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
                            drop_last=True)

test_dataloader = DataLoader(test_subset,
                             batch_size=int(data_args.batch_size),
                             collate_fn=kepler_collate_fn,
                             num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
                             drop_last=True)

# conformer_args.in_channels = 1
# conformer_args.output_dim = len(data_args.labels)
# dual_model = Astroconformer(astroconf_args)
# dual_model.pred_layer = torch.nn.Identity()
# lstm_args = {'seq_len': int(data_args.max_len_lc), 'hidden_size': 128, 'num_layers': 5, 'num_classes': len(data_args.labels) * len(optim_args.quantiles), 'in_channels': 1}
# model = LSTM_DUAL_LEGACY(dual_model,astroconf_args.encoder_dim, lstm_args, num_classes=len(data_args.labels) * len(optim_args.quantiles)).to(local_rank)
# backbone = Astroconformer(model_args)
# backbone.pred_layer = torch.nn.Identity()
# backbone = CNNEncoder(model_args)
# model = models[model_name](model_args, conformer_args=conformer_args).to(local_rank)
# backbone = MambaEncoder(model_args)
# model = SimSiam(backbone)
# model = get_lightPred_model(data_args.max_len_lc)
# encoder1 = LSTMEncoder(lstm_args)
encoder1 = CNNEncoder(cnn_args)
encoder2 = Astroconformer(astroconf_args)
model = DoubleInputRegressor(encoder1, encoder2, model_args)

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
# model = SimpleRegressor(model, regressor_args).to(local_rank)
# encoder_dim = model.simsiam.output_dim * 2
# regressor = torch.nn.Sequential(
#     torch.nn.Linear(encoder_dim, encoder_dim//2),
#     torch.nn.BatchNorm1d(encoder_dim//2),
#     model.activation,
#     torch.nn.Dropout(conformer_args.dropout_p),
#     torch.nn.Linear(encoder_dim//2, regressor_args.output_dim*regressor_args.num_quantiles)
# )
# model.regressor = regressor

model = model.to(local_rank)
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
# loss_fn = torch.nn.L1Loss()
sb_weight = optim_args.sb_weight
e_weight = optim_args.energy_weight
# loss_fn = SumLoss([CQR(quantiles=optim_args.quantiles, reduction='none'), StephanBoltzmanLoss(reduction='none')], [1-sb_weight, sb_weight])
ssl_loss_fn = SumLoss([torch.nn.MSELoss(), TotalEnergyLoss()], [1-e_weight, e_weight])  
optimizer = torch.optim.AdamW(model.parameters(), lr=float(optim_args.max_lr), weight_decay=float(optim_args.weight_decay))
scaler = GradScaler()
total_steps = int(data_args.num_epochs) * len(train_dataloader)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                    T_max=total_steps,
                                                    eta_min=float(optim_args.max_lr)/10)
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

# Save the complete configuration to a YAML file


if data_args.masked_transform:
    trainer = MaskedRegressorTrainer(model=model, optimizer=optimizer,
                        criterion=loss_fn, ssl_criterion=ssl_loss_fn, output_dim=len(data_args.labels), scaler=scaler, grad_clip=True,
                       scheduler=scheduler, train_dataloader=train_dataloader, num_quantiles=len(optim_args.quantiles),                       val_dataloader=val_dataloader, device=local_rank, w_name=None, w_init_val=1,ssl_weight=1e-5,
                           exp_num=datetime_dir, log_path=data_args.log_dir, range_update=None,
                           accumulation_step=1, max_iter=np.inf,
                        exp_name=f"{model_name}_lc_{exp_num}")  
else:
    trainer = RegressorTrainer(model=model, optimizer=optimizer,
                        criterion=loss_fn, output_dim=len(data_args.labels), scaler=scaler, grad_clip=True,
                       scheduler=scheduler, train_dataloader=train_dataloader, num_quantiles=len(optim_args.quantiles),
                            val_dataloader=val_dataloader, device=local_rank, w_name=None,
                           exp_num=datetime_dir, log_path=data_args.log_dir, range_update=None,
                           accumulation_step=1, max_iter=np.inf,
                        exp_name=f"{model_name}_lc_{exp_num}")

    # trainer = DoubleInputTrainer(model=model, optimizer=optimizer,
    #                     criterion=loss_fn, output_dim=len(data_args.labels), num_classes=1,
    #                    scheduler=scheduler, train_dataloader=train_dataloader, num_quantiles=len(optim_args.quantiles),
    #                         val_dataloader=val_dataloader, device=local_rank, 
    #                        exp_num=datetime_dir, log_path=data_args.log_dir,
    #                              accumulation_step=1, max_iter=100,
    #                     exp_name=f"{model_name}_lc_{exp_num}")
    #
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

fit_res = trainer.fit(num_epochs=data_args.num_epochs, device=local_rank,
                        early_stopping=10, best='loss', conf=True) 
output_filename = f'{data_args.log_dir}/{datetime_dir}/{model_name}_lc_{exp_num}.json'
with open(output_filename, "w") as f:
    json.dump(fit_res, f, indent=2)
fig, axes = plot_fit(fit_res, legend=exp_num, train_test_overlay=True)
plt.savefig(f"{data_args.log_dir}/{datetime_dir}/fit_{model_name}_lc_{exp_num}.png")
plt.clf()

preds_val, targets_val, info_val = trainer.predict(val_dataloader, device=local_rank, load_best=False)
preds, targets, info = trainer.predict(test_dataloader, device=local_rank, load_best=False)

low_q = preds[:, :, 0] 
high_q = preds[:, :, -1]
coverage = np.mean((targets >= low_q) & (targets <= high_q))
print('coverage: ', coverage)

cqr_errs = loss_fn.calibrate(preds_val, targets_val)
print(targets.shape, preds.shape)
preds_cqr = loss_fn.predict(preds, cqr_errs)

low_q = preds_cqr[:, :, 0]
high_q = preds_cqr[:, :, -1]
coverage = np.mean((targets >= low_q) & (targets <= high_q))
print('coverage after calibration: ', coverage)

df = save_predictions_to_dataframe(preds, targets, info, data_args.labels, optim_args.quantiles, id_name='KID')
df.to_csv(f"{data_args.log_dir}/{datetime_dir}/{model_name}_light_decode2_{exp_num}.csv", index=False)

print(df.keys())
print('predictions saved in', f"{data_args.log_dir}/{datetime_dir}/{model_name}_light_decode2_{exp_num}.csv") 
df_cqr = save_predictions_to_dataframe(preds_cqr, targets, info, data_args.labels, optim_args.quantiles, id_name='KID')
df_cqr.to_csv(f"{data_args.log_dir}/{datetime_dir}/{model_name}_light_decode2_{exp_num}_cqr.csv", index=False)
print(df_cqr.keys())
print('predictions saved in', f"{data_args.log_dir}/{datetime_dir}/{model_name}_light_decode2_{exp_num}_cqr.csv")
umap_df = create_umap(model.module.simsiam.encoder, test_dataloader, local_rank, use_w=False, dual=False)
print("umap created: ", umap_df.shape)
umap_df.to_csv(f"{data_args.log_dir}/{datetime_dir}/umap_{exp_num}_ssl.csv", index=False)
print(f"umap saved at {data_args.log_dir}/{datetime_dir}/umap_{exp_num}_ssl.csv")
exit(0)

