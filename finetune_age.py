import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Subset
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import OneCycleLR
from astropy.io import fits
from astropy.table import Table
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
from scipy import stats


warnings.filterwarnings("ignore")

import sys
from os import path
ROOT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print("running from ", ROOT_DIR) 

from transforms.transforms import *
from dataset.dataset import LightSpecDataset, FineTuneDataset
from dataset.sampler import DistinctParameterSampler
from nn.astroconf import Astroconformer, AstroEncoderDecoder
from nn.models import *
from nn.simsiam import MultiModalSimSiam, MultiModalSimCLR
from nn.moco import MultimodalMoCo, PredictiveMoco, MocoTuner
from nn.multi_modal import FineTuner, ContrastiveFineTuner, UniModalFineTuner
from nn.simsiam import SimSiam, projection_MLP
from nn.optim import CQR
from nn.utils import init_model, load_checkpoints_ddp
from util.utils import *
from nn.train import *
from tests.test_unique_sampler import run_sampler_tests
from features import multimodal_umap, create_umap
import generator
from lightspec import priority_merge_prot

META_COLUMNS = ['KID', 'Teff', 'logg', 'FeH', 'Rstar', 'Mstar', 'Lstar', 'Dist', 'kmag_abs', 'RUWE', 'Prot']


MODELS = {'Astroconformer': Astroconformer, 'CNNEncoder': CNNEncoder, 'MultiEncoder': MultiEncoder, 'MultiTaskRegressor': MultiTaskRegressor,
           'AstroEncoderDecoder': AstroEncoderDecoder, 'CNNEncoderDecoder': CNNEncoderDecoder,}

# finetune_checkpoint_path = '/data/lightSpec/logs/age_finetune_2025-04-30/age_finetune_lightspec_dual_former_6_latent_giants_finetune_age_gyro.pth'
finetune_checkpoint_path = '/data/lightSpec/logs/age_finetune_2025-05-17/age_finetune_lightspec_dual_former_6_latent_giants_nss_finetune_age_gyro.pth'
jepa_checkpoint_path= '/data/lightSpec/logs/age_finetune_2025-05-22/age_finetune_lightspec_compare_jepa.pth'
simsiam_checkpoint_path = '/data/lightSpec/logs/age_finetune_2025-05-28/age_finetune_lightspec_compare_simsiam.pth'
R_SUN_KM = 6.957e5

torch.cuda.empty_cache()

torch.manual_seed(1234)
np.random.seed(1234)

def get_kepler_meta_df():
    kepler_df = get_all_samples_df(num_qs=None, read_from_csv=True)
    kepler_meta = pd.read_csv('/data/lightPred/tables/berger_catalog_full.csv')
    kmag_df = pd.read_csv('/data/lightPred/tables/kepler_dr25_meta_data.csv')
    kepler_df = kepler_df.merge(kepler_meta, on='KID', how='right').merge(kmag_df[['KID', 'KMAG']], on='KID', how='left')
    kepler_df['kmag_abs'] = kepler_df['KMAG'] - 5 * np.log10(kepler_df['Dist']) + 5
    
    lightpred_df = pd.read_csv('/data/lightPred/tables/kepler_predictions_clean_seg_0_1_2_median.csv')
    lightpred_df['Prot_ref'] = 'lightpred'
    lightpred_df.rename(columns={'predicted period': 'Prot'}, inplace=True)
    lightpred_df['Prot_err'] = lightpred_df['observational error'] / lightpred_df['mean_period_confidence']

    # read other catalogs and create error columns
    santos_df = pd.read_csv('/data/lightPred/tables/santos_periods_19_21.csv').rename(columns={'E_Prot': 'Prot_err'})
    santos_df['Prot_ref'] = 'santos'
    santos_df['Prot_err'] = pd.to_numeric(santos_df['Prot_err'], errors='coerce')
    santos_df['Prot'] = pd.to_numeric(santos_df['Prot'], errors='coerce')
    valid = santos_df.dropna(subset=['Prot', 'Prot_err'])
    relative_mean_err = (valid['Prot_err'] / valid['Prot']).mean()
    santos_df['Prot_err'] = santos_df['Prot_err'].fillna(santos_df['Prot'] * relative_mean_err)
    
    mcq14_df = pd.read_csv('/data/lightPred/tables/Table_1_Periodic.txt')
    mcq14_df['Prot_ref'] = 'mcq14'
    reinhold_df = pd.read_csv('/data/lightPred/tables/reinhold2023.csv')
    reinhold_df['Prot_ref'] = 'reinhold'
    reinhold_df['Prot_err'] = reinhold_df['Prot'] * 0.15 # assume 15% error

    p_dfs = [santos_df, lightpred_df, mcq14_df, reinhold_df]

    kepler_df = priority_merge_prot(p_dfs, kepler_df)
    return kepler_df

def create_train_test_dfs(meta_columns):
    period_catalog = get_kepler_meta_df().drop_duplicates('KID')


    lamost_kepler_df = pd.read_csv('/data/lamost/lamost_dr8_gaia_dr3_kepler_ids.csv').rename(columns={'kepid':'KID'})
    lamost_kepler_df = lamost_kepler_df[~lamost_kepler_df['KID'].isna()]
    lamost_kepler_df['KID'] = lamost_kepler_df['KID'].astype(int)
    unamed_cols = [col for col in lamost_kepler_df.columns if 'Unnamed' in col]
    lamost_kepler_df.drop(columns=unamed_cols, inplace=True)
    lamost_kepler_df = lamost_kepler_df.merge(period_catalog, on='KID', how='left', suffixes=('_lamost', '')).drop_duplicates('KID')
    
    astero_age_df = pd.read_csv('/data/lightPred/tables/ages_dataset.csv')
    gyro_age_df = pd.read_csv('/data/lightPred/tables/ages_dataset_gyro.csv')
    unamed_cols = [col for col in gyro_age_df.columns if 'Unnamed' in col]

    age_df = gyro_age_df
    # age_df = pd.concat([gyro_age_df, astero_age_df, ]).drop_duplicates('KID')

    age_df.drop(columns=unamed_cols, inplace=True)
    
    final_df = lamost_kepler_df.merge(age_df, on='KID', suffixes=('', '_ages')).drop_duplicates('KID')
    
    print('mstar nans: ', final_df['Mstar'].isna().sum())

    print("final df: ", final_df.shape[0], ' age df: ', age_df.shape[0], ' lamost df: ', lamost_kepler_df.shape[0], ' period df: ', period_catalog.shape[0])

    final_df.dropna(subset=['final_age', 'age_error'], inplace=True)

    final_df['age_error_rel'] = final_df['age_error'] / final_df['final_age']
    final_df['final_age_norm'] = final_df['final_age'] / 11
    final_df['age_error_norm'] = final_df['age_error'] / 11

    

    # final_df = final_df[~((final_df['ESA3'] == 1) & (final_df['age_ref'] == 'asteroseismology'))]

    # final_df = final_df[final_df['age_ref']=='gyro_gyro']
    # final_df = final_df[final_df['age_ref']=='asteroseismology']

    print("number of samples in final df: ", final_df.shape[0])

    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    final_df.to_csv('/data/lightSpec/updated_finetune_age_df.csv', index=False)
    

    train_df, val_df = train_test_split(final_df, test_size=0.2, random_state=42)


    return train_df, val_df


current_date = datetime.date.today().strftime("%Y-%m-%d")
datetime_dir = f"age_finetune_{current_date}"

local_rank, world_size, gpus_per_node = setup()
args_dir = '/data/lightSpec/nn/full_config.yaml'
data_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Data'])
exp_num = data_args.exp_num + '_' + data_args.approach
light_model_name = data_args.light_model_name
spec_model_name = data_args.spec_model_name
combined_model_name = data_args.combined_model_name
if data_args.test_run:
    datetime_dir = f"test_{current_date}"

os.makedirs(f"{data_args.log_dir}/{datetime_dir}", exist_ok=True)


train_dataset, val_dataset, test_dataset, complete_config = generator.get_data(data_args,
                                                                                 create_train_test_dfs,
                                                                                dataset_name='FineTune')

print("number of training samples: ", len(train_dataset))

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
    light, spec, y, light2, spec2, info = train_dataset[i]
    print("train dataset: ", light.shape, light2.shape,  spec.shape, spec2.shape, y)

pre_trained_model, optim_args, tuner_args, complete_config, light_model, spec_model = generator.get_model(data_args, args_dir, complete_config, local_rank)

quantiles = [0.14, 0.5, 0.86]
tuner_args.out_dim = tuner_args.out_dim * len(quantiles)

only_lc = False
only_spec = False
if data_args.approach == 'unimodal_light':
    only_lc = True
    only_spec = False
    model = UniModalFineTuner(light_model.simsiam.encoder, tuner_args.get_dict(), head_type='transformer', use_sigma=False).to(local_rank)
elif data_args.approach == 'unimodal_spec':
    only_spec = True
    only_lc = False
    model = UniModalFineTuner(spec_model.encoder, tuner_args.get_dict(), head_type='transformer', use_sigma=False).to(local_rank)
elif data_args.approach == 'moco' or data_args.approach == 'simsiam' or data_args.approach == 'jepa':
    model = ContrastiveFineTuner(pre_trained_model, tuner_args.get_dict(), head_type='transformer', use_sigma=False).to(local_rank)
elif data_args.approach == 'dual_former':
    model = FineTuner(pre_trained_model, tuner_args.get_dict(), head_type='transformer', use_sigma=False).to(local_rank)

# for param in pre_trained_model.parameters():
#         param.requires_grad = False

print("loading finetune checkpoints from simsiam best run...")
model = load_checkpoints_ddp(model, simsiam_checkpoint_path)


model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())

print("number of all parameters in finetune setting: ", all_params)

print("number of trainable parameters in finetune setting: ", num_params)

loss_fn = CQR(quantiles=quantiles, reduction='none')
# loss_fn = torch.nn.L1Loss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=float(optim_args.max_lr), weight_decay=float(optim_args.weight_decay))


trainer = RegressorTrainer(
    model=model,
    optimizer=optimizer,
    criterion=loss_fn,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    device=local_rank,
    output_dim=len(data_args.prediction_labels_finetune),
    num_quantiles=len(quantiles),
    use_w = False,
    latent_vars=data_args.prediction_labels_lightspec,
    loss_weight_name=None,
    max_iter=np.inf,
    only_lc=only_lc,
    only_spec=only_spec,
    # error_name='cos_inc_err',
    log_path=data_args.log_dir,
    exp_num=datetime_dir,
    exp_name=f"age_finetune_{exp_num}",
)

# trainer = ContrastiveTrainer(model=model, optimizer=optimizer, stack_pairs=False,
#                             criterion=loss_fn, output_dim=1, scaler=None,
#                         scheduler=None, train_dataloader=train_dataloader,
#                         val_dataloader=val_dataloader, device=local_rank, ssl_weight=0,
#                                 weight_decay=True, num_quantiles=len(optim_args.quantiles),
#                             exp_num=datetime_dir, log_path=data_args.log_dir, range_update=None,
#                             accumulation_step=1, max_iter=np.inf,
#                             exp_name=f"test_prot_lc_{exp_num}")


complete_config.update(
    {"trainer": trainer.__dict__,
    "loss_fn": str(loss_fn),
    "optimizer": str(optimizer)}
)
   
config_save_path = f"{data_args.log_dir}/{datetime_dir}/finetune_age_{exp_num}_complete_config.yaml"
with open(config_save_path, "w") as config_file:
    json.dump(complete_config, config_file, indent=2, default=str)

print(f"Configuration (with model structure) saved at {config_save_path}.")

# fit_res = trainer.fit(num_epochs=data_args.num_epochs, device=local_rank,
#                         early_stopping=10, best='loss', conf=True) 
# output_filename = f'{data_args.log_dir}/{datetime_dir}/finetune_age_{exp_num}.json'
# with open(output_filename, "w") as f:
#     json.dump(fit_res, f, indent=2)
# fig, axes = plot_fit(fit_res, legend=exp_num, train_test_overlay=True)
# plt.savefig(f"{data_args.log_dir}/{datetime_dir}/fit_finetune_age_{exp_num}.png")
# plt.clf()

preds, targets, sigmas, projections, aggregated_info = trainer.predict(test_dataloader, device=local_rank)

preds_val, targets_val, sigmas_val, projections, aggregated_info_val = trainer.predict(val_dataloader, device=local_rank)

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
acc = np.mean(np.abs(targets - preds_cqr[:, :, 1]) < targets * 0.1, axis=0)
print('accuracy: ', acc)
print('mean error: ', np.mean(np.abs(targets - preds_cqr[:, :, 1])))
print('coverage after calibration: ', coverage)

print('results shapes: ', preds.shape, targets.shape, sigmas.shape)
results_df = pd.DataFrame({'sigmas': sigmas})
results_df['KID'] = aggregated_info['KID']
results_df['age_ref'] = aggregated_info['age_ref']
# results_df['ESA3'] = aggregated_info['ESA3']
# results_df['flag_gyro_quality'] = aggregated_info['flag_gyro_quality']
labels = data_args.prediction_labels_finetune
for i, label in enumerate(labels):
    results_df[label] = targets[:, i]
    for q in range(len(quantiles)):
        q_value = quantiles[q]
        y_pred_q = preds_cqr[:,:, q]
        results_df[f'pred_{label}_q{q_value:.3f}'] = y_pred_q[:, i]
print(results_df.head())
results_df.to_csv(f"{data_args.log_dir}/{datetime_dir}/predictions_finetune_age_{exp_num}.csv", index=False)
print(f"Results saved at {data_args.log_dir}/{datetime_dir}/predictions_finetune_age_{exp_num}.csv")
# Access results



