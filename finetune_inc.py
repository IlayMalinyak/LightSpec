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
from nn.multi_modal import FineTuner
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

finetune_checkpoint_path = '/data/lightSpec/logs/inc_finetune_2025-05-19/inc_finetune_lightspec_dual_former_6_latent_giants_nss_finetune_inc.pth'


R_SUN_KM = 6.957e5

torch.cuda.empty_cache()

torch.manual_seed(1234)
np.random.seed(1234)

def get_kepler_meta_df():
    kepler_df = get_all_samples_df(num_qs=None, read_from_csv=True)
    kepler_meta = pd.read_csv('/data/lightPred/tables/berger_catalog_full.csv')
    kmag_df = pd.read_csv('/data/lightPred/tables/kepler_dr25_meta_data.csv')
    kepler_df = kepler_df.merge(kepler_meta, on='KID', how='left').merge(kmag_df[['KID', 'KMAG']], on='KID', how='left')
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

    p_dfs = [lightpred_df, santos_df, mcq14_df, reinhold_df]
    kepler_df = priority_merge_prot(p_dfs, kepler_df)
    return kepler_df

def get_cos_i_err(row):
    dv = row['Prot'] * 24 * 3600 / (2 * np.pi * row['Rstar'] * R_SUN_KM) * row['vsini_err']
    dp = row['vsini'] / (2 * np.pi * row['Rstar'] * R_SUN_KM) * row['Prot_err'] * 24 * 3600
    dr = (row['vsini'] * row['Prot'] * 24 * 3600 * row['Rstar_err'] * R_SUN_KM
          / (2 * np.pi * (row['Rstar'] * R_SUN_KM) ** 2))
    err = np.sqrt(dv ** 2 + dp ** 2 + dr ** 2)
    return err

def show_inclination_behavior(final_df):
    p_refs = final_df['Prot_ref'].unique()
    for ref in p_refs:
        ref_df = final_df[final_df['Prot_ref'] == ref]
        nan_inc_ratio = len(ref_df[ref_df['cos_inc'] > 1]) / len(ref_df)
        nan_inc_ratio_err = len(ref_df[ref_df['cos_inc'] - ref_df['cos_inc_err'] > 1]) / len(ref_df)
        plt.scatter(ref_df['Prot'], ref_df['cos_inc'], label=f'{ref} %$cos(i) > 1$={nan_inc_ratio:.2f}({nan_inc_ratio_err:.2f})')
        # plt.errorbar(ref_df['Prot'], ref_df['cos_inc'],yerr=ref_df['cos_inc_err'], fmt='none', color='gray',
        #              elinewidth=1, alpha=0.1, capsize=3)
    plt.legend()
    plt.hlines(1, 0, 70, colors='black', linestyles='dashed')
    plt.ylabel('cos(i)')
    plt.xlabel('Prot (days)')
    plt.semilogy()
    plt.savefig('/data/lightSpec/images/cos_inc_err_pref.png')
    plt.close()

    plt.scatter(final_df['Prot'], final_df['cos_inc'], c=np.log(final_df['cos_inc_err']))
    plt.hlines(1, 0, 70, colors='black', linestyles='dashed')
    plt.ylabel('cos(i)')
    plt.xlabel('Prot (days)')
    plt.colorbar()
    plt.semilogy()
    plt.savefig('/data/lightSpec/images/cos_inc_err.png')
    plt.close()

def physical_cos_inc(cos_inc_value, cos_inc_err):
    """
    Return a physically plausible cos(inclination) value by sampling from
    a truncated normal distribution bounded between 0 and 1.
    """
    # For values already in the physical range with small errors, keep as is
    if 0 <= cos_inc_value <= 1:
        return cos_inc_value

    # For non-physical values, sample from truncated normal
    a, b = 0, 1  # bounds of the truncated distribution

    # Calculate parameters for the truncated normal
    loc = np.clip(cos_inc_value, 0.01, 0.99)  # Move extreme values slightly inward
    scale = cos_inc_err * 2
    # print("cos_inc_value: ", cos_inc_value, "cos_inc_err: ", cos_inc_err, "loc: ", loc, "scale: ", scale)

    # Create a truncated normal distribution
    trunc_norm = stats.truncnorm(
        (a - loc) / scale,
        (b - loc) / scale,
        loc=loc,
        scale=scale
    )

    # Return a random sample from this distribution
    return trunc_norm.rvs()

def create_train_test_dfs(meta_columns):
    inc_dataset = pd.read_csv('/data/lightPred/tables/inc_dataset.csv')
    inc_dataset['i_rad'] = np.deg2rad(inc_dataset['i'])
    inc_dataset['i_err_rad'] = np.deg2rad(inc_dataset['i_err'])
    inc_dataset['i_rad_norm'] = inc_dataset['i_rad'] / (np.pi / 2)
    inc_dataset['i_err_rad_norm'] = inc_dataset['i_err_rad'] / (np.pi / 2)
    inc_dataset['cos_inc'] = np.cos(inc_dataset['i_rad'])
    inc_dataset['cos_inc_err'] = np.cos(inc_dataset['i_err_rad'])

    kepler_meta = get_kepler_meta_df()
    inc_dataset = inc_dataset.merge(kepler_meta, on='KID', how='left')

    kepler_apogee = pd.read_csv('/data/apogee/crossmatched_catalog_Kepler.csv')
    kepler_apogee = kepler_apogee[~kepler_apogee['APOGEE_VSINI'].isna()].rename(columns={'APOGEE_VSINI': 'vsini'})
    kepler_apogee['vsini_ref'] = 'apogee'
    lamost_kepler_df = pd.read_csv('/data/lamost/lamost_dr8_gaia_dr3_kepler_ids.csv').rename(columns={'kepid':'KID'})
    print("lamost kepler df: ", len(lamost_kepler_df))
    lamost_kepler_df = lamost_kepler_df[~lamost_kepler_df['KID'].isna()]
    lamost_kepler_df['KID'] = lamost_kepler_df['KID'].astype(int)
    lamost_kepler_apogee = lamost_kepler_df.merge(kepler_apogee[['KID', 'vsini', 'vsini_ref']], on='KID')
    final_apogee = lamost_kepler_apogee.merge(kepler_meta, on='KID',
                                                         suffixes=['', '_kep']).drop_duplicates('ObsID')
    final_apogee['vsini_err'] = final_apogee['vsini'] * 0.1

    cks_catalog = Table.read('/data/lightPred/tables/CKS2017.fit', format='fits').to_pandas()
    cks_catalog['vsini_ref'] = 'cks'
    final_cks = cks_catalog.merge(kepler_meta, left_on='KIC', right_on='KID', suffixes=['', '_kep'])
    final_cks = final_cks.merge(lamost_kepler_df, left_on='KIC', right_on='KID', suffixes=['', '_lamost'])
    final_cks = final_cks.drop_duplicates('KIC')
    final_cks['vsini_err'] = final_cks['vsini'] * 0.1 # assume 10% error

    vsini_df = pd.concat([final_apogee, final_cks])
    print(" final apogee: ", len(final_apogee), "final cks", len(final_cks))
    vsini_df['Rstar_err'] = (vsini_df['E_Rstar'] - vsini_df['e_Rstar']) / 2

    vsini_df.dropna(subset=['Prot', 'vsini', 'Rstar'], inplace=True)

    vsini_df['cos_inc'] = (vsini_df['vsini'] * vsini_df['Prot'] * 24 * 3600
                           / (2 * np.pi * vsini_df['Rstar'] * R_SUN_KM))
    
    vsini_df['cos_inc_err'] = vsini_df.apply(get_cos_i_err, axis=1)
    vsini_df = vsini_df[vsini_df['cos_inc'].abs() < 1]
    vsini_df['i_rad'] = np.arccos(vsini_df['cos_inc'])
    vsini_df['i_rad_err'] = np.arccos(vsini_df['cos_inc_err'])
    vsini_df['i_rad_norm'] = vsini_df['i_rad'] / (np.pi / 2)
    vsini_df['i_err_rad_norm'] = vsini_df['i_rad_err'] / (np.pi / 2)

    inc_dataset = inc_dataset.merge(lamost_kepler_df, on='KID', how='left').dropna(subset=['ObsID'])
    final_df  = pd.concat([inc_dataset, vsini_df]).drop_duplicates('KID')
    print("len inc dataset: ", len(inc_dataset))
    print("len vsini dataset: ", len(vsini_df))
    print("len final df: ", len(final_df))

    # final_df['ObsID'].fillna(0, inplace=True)
    
    train_df, val_df = train_test_split(final_df, test_size=0.2, random_state=42)

    return train_df, val_df


    

def create_train_test_dfs_vsini(meta_columns):
    period_catalog = get_kepler_meta_df().dropna(subset=['Prot']).drop_duplicates('KID')
    print("nans in period catalog: ", period_catalog['Prot'].isna().sum())

    kepler_apogee = pd.read_csv('/data/apogee/crossmatched_catalog_Kepler.csv')
    kepler_apogee = kepler_apogee[~kepler_apogee['APOGEE_VSINI'].isna()].rename(columns={'APOGEE_VSINI': 'vsini'})
    kepler_apogee['vsini_ref'] = 'apogee'
    lamost_kepler_df = pd.read_csv('/data/lamost/lamost_dr8_gaia_dr3_kepler_ids.csv').rename(columns={'kepid':'KID'})
    print("lamost kepler df: ", len(lamost_kepler_df))
    lamost_kepler_df = lamost_kepler_df[~lamost_kepler_df['KID'].isna()]
    lamost_kepler_df['KID'] = lamost_kepler_df['KID'].astype(int)
    lamost_kepler_apogee = lamost_kepler_df.merge(kepler_apogee[['KID', 'vsini', 'vsini_ref']], on='KID')
    final_apogee = lamost_kepler_apogee.merge(period_catalog, on='KID',
                                                         suffixes=['', '_kep']).drop_duplicates('ObsID')
    final_apogee['vsini_err'] = final_apogee['vsini'] * 0.1

    cks_catalog = Table.read('/data/lightPred/tables/CKS2017.fit', format='fits').to_pandas()
    cks_catalog['vsini_ref'] = 'cks'
    print("len cks catalog: ", len(cks_catalog))
    final_cks = cks_catalog.merge(period_catalog, left_on='KIC', right_on='KID', suffixes=['', '_kep'])
    print("after merge, len final_cks: ", len(final_cks), len(period_catalog))
    final_cks = final_cks.merge(lamost_kepler_df, left_on='KIC', right_on='KID', suffixes=['', '_lamost'])
    print("after merge 2, len final_cks: ", len(final_cks))
    final_cks = final_cks.drop_duplicates('KIC')
    final_cks['vsini_err'] = final_cks['vsini'] * 0.1 # assume 10% error

    final_df = pd.concat([final_apogee, final_cks])
    print(" final apogee: ", len(final_apogee), "final cks", len(final_cks),
          " final df: ", len(final_df))
    final_df['Rstar_err'] = (final_df['E_Rstar'] - final_df['e_Rstar']) / 2

    kois = pd.read_csv('/data/lightPred/tables/kois.csv')
    kois = kois[kois['koi_disposition'] == 'CONFIRMED']
    kois = kois[['KID', 'kepoi_name', 'kepler_name', 'koi_disposition', 'planet_Prot']]
    final_df = final_df.merge(kois, on='KID', how='left')

    print("kois general : ", len(kois), " kois in final df: ", len(final_df[~final_df['kepoi_name'].isna()]))

    final_df.dropna(subset=['Prot', 'vsini', 'Rstar'], inplace=True)

    final_df['cos_inc'] = (final_df['vsini'] * final_df['Prot'] * 24 * 3600
                           / (2 * np.pi * final_df['Rstar'] * R_SUN_KM))
    final_df['cos_inc_err'] = final_df.apply(get_cos_i_err, axis=1)

    final_df = final_df[final_df['cos_inc'].abs() < 1]
    print("final df after removing nan cos_inc: ", len(final_df))
    final_df['physical_cos_inc'] = final_df.apply(
        lambda row: physical_cos_inc(row['cos_inc'], row['cos_inc_err']),
        axis=1
    )
    final_df['trunc_cos_inc'] = final_df['cos_inc'].apply(lambda x: x if x < 1 else np.nan)
    plt.hist(final_df['physical_cos_inc'], bins=30, density=True, histtype='step', label='physical')
    plt.hist(final_df['trunc_cos_inc'], bins=30,density=True, histtype='step', label='cos')
    plt.close()
    final_df['log_cos_inc'] = -np.log(final_df['physical_cos_inc'] + 1e-3)
    final_df['inc'] = np.arccos(
        final_df['physical_cos_inc']) * 180 / np.pi  # here im using arccos becasue the real angle is 90 - inc
    final_df['norm_vsini'] = (final_df['vsini'] - final_df['vsini'].min()) / (
                final_df['vsini'].max() - final_df['vsini'].min())
    final_df['norm_Prot'] = (final_df['Prot'] - final_df['Prot'].min()) / (
                final_df['Prot'].max() - final_df['Prot'].min())
    show_inclination_behavior(final_df)

    final_df = final_df[~final_df['inc'].isna()]
    print("final df after removing nan inc: ", len(final_df))
    # final_df['Prot'] /= 70

    train_df, val_df = train_test_split(final_df, test_size=0.2, random_state=42)

    # for ref in final_df['vsini_ref'].unique():
    #     print("ref: ", ref, " samples: ", len(final_df[final_df['vsini_ref'] == ref]))
    #     ref_df = train_df[train_df['vsini_ref'] == ref]
    #     if len(ref_df) < 20:
    #         continue
    #     plt.hist(ref_df['physical_cos_inc'], bins=40, histtype='step', density=True, label=ref)
    kois_df = final_df[final_df['koi_disposition'] == 'CONFIRMED']
    print("kois df: ", len(kois_df))
    plt.hist(kois_df['physical_cos_inc'], bins=40, histtype='step', density=True, label='KOI physical')
    plt.hist(final_df['physical_cos_inc'], bins=40, histtype='step', density=True, label='All physical')
    plt.hist(final_df['trunc_cos_inc'], bins=40, histtype='step', density=True, label='All raw')
    plt.legend()
    plt.savefig(f"/data/lightSpec/images/cos_inc_hist.png")
    plt.close()


    return train_df, val_df


current_date = datetime.date.today().strftime("%Y-%m-%d")
datetime_dir = f"inc_finetune_{current_date}"

local_rank, world_size, gpus_per_node = setup()
args_dir = '/data/lightSpec/nn/full_config.yaml'
data_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Data'])
exp_num = data_args.exp_num
light_model_name = data_args.light_model_name
spec_model_name = data_args.spec_model_name
combined_model_name = data_args.combined_model_name
if data_args.test_run:
    datetime_dir = f"test_{current_date}"

os.makedirs(f"{data_args.log_dir}/{datetime_dir}", exist_ok=True)


train_dataset, val_dataset, test_dataset, complete_config = generator.get_data(data_args,
                                                                                 create_train_test_dfs,
                                                                                dataset_name='FineTune')

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

# for param in pre_trained_model.parameters():
#         param.requires_grad = False

quantiles = [0.14, 0.5, 0.86]
tuner_args.out_dim = tuner_args.out_dim * len(quantiles)
model = FineTuner(pre_trained_model, tuner_args.get_dict(), head_type='mlp').to(local_rank)

# print("loading model from ", finetune_checkpoint_path)
# model = load_checkpoints_ddp(model, finetune_checkpoint_path)

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
    only_lc=False,
    latent_vars=data_args.prediction_labels_lightspec,
    loss_weight_name=None,
    max_iter=np.inf,
    # error_name='cos_inc_err',
    log_path=data_args.log_dir,
    exp_num=datetime_dir,
    exp_name=f"inc_finetune_{exp_num}",
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
   
config_save_path = f"{data_args.log_dir}/{datetime_dir}/finetune_inc_{exp_num}_complete_config.yaml"
with open(config_save_path, "w") as config_file:
    json.dump(complete_config, config_file, indent=2, default=str)

print(f"Configuration (with model structure) saved at {config_save_path}.")

fit_res = trainer.fit(num_epochs=data_args.num_epochs, device=local_rank,
                        early_stopping=10, best='loss', conf=True) 
output_filename = f'{data_args.log_dir}/{datetime_dir}/finetune_inc_{exp_num}.json'
with open(output_filename, "w") as f:
    json.dump(fit_res, f, indent=2)
fig, axes = plot_fit(fit_res, legend=exp_num, train_test_overlay=True)
plt.savefig(f"{data_args.log_dir}/{datetime_dir}/fit_finetune_inc_{exp_num}.png")
plt.clf()

preds, targets, sigmas, projections, aggregated_info = trainer.predict(test_dataloader, device=local_rank)
preds_train, targets_train, sigmas_train, projections_train, aggregated_info_train = trainer.predict(train_dataloader, device=local_rank)
preds_val, targets_val, sigmas_val, projections_val, aggregated_info_val = trainer.predict(val_dataloader, device=local_rank)

np.save(f"{data_args.log_dir}/{datetime_dir}/projections_test_{exp_num}.npy", projections)
np.save(f"{data_args.log_dir}/{datetime_dir}/projections_train_{exp_num}.npy", projections_train)
np.save(f"{data_args.log_dir}/{datetime_dir}/projections_val_{exp_num}.npy", projections_val)

kids_test = aggregated_info['KID']
kids_train = aggregated_info_train['KID']
kids_val = aggregated_info_val['KID']

df_test = pd.DataFrame({'sigmas': sigmas, 'KID': kids_test}).to_csv(f"{data_args.log_dir}/{datetime_dir}/test_kids.csv", index=False)
df_train = pd.DataFrame({'sigmas': sigmas_train, 'KID': kids_train}).to_csv(f"{data_args.log_dir}/{datetime_dir}/train_kids.csv", index=False)
df_val = pd.DataFrame({'sigmas': sigmas_val, 'KID': kids_val}).to_csv(f"{data_args.log_dir}/{datetime_dir}/val_kids.csv", index=False)

print('predictions shapes: ', preds.shape, targets.shape, sigmas.shape, projections.shape)

print('results shapes: ', preds.shape, targets.shape, sigmas.shape)
results_df = pd.DataFrame({'sigmas': sigmas})
results_df['KID'] = aggregated_info['KID']
labels = data_args.prediction_labels_finetune
for i, label in enumerate(labels):
    results_df[label] = targets[:, i]
    for q in range(len(quantiles)):
        y_pred_q = preds[:,:, q]
        results_df[f'pred_{label}_q{q_value:.3f}'] = y_pred_q[:, i]
print(results_df.head())
results_df.to_csv(f"{data_args.log_dir}/{datetime_dir}/test_predictions.csv", index=False)
# Access results



