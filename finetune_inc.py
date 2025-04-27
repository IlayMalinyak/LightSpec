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
from nn.simsiam import SimSiam, projection_MLP
from nn.optim import CQR
from nn.utils import init_model, load_checkpoints_ddp
from util.utils import *
from nn.train import *
from features import multimodal_umap, create_umap
import generator
from nn.multi_modal import FineTuner
from lightspec import priority_merge_prot

META_COLUMNS = ['KID', 'Teff', 'logg', 'FeH', 'Rstar', 'Mstar', 'Lstar', 'Dist', 'kmag_abs', 'RUWE', 'Prot']


MODELS = {'Astroconformer': Astroconformer, 'CNNEncoder': CNNEncoder, 'MultiEncoder': MultiEncoder, 'MultiTaskRegressor': MultiTaskRegressor,
           'AstroEncoderDecoder': AstroEncoderDecoder, 'CNNEncoderDecoder': CNNEncoderDecoder,}

R_SUN_KM = 6.957e5

torch.cuda.empty_cache()

torch.manual_seed(1234)
np.random.seed(1234)

def get_kepler_meta_df():
    kepler_df = get_all_samples_df(num_qs=None, read_from_csv=True)
    kepler_meta = pd.read_csv('../../kepler/data/lightPred/tables/berger_catalog_full.csv')
    kmag_df = pd.read_csv('../../kepler/data/lightPred/tables/kepler_dr25_meta_data.csv')
    kepler_df = kepler_df.merge(kepler_meta, on='KID', how='left').merge(kmag_df[['KID', 'KMAG']], on='KID', how='left')
    kepler_df['kmag_abs'] = kepler_df['KMAG'] - 5 * np.log10(kepler_df['Dist']) + 5
    lightpred_df = pd.read_csv('../../kepler/data/lightPred/tables/kepler_predictions_clean_seg_0_1_2_median.csv')
    lightpred_df['Prot_ref'] = 'lightpred'
    lightpred_df.rename(columns={'predicted period': 'Prot'}, inplace=True)

    santos_df = pd.read_csv('../../kepler/data/lightPred/tables/santos_periods_19_21.csv')
    santos_df['Prot_ref'] = 'santos'
    mcq14_df = pd.read_csv('../../kepler/data/lightPred/tables/Table_1_Periodic.txt')
    mcq14_df['Prot_ref'] = 'mcq14'
    reinhold_df = pd.read_csv('../../kepler/data/lightPred/tables/reinhold2023.csv')
    reinhold_df['Prot_ref'] = 'reinhold'

    p_dfs = [lightpred_df, santos_df, mcq14_df, reinhold_df]
    kepler_df = priority_merge_prot(p_dfs, kepler_df)
    return kepler_df

def create_train_test_dfs(meta_columns):
    period_catalog = get_kepler_meta_df()
    # period_catalog = pd.read_csv('/data/lightPred/tables/kepler_predictions_clean_seg_0_1_2_median.csv')
    # period_catalog.rename(columns={'predicted period': 'Prot'}, inplace=True)
    santos_catalog = pd.read_csv('../../kepler/data/lightPred/tables/santos_periods_19_21.csv')
    frasca_catalog = Table.read('../../kepler/data/lightPred/tables/frasca2016.fit', format='fits').to_pandas()
    cks_catalog = Table.read('../../kepler/data/lightPred/tables/CKS2017.fit', format='fits').to_pandas()
    kepler_meta_df = get_kepler_meta_df()

    kepler_apogee = pd.read_csv('../../kepler/data/apogee/crossmatched_catalog_Kepler.csv')
    kepler_apogee = kepler_apogee[~kepler_apogee['APOGEE_VSINI'].isna()].rename(columns={'APOGEE_VSINI': 'vsini'})
    kepler_apogee['vsini_ref'] = 'apogee'
    lamost_kepler_df = (pd.read_csv('../../kepler/data/lightPred/tables/lamost_dr8_gaia_dr3_kepler_ids.csv')
                        .rename(columns={'kepid': 'KID', 'obsid': 'ObsID'}))
    lamost_kepler_df = lamost_kepler_df[~lamost_kepler_df['KID'].isna()]
    lamost_kepler_df['KID'] = lamost_kepler_df['KID'].astype(int)
    lamost_kepler_apogee = lamost_kepler_df.merge(kepler_apogee[['KID', 'vsini', 'vsini_ref']], on='KID')
    lamost_kepler_periods_1 = lamost_kepler_apogee.merge(period_catalog, on='KID', suffixes=['', '_kep']).drop_duplicates('ObsID')
    lamost_kepler_periods_2 = lamost_kepler_apogee.merge(santos_catalog, on='KID', suffixes=['', '_kep']).drop_duplicates('ObsID')
    final_apogee = pd.concat([lamost_kepler_periods_1, lamost_kepler_periods_2]).drop_duplicates('ObsID')
    
    frasca_catalog = frasca_catalog[frasca_catalog['vsini'] > 120]
    frasca_catalog['vsini_ref'] = 'frasca2016'
    frasca_kepler_1 = frasca_catalog.merge(period_catalog, left_on='KIC', right_on='KID', suffixes=['', '_kep'])
    frasca_kepler_2 = frasca_catalog.merge(santos_catalog, left_on='KIC', right_on='KID', suffixes=['', '_kep'])
    print("frasca kepler 1: ", len(frasca_kepler_1), " frasca kepler 2: ", len(frasca_kepler_2))
    print("frasca catalog: ", len(frasca_catalog), " kepler apogee: ", len(kepler_apogee), " lamost kepler: ", len(lamost_kepler_df))
    final_frasca = pd.concat([frasca_kepler_1, frasca_kepler_2]).drop_duplicates('KIC')
    final_frasca = final_frasca.merge(lamost_kepler_df, left_on='KIC', right_on='KID', suffixes=['', '_lamost'])
    
    cks_catalog['vsini_ref'] = 'cks'
    cks_kepler_1 = cks_catalog.merge(period_catalog, left_on='KIC', right_on='KID', suffixes=['', '_kep'])
    cks_kepler_2 = cks_catalog.merge(santos_catalog, left_on='KIC', right_on='KID', suffixes=['', '_kep'])
    final_cks = pd.concat([cks_kepler_1, cks_kepler_2]).drop_duplicates('KIC')
    final_cks = final_cks.merge(lamost_kepler_df, left_on='KIC', right_on='KID', suffixes=['', '_lamost'])

    final_df = pd.concat([final_apogee, final_frasca, final_cks])
    print("final frasca: ", len(final_frasca), " final apogee: ", len(final_apogee), "final cks", len(final_cks), " final df: ", len(final_df))
    # berger_catalog = pd.read_csv('/data/lightPred/tables/berger_catalog.csv')
    # kmag_df = pd.read_csv('/data/lightPred/tables/kepler_dr25_meta_data.csv')
    # final_df = final_df.merge(kmag_df[['KID', 'KMAG']], on='KID', how='left')
    # final_df['kmag_abs'] = final_df['KMAG'] - 5 * np.log10(final_df['Dist']) + 5
    # final_df = final_df.merge(kepler_meta_df[META_COLUMNS], on='KID', how='left').rename(columns={'Prot_x': 'Prot'})

    kois = pd.read_csv('../../kepler/data/lightPred/tables/kois.csv')
    kois =  kois[kois['koi_disposition'] == 'CONFIRMED']
    kois = kois[['KID','kepoi_name','kepler_name','koi_disposition', 'planet_Prot']]
    final_df = final_df.merge(kois, on='KID', how='left')

    print("kois general : ", len(kois), " kois in final df: ", len(final_df[~final_df['kepoi_name'].isna()]))

    final_df['cos_inc'] = (final_df['vsini'] * final_df['Prot'] * 24 * 3600
                                / (2 * np.pi * final_df['Rstar'] * R_SUN_KM) )
    final_df['log_cos_inc'] = -np.log(final_df['cos_inc'] + 1e-3)
    final_df['inc'] = np.arccos(final_df['cos_inc']) * 180 / np.pi # here im using arccos becasue the real angle is 90 - inc
    final_df['norm_vsini'] = (final_df['vsini'] - final_df['vsini'].min()) / (final_df['vsini'].max() - final_df['vsini'].min())
    final_df['norm_Prot'] = (final_df['Prot'] - final_df['Prot'].min()) / (final_df['Prot'].max() - final_df['Prot'].min())
    final_df = final_df[~final_df['inc'].isna()]
    # final_df['Prot'] /= 70
    print("prot nans: ", final_df['Prot'].isna().sum(), " Rstar nans: ", final_df['Rstar'].isna().sum(), "VSINI nans: ", final_df['vsini'].isna().sum(), "inc nans: ", final_df['inc'].isna().sum())
    
    train_df, val_df  = train_test_split(final_df, test_size=0.2, random_state=42)
    print("samples kepler with period: ", len(period_catalog), " kepler_apogee", len(kepler_apogee),
     " lamost-kepler-apogee:", len(lamost_kepler_df), " lamost_kepler_apogee_periods_1", len(lamost_kepler_periods_1),
     " lamost_kepler_apogee_periods_2", len(lamost_kepler_periods_2), " final_df", len(final_df)
    )
    
    for ref in final_df['vsini_ref'].unique():
        print("ref: ", ref, " samples: ", len(final_df[final_df['vsini_ref'] == ref]))
        ref_df = train_df[train_df['vsini_ref'] == ref]
        if len(ref_df) < 20:
            continue
        plt.hist(ref_df['cos_inc'], bins=40, histtype='step', density=True, label=ref)
    kois_df = final_df[final_df['koi_disposition'] == 'CONFIRMED']
    print("kois df: ", len(kois_df))
    plt.hist(kois_df['cos_inc'], bins=40, histtype='step', density=True, label='KOI')
    plt.legend()
    plt.savefig(f"images/cos_inc_hist.png")
    plt.close()

    # final_df.to_csv('../../kepler/data/lightPred/tables/finetune_inc.csv', index=False)


    return train_df, val_df 


current_date = datetime.date.today().strftime("%Y-%m-%d")
datetime_dir = f"inc_finetune_{current_date}"

local_rank, world_size, gpus_per_node = setup()
args_dir = 'nn/full_config.yaml'
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
for i in range(10):
    light, spec, y, light2, spec2, info = train_dataset[i]
    print("train dataset: ", light.shape, light2.shape,  spec.shape, spec2.shape, y)

train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
train_dataloader = DataLoader(train_dataset,
                              batch_size=int(data_args.batch_size), \
                              collate_fn=kepler_collate_fn,
                              sampler=train_sampler
                              )


val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank)
val_dataloader = DataLoader(val_dataset,
                            batch_size=int(data_args.batch_size),
                            collate_fn=kepler_collate_fn,
                            sampler=val_sampler,
                            )

test_dataloader = DataLoader(test_dataset,
                            batch_size=int(data_args.batch_size),
                            collate_fn=kepler_collate_fn
                             )

for i in range(10):
    light, spec, y, light2, spec2, info = train_dataset[i]
    print("train dataset: ", light.shape, light2.shape,  spec.shape, spec2.shape, y)

pre_trained_model, optim_args, tuner_args, complete_config, light_model, spec_model = generator.get_model(data_args, args_dir, complete_config, local_rank)

for param in pre_trained_model.parameters():
        param.requires_grad = False

quantiles = [0.5]
tuner_args.out_dim = tuner_args.out_dim * len(quantiles)
model = FineTuner(pre_trained_model, tuner_args.get_dict()).to(local_rank)
# model = load_checkpoints_ddp(model, finetune_checkpoint_path)
# model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())

print("number of all parameters in finetune setting: ", all_params)

print("number of trainable parameters in finetune setting: ", num_params)

# loss_fn = CQR(quantiles=quantiles, reduction='none')
loss_fn = torch.nn.L1Loss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=float(optim_args.max_lr),
                             weight_decay=float(optim_args.weight_decay))

trainer = RegressorTrainer(
    model=model,
    optimizer=optimizer,
    criterion=loss_fn,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    device=local_rank,
    output_dim=len(data_args.prediction_labels_finetune),
    num_quantiles=len(quantiles),
    use_w=False,
    only_lc=False,
    latent_vars=data_args.prediction_labels_lightspec,
    loss_weight_name=None,
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

preds, targets, sigmas, aggregated_info = trainer.predict(test_dataloader, device=local_rank)

print('results shapes: ', preds.shape, targets.shape, sigmas.shape)
results_df = pd.DataFrame({'target_cos_inc': targets, 'sigmas_cos_inc': sigmas})
results_df['KID'] = aggregated_info['KID']
for q in range(len(quantiles)):
    y_pred_q = preds[:, :, q]
    for i, label in enumerate(['cos_inc']):
        results_df[f'{label}_{q}'] = y_pred_q[:, i]
print(results_df.head())
results_df.to_csv(f"{data_args.log_dir}/{datetime_dir}/test_predictions.csv", index=False)
# Access results



