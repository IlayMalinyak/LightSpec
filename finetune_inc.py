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
from tests.test_unique_sampler import run_sampler_tests
from features import multimodal_umap, create_umap
import generator

META_COLUMNS = ['KID', 'Teff', 'logg', 'FeH', 'Rstar', 'Mstar', 'kmag_abs']


MODELS = {'Astroconformer': Astroconformer, 'CNNEncoder': CNNEncoder, 'MultiEncoder': MultiEncoder, 'MultiTaskRegressor': MultiTaskRegressor,
           'AstroEncoderDecoder': AstroEncoderDecoder, 'CNNEncoderDecoder': CNNEncoderDecoder,}

R_SUN_KM = 6.957e5

torch.cuda.empty_cache()

torch.manual_seed(1234)
np.random.seed(1234)


def create_train_test_dfs(meta_columns):
    period_catalog = pd.read_csv('/data/lightPred/tables/kepler_predictions_clean_seg_0_1_2_median.csv')
    period_catalog.rename(columns={'predicted period': 'Prot'}, inplace=True)
    santos_catalog = pd.read_csv('/data/lightPred/tables/santos_periods_19_21.csv')
    frasca_catalog = Table.read('/data/lightPred/tables/frasca2016.fit', format='fits').to_pandas()
    cks_catalog = Table.read('/data/lightPred/tables/CKS2017.fit', format='fits').to_pandas()

    kepler_apogee = pd.read_csv('/data/apogee/crossmatched_catalog_Kepler.csv')
    kepler_apogee = kepler_apogee[~kepler_apogee['APOGEE_VSINI'].isna()].rename(columns={'APOGEE_VSINI': 'vsini'})
    kepler_apogee['vsini_ref'] = 'apogee'
    lamost_kepler_df = pd.read_csv('/data/lamost/lamost_dr8_gaia_dr3_kepler_ids.csv')
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
    berger_catalog = pd.read_csv('/data/lightPred/tables/berger_catalog.csv')
    final_df = final_df.merge(berger_catalog, on='KID', suffixes=['', '_berger'])
    kmag_df = pd.read_csv('/data/lightPred/tables/kepler_dr25_meta_data.csv')
    final_df = final_df.merge(kmag_df[['KID', 'KMAG']], on='KID', how='left')
    final_df['kmag_abs'] = final_df['KMAG'] - 5 * np.log10(final_df['Dist']) + 5
    
    final_df['sin_inc'] = (final_df['vsini'] * final_df['Prot'] * 24 * 3600
                                / (2 * np.pi * final_df['Rstar'] * R_SUN_KM) )
    final_df['log_sin_inc'] = -np.log(final_df['sin_inc'] + 1e-3)
    final_df['inc'] = np.arcsin(final_df['sin_inc']) * 180 / np.pi
    final_df['norm_vsini'] = (final_df['vsini'] - final_df['vsini'].min()) / (final_df['vsini'].max() - final_df['vsini'].min())
    final_df['norm_Prot'] = (final_df['Prot'] - final_df['Prot'].min()) / (final_df['Prot'].max() - final_df['Prot'].min())
    final_df = final_df[~final_df['inc'].isna()]
    print("prot nans: ", final_df['Prot'].isna().sum(), " Rstar nans: ", final_df['Rstar'].isna().sum(), "VSINI nans: ", final_df['vsini'].isna().sum(), "inc nans: ", final_df['inc'].isna().sum())
    
    train_df, val_df  = train_test_split(final_df, test_size=0.2, random_state=42)
    print("samples kepler with period: ", len(period_catalog), " kepler_apogee", len(kepler_apogee),
     " lamost-kepler-apogee:", len(lamost_kepler_df), " lamost_kepler_apogee_periods_1", len(lamost_kepler_periods_1),
     " lamost_kepler_apogee_periods_2", len(lamost_kepler_periods_2), " final_df", len(final_df)
    )
    print("final_df columns: ", final_df.columns)
    
    for ref in final_df['vsini_ref'].unique():
        print("ref: ", ref, " samples: ", len(final_df[final_df['vsini_ref'] == ref]))
        ref_df = train_df[train_df['vsini_ref'] == ref]
        if len(ref_df) < 20:
            continue
        plt.hist(ref_df['log_sin_inc'], bins=40, histtype='step', density=True, label=ref)
    plt.legend()
    plt.savefig(f"/data/lightSpec/images/log_sin_inc_hist.png")
    plt.close()


    return train_df, val_df 


current_date = datetime.date.today().strftime("%Y-%m-%d")
datetime_dir = f"inc_finetune_{current_date}"

local_rank, world_size, gpus_per_node = setup()
args_dir = '/data/lightSpec/nn/config_finetune_inc.yaml'
data_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Data'])
exp_num = data_args.exp_num
light_model_name = data_args.light_model_name
spec_model_name = data_args.spec_model_name
combined_model_name = data_args.combined_model_name
if data_args.test_run:
    datetime_dir = f"test_{current_date}"

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
    light, spec, y, _, _, info = train_dataset[i]
    print("train dataset: ", light.shape, spec.shape, y)

model, optim_args, complete_config = generator.get_model(data_args, args_dir, complete_config, local_rank)

loss_fn = CQR(quantiles=optim_args.quantiles)
optimizer = torch.optim.Adam(model.parameters(), lr=float(optim_args.max_lr), weight_decay=float(optim_args.weight_decay))

# kfold_trainer = KFoldTrainer(
#     model=model,
#     optimizer=optimizer,
#     criterion=loss_fn,
#     dataset=train_dataset,
#     device=local_rank,
#     n_splits=5,
#     batch_size=data_args.batch_size,
#     output_dim=len(data_args.prediction_labels),
#     num_quantiles=len(optim_args.quantiles),
#     log_path=data_args.log_dir,
#     exp_num=datetime_dir,
#     exp_name=f"inc_finetune_{exp_num}",
# )

# Run k-fold cross validation
# k_results = kfold_trainer.run_kfold(num_epochs=100, early_stopping=10)
      

# test_dataloader = DataLoader(test_dataset, batch_size=data_args.batch_size, shuffle=False, collate_fn=kepler_collate_fn)
# final_results = kfold_trainer.train_final_model_and_test(
#     test_dataloader=test_dataloader,
#     num_epochs=1000,
#     early_stopping=15
# )


trainer = RegressorTrainer(
    model=model,
    optimizer=optimizer,
    criterion=loss_fn,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    device=local_rank,
    output_dim=len(data_args.prediction_labels),
    num_quantiles=len(optim_args.quantiles),
    use_w = True,
    log_path=data_args.log_dir,
    exp_num=datetime_dir,
    exp_name=f"inc_finetune_{exp_num}",
)


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
output_filename = f'{data_args.log_dir}/{datetime_dir}/{model_name}_inc_{exp_num}.json'
with open(output_filename, "w") as f:
    json.dump(fit_res, f, indent=2)
fig, axes = plot_fit(fit_res, legend=exp_num, train_test_overlay=True)
plt.savefig(f"{data_args.log_dir}/{datetime_dir}/fit_{model_name}_inc_{exp_num}.png")
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



