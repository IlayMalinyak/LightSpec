
import os
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Subset
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import OneCycleLR
from astropy.io import fits
from astropy.table import Table
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
from nn.train import ContrastiveTrainer, MaskedSSLTrainer
from nn.utils import deepnorm_init, load_checkpoints_ddp
from nn.optim import CQR
from util.utils import *
from features import create_umap
import generator

R_SUN_KM = 6.957e5


def priority_merge_prot(dataframes, target_df):
    """
    Merge 'Prot' values from multiple dataframes into target dataframe in priority order,
    using 'KID' as the merge key. Much more efficient implementation.
    
    Args:
        dataframes: List of dataframes, each containing 'Prot' and 'KID' columns (in decreasing priority order)
        target_df: Target dataframe to merge 'Prot' values into (must contain 'KID' column)
    
    Returns:
        DataFrame with aggregated 'Prot' values merged into target_df
    """
    # Create a copy of the target dataframe
    result = target_df.copy()
    
    # Create an empty dataframe with just KID and Prot columns
    prot_values = pd.DataFrame({'KID': [], 'Prot': [], 'Prot_ref': []})
    
    # Process dataframes in priority order
    for df in dataframes:
        print(f"Processing dataframe with {len(df)} rows. currently have {len(prot_values)} prot values")
        # Extract just the KID and Prot columns
        current = df[['KID', 'Prot', 'Prot_ref']].copy()
        
        # Only add keys that aren't already in our prot_values dataframe
        missing_keys = current[~current['KID'].isin(prot_values['KID'])]
        
        # Concatenate with existing values
        prot_values = pd.concat([prot_values, missing_keys])
    
    # Merge the aggregated Prot values into the result dataframe
    result = result.merge(prot_values, on='KID', how='left')
    
    return result

def create_train_test_dfs(meta_columns):
    kepler_df = get_all_samples_df(num_qs=None, read_from_csv=True)
    berger_cat = pd.read_csv('/data/lightPred/tables/berger_catalog_full.csv')
    kepler_meta = pd.read_csv('/data/lightPred/tables/kepler_dr25_meta_data.csv')
    activity_cat = pd.read_csv('/data/logs/activity_proxies/proxies_full_dist.csv')
    kepler_df = kepler_df.merge(berger_cat, on='KID').merge(kepler_meta[['KID', 'KMAG']], on='KID', how='left').merge(activity_cat, on='KID', how='left')
    kepler_df['kmag_abs'] = kepler_df['KMAG'] - 5 * np.log10(kepler_df['Dist']) + 5

    lightpred_df = pd.read_csv('/data/lightPred/tables/kepler_predictions_clean_seg_0_1_2_median.csv')
    lightpred_df['Prot_ref'] = 'lightpred'
    lightpred_df.rename(columns={'predicted period': 'Prot'}, inplace=True)

    santos_df = pd.read_csv('/data/lightPred/tables/santos_periods_19_21.csv')
    santos_df['Prot_ref'] = 'santos'
    mcq14_df = pd.read_csv('/data/lightPred/tables/Table_1_Periodic.txt')
    mcq14_df['Prot_ref'] = 'mcq14'
    reinhold_df = pd.read_csv('/data/lightPred/tables/reinhold2023.csv')
    reinhold_df['Prot_ref'] = 'reinhold'

    p_dfs = [lightpred_df, santos_df, mcq14_df, reinhold_df]
    kepler_df = priority_merge_prot(p_dfs, kepler_df)

    train_df, test_df = train_test_split(kepler_df, test_size=0.2, random_state=1234)
    return train_df, test_df

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
torch.cuda.empty_cache()

models = {'Astroconformer': Astroconformer, 'CNNEncoder': CNNEncoder, 'MultiEncoder': MultiEncoder,
            "CNNEncoderDecoder": CNNEncoderDecoder, 'CNNRegressor': CNNRegressor}

current_date = datetime.date.today().strftime("%Y-%m-%d")
datetime_dir = f"light_{current_date}"

local_rank, world_size, gpus_per_node = setup()
args_dir = '/data/lightSpec/nn/full_config.yaml'
data_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Data'])
exp_num = data_args.exp_num


train_dataset, val_dataset, test_dataset, complete_config = generator.get_data(data_args, data_generation_fn=create_train_test_dfs, dataset_name='Kepler')

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

_, optim_args, complete_config, model, _ = generator.get_model(data_args, args_dir, complete_config, local_rank)

model = model.to(local_rank)
model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

if data_args.create_umap:
    umap_df = create_umap(model.module.simsiam.encoder, test_dataloader, local_rank, use_w=False, dual=False)
    print("umap created: ", umap_df.shape)
    umap_df.to_csv(f"{data_args.log_dir}/{checkpoint_dir}/umap_{checkpoint_exp}_ssl.csv", index=False)
    print(f"umap saved at {data_args.log_dir}/{checkpoint_dir}/umap_{checkpoint_exp}_ssl.csv")
    exit()
    
loss_fn = CQR(quantiles=optim_args.quantiles, reduction='none')
ssl_loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=float(optim_args.max_lr), weight_decay=float(optim_args.weight_decay))
scaler = GradScaler()
total_steps = int(data_args.num_epochs) * len(train_dataloader)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                    T_max=total_steps,
                                                    eta_min=float(optim_args.max_lr)/10)
# scheduler = OneCycleLR(
#         optimizer,
#         max_lr=float(optim_args.max_lr),
#         epochs= int(data_args.num_epochs),
#         steps_per_epoch = len(train_dataloader),
#         pct_start=float(optim_args.warmup_pct),
#         anneal_strategy='cos',
#         cycle_momentum=True,
#         base_momentum=0.85,
#         max_momentum=0.95,
#         div_factor=10.0,
#         final_div_factor=100.0
#     )

# Save the complete configuration to a YAML file


if data_args.masked_transform:
    trainer = MaskedRegressorTrainer(model=model, optimizer=optimizer,
                        criterion=loss_fn, ssl_criterion=ssl_loss_fn, output_dim=1, scaler=scaler,
                       scheduler=scheduler, train_dataloader=train_dataloader,
                       val_dataloader=val_dataloader, device=local_rank,
                           exp_num=datetime_dir, log_path=data_args.log_dir, range_update=None,
                           accumulation_step=1, max_iter=np.inf,
                        exp_name=f"{model_name}_lc_{exp_num}")  
else:
    trainer = ContrastiveTrainer(model=model, optimizer=optimizer,
                            criterion=loss_fn, output_dim=1, scaler=scaler,
                        scheduler=scheduler, train_dataloader=train_dataloader,
                        val_dataloader=val_dataloader, device=local_rank, ssl_weight=data_args.ssl_weight,
                                weight_decay=True, num_quantiles=len(optim_args.quantiles), stack_pairs=False,
                            exp_num=datetime_dir, log_path=data_args.log_dir, range_update=None,
                            accumulation_step=1, max_iter=np.inf,
                            exp_name=f"test_prot_lc_{exp_num}") 


complete_config.update(
    {"trainer": trainer.__dict__,
    "loss_fn": str(loss_fn),
    "optimizer": str(optimizer)}
)
   
config_save_path = f"{data_args.log_dir}/{datetime_dir}/finetune_sim{exp_num}_complete_config.yaml"
with open(config_save_path, "w") as config_file:
    json.dump(complete_config, config_file, indent=2, default=str)

print(f"Configuration (with model structure) saved at {config_save_path}.")


# complete_config = {
#     "model_name": model_name,
#     "data_args": data_args.__dict__,
#     "model_args": model_args.__dict__,
#     "astroconf_args": astroconf_args.__dict__,
#     "cnn_args": cnn_args.__dict__,
#     "optim_args": optim_args.__dict__,
#     "num_params": num_params,
#     "model_structure": str(model),  # Add the model structure to the configuration
#     "transforms": str(transforms),
#     'trainer': trainer.__dict__
# }
# config_save_path = f"{data_args.log_dir}/{datetime_dir}/{model_name}_lc_{exp_num}_complete_config.json"
# with open(config_save_path, "w") as config_file:
#      json.dump(complete_config, config_file, indent=2, default=str)
# print(f"Configuration (with model structure) saved at {config_save_path}.")

fit_res = trainer.fit(num_epochs=data_args.num_epochs, device=local_rank,
                        early_stopping=40, best='loss', conf=True) 
output_filename = f'{data_args.log_dir}/{datetime_dir}/{model_name}_lc_{exp_num}.json'
with open(output_filename, "w") as f:
    json.dump(fit_res, f, indent=2)
fig, axes = plot_fit(fit_res, legend=exp_num, train_test_overlay=True)
plt.savefig(f"{data_args.log_dir}/{datetime_dir}/fit_{model_name}_lc_{exp_num}.png")
plt.clf()

preds_val, targets_val, info = trainer.predict(val_dataloader, device=local_rank)

preds, targets, info = trainer.predict(test_dataloader, device=local_rank)

print(info.keys())

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
df = save_predictions_to_dataframe(preds, targets, info, data_args.labels, optim_args.quantiles,
 id_name='KID', info_keys=['Prot_ref'])
print(df.columns)
df.to_csv(f"{data_args.log_dir}/{datetime_dir}/{model_name}_lc_decode2_{exp_num}.csv", index=False)
print('predictions saved in', f"{data_args.log_dir}/{datetime_dir}/{model_name}_lc_decode2_{exp_num}.csv") 
df_cqr = save_predictions_to_dataframe(preds_cqr, targets, info, data_args.labels, optim_args.quantiles,
 id_name='KID', info_keys=['Prot_ref'])
df_cqr.to_csv(f"{data_args.log_dir}/{datetime_dir}/{model_name}_lc_decode2_{exp_num}_cqr.csv", index=False)
print('predictions saved in', f"{data_args.log_dir}/{datetime_dir}/{model_name}_lc_decode2_{exp_num}_cqr.csv")

umap_df = create_umap(model.module.simsiam.encoder, test_dataloader, local_rank, use_w=False, dual=False)
print("umap created: ", umap_df.shape)
umap_df.to_csv(f"{data_args.log_dir}/{datetime_dir}/umap_{exp_num}_ssl.csv", index=False)
print(f"umap saved at {data_args.log_dir}/{datetime_dir}/umap_{exp_num}_ssl.csv")
