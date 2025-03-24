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
from tqdm import tqdm

warnings.filterwarnings("ignore")

import sys
from os import path
ROOT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print("running from ", ROOT_DIR) 

from transforms.transforms import *
from dataset.dataset import LightSpecDataset, FineTuneDataset
from dataset.sampler import BalancedDistributedSampler
from nn.astroconf import Astroconformer, AstroEncoderDecoder
from nn.models import *
from nn.simsiam import MultiModalSimSiam, MultiModalSimCLR
from nn.moco import *
from nn.simsiam import SimSiam, projection_MLP
from nn.optim import CQR
from nn.utils import init_model, load_checkpoints_ddp
from util.utils import *
from nn.train import *
from tests.test_unique_sampler import run_sampler_tests
from features import multimodal_umap, create_umap
import generator


# MODELS = {'Astroconformer': Astroconformer, 'CNNEncoder': CNNEncoder, 'MultiEncoder': MultiEncoder, 'MultiTaskRegressor': MultiTaskRegressor,
#            'AstroEncoderDecoder': AstroEncoderDecoder, 'CNNEncoderDecoder': CNNEncoderDecoder,}

R_SUN_KM = 6.957e5

torch.cuda.empty_cache()

torch.manual_seed(1234)
np.random.seed(1234)


def create_train_test_dfs(meta_columns):
    nss_df = pd.read_csv('/data/lightPred/tables/nss_dataset.csv')
    berger_df = pd.read_csv('/data/lightPred/tables/berger_catalog_full.csv')
    kepler_meta = pd.read_csv('/data/lightPred/tables/kepler_dr25_meta_data.csv')
    nss_df = nss_df.merge(berger_df, on='KID', how='left', suffixes=('', '_berger')).merge(kepler_meta[['KID', 'KMAG']], on='KID', how='left')
    nss_df['kmag_abs'] = nss_df['KMAG'] - 5 * np.log10(nss_df['Dist']) + 5
    binaries = nss_df[nss_df['binary_prob'] > 0.5]
    print("number of binaries: ", binaries.shape[0], "number pf nss samples: ", nss_df.shape[0])
    lamost_kepler_df = pd.read_csv('/data/lamost/lamost_dr8_gaia_dr3_kepler_ids.csv')
    lamost_kepler_df = lamost_kepler_df[~lamost_kepler_df['KID'].isna()]
    lamost_kepler_df['KID'] = lamost_kepler_df['KID'].astype(int)
    final_df = lamost_kepler_df.merge(nss_df, on='KID', how='inner')
    true_binaries = final_df[final_df['binary_prob'] == 1]
    print("number of confirmed binaries: ", true_binaries.shape[0], "number pf nss samples: ", final_df.shape[0])
    # final_df.dropna(subset=['binary_prob', 'k_prob', 'p_prob'], inplace=True)
    final_df['binary_prob_hard'] = final_df['binary_prob'].apply(lambda x: 1 if x > 0.5 else 0).astype(int)
    train_df, val_df  = train_test_split(final_df, test_size=0.2, random_state=42)
    print("final_df columns: ", final_df.columns)
    print("number of nss: ", len(final_df), " with prob 1: ", final_df[final_df['binary_prob_hard'] == 1].shape[0])
    plt.hist(final_df['binary_prob'], bins=40)
    plt.savefig('/data/lightSpec/images/finetune_nss_dataset.png')
    return train_df, val_df 


current_date = datetime.date.today().strftime("%Y-%m-%d")
datetime_dir = f"nss_finetune_{current_date}"

local_rank, world_size, gpus_per_node = setup()
args_dir = '/data/lightSpec/nn/config_finetune_nss.yaml'
data_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Data'])
exp_num = data_args.exp_num
if data_args.test_run:
    datetime_dir = f"test_{current_date}"

os.makedirs(f"{data_args.log_dir}/{datetime_dir}", exist_ok=True)

train_dataset, val_dataset, test_dataset, complete_config = generator.get_data(data_args,
                                                                                create_train_test_dfs,
                                                                                datase_name='FineTune')

train_df, test_df = create_train_test_dfs(data_args.meta_columns)
binary_labels = train_df['binary_prob'].tolist()

train_sampler = BalancedDistributedSampler(
    dataset=train_dataset,
    labels=binary_labels,
    num_replicas=world_size,
    rank=local_rank,
    shuffle=True,
    balanced_ratio=1.0,  # 1.0 means equal distribution
    verbose=True  # Set to True for debugging
)

train_loader = DataLoader(train_dataset, batch_size=data_args.batch_size,
                             sampler=train_sampler, collate_fn=kepler_collate_fn)
val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=data_args.batch_size,
                         sampler=val_sampler, collate_fn=kepler_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=data_args.batch_size, 
                         collate_fn=kepler_collate_fn)

model, optim_args, complete_config = generator.get_model(data_args,
                            args_dir,
                            complete_config,
                            local_rank)

loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=float(optim_args.max_lr), weight_decay=float(optim_args.weight_decay))

trainer = ClassificationTrainer(
    model=model,
    optimizer=optimizer,
    criterion=loss_fn,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    device=local_rank,
    output_dim=len(data_args.prediction_labels),
    num_cls=len(data_args.prediction_labels),
    use_w = True,
    log_path=data_args.log_dir,
    exp_num=datetime_dir,
    exp_name=f"nss_finetune_{exp_num}",
)
complete_config.update(
    {"trainer": trainer.__dict__,
    "loss_fn": str(loss_fn),
    "optimizer": str(optimizer)}
)
   
config_save_path = f"{data_args.log_dir}/{datetime_dir}/finetune_nss_{exp_num}_complete_config.yaml"
with open(config_save_path, "w") as config_file:
    json.dump(complete_config, config_file, indent=2, default=str)

print(f"Configuration (with model structure) saved at {config_save_path}.")

fit_res = trainer.fit(num_epochs=data_args.num_epochs, device=local_rank,
                        early_stopping=10, best='loss', conf=True) 
output_filename = f'{data_args.log_dir}/{datetime_dir}/{model_name}_nss_{exp_num}.json'
with open(output_filename, "w") as f:
    json.dump(fit_res, f, indent=2)
fig, axes = plot_fit(fit_res, legend=exp_num, train_test_overlay=True)
plt.savefig(f"{data_args.log_dir}/{datetime_dir}/fit_{model_name}_nss_{exp_num}.png")
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


