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
from dataset.dataset import LightSpecDatasetV2, create_unique_loader
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

prediction_labels = ['teff', 'logg', 'feh']
torch.cuda.empty_cache()

torch.manual_seed(1234)
np.random.seed(1234)


def test_dataset_samples(dataset, num_iters=100):
    start = time.time()
    for i in range(num_iters):
        lc, spec, lc_target, spec_target, lc_info, spec_info = dataset[i]
        print(lc.shape, spec.shape, lc_target.shape, spec_target.shape, lc.max(), spec.max())
    print(f"Averaged over {num_iters} iterations: {(time.time()-start) / num_iters} seconds per iteration")

def split_datasets_with_shared_samples(
    lamost_df, 
    kepler_df, 
    shared_df,
    lamost_id='combined_obsid',
    kepler_id='KID',
    test_size=0.2,
    val_size=0.1,
    random_state=42
):
    """
    Split multiple dataframes while maintaining consistency for shared samples.
    
    Args:
        lamost_df (pd.DataFrame): LAMOST spectra dataframe
        kepler_df (pd.DataFrame): Kepler light curve dataframe
        shared_df (pd.DataFrame): Shared samples dataframe with both IDs
        lamost_id (str): Column name for LAMOST IDs
        kepler_id (str): Column name for Kepler IDs
        test_size (float): Proportion of data for test set
        val_size (float): Proportion of data for validation set
        random_state (int): Random seed for reproducibility
        
    Returns:
        dict: Dictionary containing train/val/test splits for all dataframes
    """
    # First, split the shared dataset to establish the sample groups
    total_val_test_size = test_size + val_size
    
    # Get unique pairs of IDs from shared dataset
    shared_id_pairs = shared_df[[lamost_id, kepler_id]].drop_duplicates()
    
    # First split: train vs (val+test)
    train_pairs, valtest_pairs = train_test_split(
        shared_id_pairs,
        test_size=total_val_test_size,
        random_state=random_state
    )
    
    # Second split: val vs test
    val_pairs, test_pairs = train_test_split(
        valtest_pairs,
        test_size=(test_size/total_val_test_size),
        random_state=random_state
    )
    
    # Function to split LAMOST dataframe
    def split_lamost_df(df):
        train = df[df[lamost_id].isin(train_pairs[lamost_id])]
        val = df[df[lamost_id].isin(val_pairs[lamost_id])]
        test = df[df[lamost_id].isin(test_pairs[lamost_id])]
        
        # Handle non-shared samples
        non_shared = df[~df[lamost_id].isin(shared_id_pairs[lamost_id])]
        if len(non_shared) > 0:
            non_shared_train, non_shared_valtest = train_test_split(
                non_shared,
                test_size=total_val_test_size,
                random_state=random_state
            )
            non_shared_val, non_shared_test = train_test_split(
                non_shared_valtest,
                test_size=(test_size/total_val_test_size),
                random_state=random_state
            )
            
            train = pd.concat([train, non_shared_train])
            val = pd.concat([val, non_shared_val])
            test = pd.concat([test, non_shared_test])
        
        return train, val, test
    
    # Function to split Kepler dataframe
    def split_kepler_df(df):
        train = df[df[kepler_id].isin(train_pairs[kepler_id])]
        val = df[df[kepler_id].isin(val_pairs[kepler_id])]
        test = df[df[kepler_id].isin(test_pairs[kepler_id])]
        
        # Handle non-shared samples
        non_shared = df[~df[kepler_id].isin(shared_id_pairs[kepler_id])]
        if len(non_shared) > 0:
            non_shared_train, non_shared_valtest = train_test_split(
                non_shared,
                test_size=total_val_test_size,
                random_state=random_state
            )
            non_shared_val, non_shared_test = train_test_split(
                non_shared_valtest,
                test_size=(test_size/total_val_test_size),
                random_state=random_state
            )
            
            train = pd.concat([train, non_shared_train])
            val = pd.concat([val, non_shared_val])
            test = pd.concat([test, non_shared_test])
        
        return train, val, test
    
    # Split all datasets
    lamost_splits = split_lamost_df(lamost_df)
    kepler_splits = split_kepler_df(kepler_df)
    shared_splits = split_lamost_df(shared_df)  # Using LAMOST IDs for shared df
    
    # Return dictionary with all splits
    return {
        'lamost': {
            'train': lamost_splits[0],
            'val': lamost_splits[1],
            'test': lamost_splits[2]
        },
        'kepler': {
            'train': kepler_splits[0],
            'val': kepler_splits[1],
            'test': kepler_splits[2]
        },
        'shared': {
            'train': shared_splits[0],
            'val': shared_splits[1],
            'test': shared_splits[2]
        }
    }



current_date = datetime.date.today().strftime("%Y-%m-%d")
datetime_dir = f"combined_deocde_{current_date}"

local_rank, world_size, gpus_per_node = setup()
args_dir = '/data/lightSpec/nn/config_combined_decode.yaml'
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

lamost_catalog = pd.read_csv('/data/lamost/lamost_afgkm_teff_3000_7500_catalog.csv', sep='|')
lamost_catalog = lamost_catalog.drop_duplicates(subset=['combined_obsid'])
lamost_catalog = lamost_catalog[lamost_catalog['combined_snrg'] > 0]
lamost_catalog = lamost_catalog.dropna(subset=['combined_teff', 'combined_logg', 'combined_feh'])
print(lamost_catalog['combined_snrg'].describe())


kepler_df = get_all_samples_df(num_qs=None, read_from_csv=True)
kepler_meta = pd.read_csv('/data/lightPred/tables/berger_catalog.csv')
kmag_df = pd.read_csv('/data/lightPred/tables/kepler_dr25_meta_data.csv') 
kepler_df = kepler_df.merge(kepler_meta, on='KID', how='left', suffixes=['_lightpred', '']).merge(kmag_df[['KID','KMAG']], on='KID', how='left')
kepler_df['kmag_abs'] = kepler_df['KMAG'] - 5 * np.log10(kepler_df['Dist']) + 5
print("number of samples: ", len(kepler_df))
print("dist in df:", 'Dist' in kepler_df.columns)
lamost_kepler_df = pd.read_csv('/data/lamost/lamost_dr8_gaia_dr3_kepler_ids.csv')
lamost_kepler_df = lamost_kepler_df[~lamost_kepler_df['KID'].isna()]
lamost_kepler_df['KID'] = lamost_kepler_df['KID'].astype(int)
lamost_kepler_df = lamost_kepler_df.merge(kepler_df[META_COLUMNS], on='KID', how='inner')
lamost_kepler_df['main_seq'] = lamost_kepler_df.apply(giant_cond, axis=1)
lamost_kepler_df = lamost_kepler_df[lamost_kepler_df['main_seq']==True]
print("shared columns :", lamost_kepler_df.columns)
lamost_kepler_df['combined_obsid'] = lamost_kepler_df['ObsID']

data = split_datasets_with_shared_samples(lamost_catalog, kepler_df, lamost_kepler_df, test_size=0.2, val_size=0.1, random_state=42)

train_dataset = LightSpecDatasetV2(spec_df=data['lamost']['train'],
                                   lc_df=data['kepler']['train'],
                                   shared_df=data['shared']['train'],
                                   light_transforms=light_transforms,
                                spec_transforms=spec_transforms,
                                npy_path = '/data/lightPred/data/npy',
                                spec_data_dir = data_args.spectra_dir,
                                light_seq_len=int(data_args.max_len_lc),
                                use_acf=data_args.use_acf,
                                spec_seq_len=int(data_args.max_len_spectra),
                                )

val_dataset = LightSpecDatasetV2(spec_df=data['lamost']['val'],
                                 lc_df=data['kepler']['val'],
                                 shared_df=data['shared']['val'],
                                 light_transforms=light_transforms,
                                spec_transforms=spec_transforms,
                                npy_path = '/data/lightPred/data/npy',
                                spec_data_dir = data_args.spectra_dir,
                                light_seq_len=int(data_args.max_len_lc),
                                use_acf=data_args.use_acf,
                                spec_seq_len=int(data_args.max_len_spectra),
                                )

test_dataset = LightSpecDatasetV2(spec_df=data['lamost']['test'],
                                  lc_df=data['kepler']['test'],
                                  shared_df=data['shared']['test'],
                                  light_transforms=light_transforms,
                                spec_transforms=spec_transforms,
                                npy_path = '/data/lightPred/data/npy',
                                spec_data_dir = data_args.spectra_dir,
                                light_seq_len=int(data_args.max_len_lc),
                                use_acf=data_args.use_acf,
                                spec_seq_len=int(data_args.max_len_spectra),
                                )


test_dataset_samples(train_dataset, num_iters=100)

train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank)
test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=world_size, rank=local_rank)

train_loader = DataLoader(train_dataset,
                          batch_size=data_args.batch_size,
                          sampler=train_sampler,
                          collate_fn=kepler_collate_fn)
val_loader = DataLoader(val_dataset,
                        batch_size=data_args.batch_size,
                        sampler=val_sampler,
                          collate_fn=kepler_collate_fn)
test_loader = DataLoader(test_dataset,
                         batch_size=data_args.batch_size,
                         sampler=test_sampler,
                          collate_fn=kepler_collate_fn)
