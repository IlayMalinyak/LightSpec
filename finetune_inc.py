import os
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
from nn.models import CNNEncoder, CNNEncoderDecoder, MultiEncoder, MultiTaskRegressor, Transformer
from nn.simsiam import MultiModalSimSiam, MultiModalSimCLR
from nn.moco import MultimodalMoCo, PredictiveMoco, MocoTuner
from nn.simsiam import SimSiam, projection_MLP
from nn.optim import CQR
from nn.utils import init_model, load_checkpoints_ddp
from util.utils import *
from nn.train import KFoldTrainer
from tests.test_unique_sampler import run_sampler_tests
from features import multimodal_umap, create_umap

META_COLUMNS = ['KID', 'Teff', 'logg', 'FeH', 'Rstar', 'Mstar', 'kmag_abs']


MODELS = {'Astroconformer': Astroconformer, 'CNNEncoder': CNNEncoder, 'MultiEncoder': MultiEncoder, 'MultiTaskRegressor': MultiTaskRegressor,
           'AstroEncoderDecoder': AstroEncoderDecoder, 'CNNEncoderDecoder': CNNEncoderDecoder,}

R_SUN_KM = 6.957e5

torch.cuda.empty_cache()

torch.manual_seed(1234)
np.random.seed(1234)


def create_train_test_dfs():
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
    final_df['inc'] = np.arcsin(final_df['sin_inc']) * 180 / np.pi
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
        plt.hist(ref_df['inc'], bins=20, histtype='step', density=True, label=ref)
    plt.legend()
    plt.savefig(f"/data/lightSpec/images/inc_hist.png")
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
light_model_args = Container(**yaml.safe_load(open(args_dir, 'r'))[f'{light_model_name}_lc'])
spec_model_args = Container(**yaml.safe_load(open(args_dir, 'r'))[f'{spec_model_name}_spec'])
combined_model_args = Container(**yaml.safe_load(open(args_dir, 'r'))[f'{combined_model_name}_combined'])
conformer_args_lc = Container(**yaml.safe_load(open(args_dir, 'r'))['Conformer_lc'])
conformer_args_spec = Container(**yaml.safe_load(open(args_dir, 'r'))['Conformer_spec'])
conformer_args_combined = Container(**yaml.safe_load(open(args_dir, 'r'))['Conformer_combined'])
lightspec_args = Container(**yaml.safe_load(open(args_dir, 'r'))['MultiEncoder_lightspec'])
transformer_args_lightspec = Container(**yaml.safe_load(open(args_dir, 'r'))['Transformer_lightspec'])
predictor_args = Container(**yaml.safe_load(open(args_dir, 'r'))['predictor'])
loss_args = Container(**yaml.safe_load(open(args_dir, 'r'))['loss'])
moco_args = Container(**yaml.safe_load(open(args_dir, 'r'))['MoCo'])
optim_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Optimization'])
tuner_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Tuner'])

os.makedirs(f"{data_args.log_dir}/{datetime_dir}", exist_ok=True)

light_transforms = Compose([ RandomCrop(int(data_args.max_len_lc)),
                        MovingAvg(13),
                        ACF(max_lag_day=None, max_len=int(data_args.max_len_lc)),
                        Normalize('std'),
                        ToTensor(), ])
                        
spec_transforms = Compose([LAMOSTSpectrumPreprocessor(continuum_norm=data_args.continuum_norm, plot_steps=False),
                            ToTensor()
                           ])
train_df, test_df = create_train_test_dfs()
train_dataset = FineTuneDataset(df=train_df, light_transforms=light_transforms,
                                spec_transforms=spec_transforms,
                                npy_path = '/data/lightPred/data/npy',
                                spec_path = data_args.spectra_dir,
                                light_seq_len=int(data_args.max_len_lc),
                                spec_seq_len=int(data_args.max_len_spectra),
                                use_acf=data_args.use_acf,
                                labels=['inc']
                                )

test_dataset =FineTuneDataset(df=test_df, light_transforms=light_transforms,
                                spec_transforms=spec_transforms,
                                npy_path = '/data/lightPred/data/npy',
                                spec_path = data_args.spectra_dir,
                                light_seq_len=int(data_args.max_len_lc),
                                spec_seq_len=int(data_args.max_len_spectra),
                                use_acf=data_args.use_acf,
                                labels=['inc']
                                )

for i in range(10):
    light, spec, _, _, y, info = train_dataset[i]
    print("train dataset: ", light.shape, spec.shape, y)

light_backbone = MODELS[light_model_name](light_model_args, conformer_args=conformer_args_lc)
if light_model_name == 'Astroconformer':
    light_backbone.pred_layer = torch.nn.Identity()
light_model = SimSiam(light_backbone)
# light_model = light_backbone

light_model = init_model(light_model, light_model_args)

num_params_lc_all = sum(p.numel() for p in light_model.parameters() if p.requires_grad)
print(f"Number of trainble parameters in light model: {num_params_lc_all}")
num_params_lc = sum(p.numel() for p in light_model.encoder.parameters() if p.requires_grad)
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

moco_model = PredictiveMoco(spec_model.encoder, light_model.encoder,
                         transformer_args_lightspec,
                         predictor_args.get_dict(),
                         loss_args,
                        combined_encoder=combined_encoder,
                        calc_loss=False,
                        **moco_args.get_dict()).to(local_rank)



if data_args.load_checkpoint:
    datetime_dir = os.path.basename(os.path.dirname(data_args.checkpoint_path))
    exp_num = os.path.basename(data_args.checkpoint_path).split('.')[0].split('_')[-1]
    print(datetime_dir)
    print("loading checkpoint from: ", data_args.checkpoint_path)
    moco_model = load_checkpoints_ddp(moco_model, data_args.checkpoint_path)
    print("loaded checkpoint from: ", data_args.checkpoint_path)

model = MocoTuner(moco_model, tuner_args.get_dict()).to(local_rank)
model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainble parameters: {num_params}")
all_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters in model: {all_params}")
print("number of training samples: ", len(train_dataset), len(test_dataset))

loss_fn = CQR(quantiles=optim_args.quantiles)
optimizer = torch.optim.Adam(model.parameters(), lr=float(optim_args.max_lr), weight_decay=float(optim_args.weight_decay))

kfold_trainer = KFoldTrainer(
    model=model,
    optimizer=optimizer,
    criterion=loss_fn,
    dataset=train_dataset,
    device=local_rank,
    n_splits=5,
    batch_size=data_args.batch_size,
    output_dim=1,
    log_path=data_args.log_dir,
    exp_num=datetime_dir,
    exp_name=f"inc_finetune_{exp_num}",
)

# Run k-fold cross validation
results = kfold_trainer.run_kfold(num_epochs=100, early_stopping=10)

# Access results
print("Average metrics:", results['average_metrics'])
print("Individual fold results:", results['fold_results'])

