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
from dataset.dataset import *
from dataset.sampler import BalancedDistributedSampler
from nn.astroconf import Astroconformer, AstroEncoderDecoder
from nn.models import *
from nn.simsiam import MultiModalSimSiam, MultiModalSimCLR
from nn.moco import *
from nn.simsiam import SimSiam, projection_MLP
from nn.optim import CQR
from nn.utils import init_model, load_checkpoints_ddp, deepnorm_init
from util.utils import *
from nn.train import *
from tests.test_unique_sampler import run_sampler_tests
from features import multimodal_umap, create_umap

MODELS = {'Astroconformer': Astroconformer, 'CNNEncoder': CNNEncoder, 'MultiEncoder': MultiEncoder,
            'MultiTaskRegressor': MultiTaskRegressor,
           'AstroEncoderDecoder': AstroEncoderDecoder, 'CNNEncoderDecoder': CNNEncoderDecoder,}

DATASETS = {'LightSpec': LightSpecDataset, 'FineTune': FineTuneDataset, 'Simulation': DualDataset, 'Kepler':
                KeplerDataset, 'Spectra': SpectraDataset}

SIMULATION_DATA_DIR = '/data/simulations/dataset_big/lc'
SIMULATION_LABELS_PATH = '/data/simulations/dataset_big/simulation_properties.csv'


def get_kepler_data(data_args, df, transforms):

    return KeplerDataset(df=df, transforms=transforms,
                                target_transforms=transforms,
                                npy_path = '/data/lightPred/data/raw_npy',
                                seq_len=int(data_args.max_len_lc),
                                masked_transforms = data_args.masked_transform,
                                use_acf=data_args.use_acf,
                                use_fft=data_args.use_fft,
                                scale_flux=data_args.scale_flux,
                                labels=data_args.prediction_labels_lc,
                                dims=data_args.in_channels,
                                )
def get_lamost_data(data_args, df, transforms):

    return SpectraDataset(data_args.data_dir, transforms=transforms, df=df, 
                                 max_len=int(data_args.max_len_spectra))

def get_lightspec_data(data_args, df, light_transforms, spec_transforms):

    return LightSpecDataset(df=df,
                            light_transforms=light_transforms,
                            spec_transforms=spec_transforms,
                            npy_path = '/data/lightPred/data/raw_npy',
                            spec_path = data_args.spectra_dir,
                            light_seq_len=int(data_args.max_len_lc),
                            spec_seq_len=int(data_args.max_len_spectra),
                            use_acf=data_args.use_acf,
                            use_fft=data_args.use_fft,
                            meta_columns=data_args.meta_columns_lightspec,
                            scale_flux=data_args.scale_flux,
                            labels=data_args.prediction_labels_lightspec
                            )

def get_finetune_data(data_args, df, light_transforms, spec_transforms):

    return FineTuneDataset(df=df,
                            light_transforms=light_transforms,
                            spec_transforms=spec_transforms,
                            npy_path = '/data/lightPred/data/raw_npy',
                            spec_path = data_args.spectra_dir,
                            light_seq_len=int(data_args.max_len_lc),
                            spec_seq_len=int(data_args.max_len_spectra),
                            use_acf=data_args.use_acf,
                            use_fft=data_args.use_fft,
                            meta_columns=data_args.meta_columns_finetune,
                            scale_flux=data_args.scale_flux,
                            labels=data_args.prediction_labels_finetune
                            )

def get_simulation_data(data_args, df, light_transforms, spec_transforms):
    return DualDataset(df=df,
                        light_transforms=light_transforms,
                        spec_transforms=spec_transforms,
                        npy_path = '/data/lightPred/data/raw_npy',
                        spec_path = data_args.spectra_dir,
                        light_seq_len=int(data_args.max_len_lc),
                        spec_seq_len=int(data_args.max_len_spectra),
                        use_acf=data_args.use_acf,
                        use_fft=data_args.use_fft,
                        meta_columns=data_args.meta_columns_simulation,
                        scale_flux=data_args.scale_flux,
                        labels=data_args.prediction_labels_simulation
                        )


def get_data(data_args, data_generation_fn, dataset_name='FineTune', config=None):


    light_transforms = Compose([ RandomCrop(int(data_args.max_len_lc)),
                            MovingAvg(13),
                            ACF(max_lag_day=None, max_len=int(data_args.max_len_lc)),
                            FFT(seq_len=int(data_args.max_len_lc)),
                            Normalize(['mag_median', 'std']),
                            ToTensor(), ])
    spec_transforms = Compose([LAMOSTSpectrumPreprocessor(continuum_norm=data_args.continuum_norm, plot_steps=False),
                                ToTensor()
                            ])
    if dataset_name == 'Kepler':
        train_df, val_df = data_generation_fn(meta_columns=data_args.meta_columns_lc)
        val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=42)
        spec_transforms = None
        train_dataset = get_kepler_data(data_args, train_df, light_transforms)
        val_dataset = get_kepler_data(data_args, val_df, light_transforms)
        test_dataset = get_kepler_data(data_args, test_df, light_transforms)
    elif dataset_name == 'Spectra':
        train_df, val_df = data_generation_fn(meta_columns=data_args.meta_columns_spec)
        val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=42)
        spec_transforms = Compose([LAMOSTSpectrumPreprocessor(continuum_norm=data_args.continuum_norm, plot_steps=False),
                                ToTensor()
                            ])
        light_transforms = None
        train_dataset = get_lamost_data(data_args, train_df, spec_transforms)
        val_dataset = get_lamost_data(data_args, val_df, spec_transforms)
        test_dataset = get_lamost_data(data_args, test_df, spec_transforms)

    elif dataset_name == 'LightSpec':
        train_df, val_df = data_generation_fn(meta_columns=data_args.meta_columns_lightspec)
        val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=42)
        train_dataset = get_lightspec_data(data_args, train_df, light_transforms, spec_transforms)
        val_dataset = get_lightspec_data(data_args, val_df, light_transforms, spec_transforms)
        test_dataset = get_lightspec_data(data_args, test_df, light_transforms, spec_transforms)
    
    elif dataset_name == 'FineTune':
        train_df, val_df = data_generation_fn(meta_columns=data_args.meta_columns_finetune)
        val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=42)
        train_dataset = get_finetune_data(data_args, train_df, light_transforms, spec_transforms)
        val_dataset = get_finetune_data(data_args, val_df, light_transforms, spec_transforms)
        test_dataset = get_finetune_data(data_args, test_df, light_transforms, spec_transforms)
    
    elif dataset_name == 'Simulation':
        spec_transforms = Compose([LAMOSTSpectrumPreprocessor(rv_norm=False, continuum_norm=data_args.continuum_norm, plot_steps=False),
                                ToTensor()
                            ])
        train_df, val_df = data_generation_fn(meta_columns=data_args.meta_columns_simulation)
        val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=42)
        train_dataset = get_simulation_data(data_args, train_df, light_transforms, spec_transforms)
        val_dataset = get_simulation_data(data_args, val_df, light_transforms, spec_transforms)
        test_dataset = get_simulation_data(data_args, test_df, light_transforms, spec_transforms)

    else: 
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")

    if config is None:
        config = {}
    config.update({
        "light_transforms": str(light_transforms),
        "spec_transforms": str(spec_transforms),
        "train_dataset": str(train_dataset),
        "val_dataset": str(val_dataset),
        "test_dataset": str(test_dataset)
    }
    )
    return train_dataset, val_dataset, test_dataset, config

def get_model(data_args,
            args_dir,
            config,
             local_rank,
             freeze=False):
    
    light_model_name = data_args.light_model_name
    spec_model_name = data_args.spec_model_name
    light_model_args = Container(**yaml.safe_load(open(args_dir, 'r'))[f'{light_model_name}_lc'])
    spec_model_args = Container(**yaml.safe_load(open(args_dir, 'r'))[f'{spec_model_name}_spec'])
    conformer_args_spec = Container(**yaml.safe_load(open(args_dir, 'r'))['Conformer_spec'])
    astroconformer_args_lc = Container(**yaml.safe_load(open(args_dir, 'r'))['AstroConformer_lc'])
    cnn_args_lc = Container(**yaml.safe_load(open(args_dir, 'r'))['CNNEncoder_lc'])
    # lstm_args_lc = Container(**yaml.safe_load(open(args_dir, 'r'))['LSTMEncoder_lc'])
    lightspec_args = Container(**yaml.safe_load(open(args_dir, 'r'))['MultiEncoder_lightspec'])
    transformer_args_lightspec = Container(**yaml.safe_load(open(args_dir, 'r'))['Transformer_lightspec'])
    projector_args = Container(**yaml.safe_load(open(args_dir, 'r'))['projector'])
    predictor_args = Container(**yaml.safe_load(open(args_dir, 'r'))['predictor'])
    moco_pred_args = Container(**yaml.safe_load(open(args_dir, 'r'))['reg_predictor'])
    loss_args = Container(**yaml.safe_load(open(args_dir, 'r'))['loss'])
    moco_args = Container(**yaml.safe_load(open(args_dir, 'r'))['MoCo'])
    optim_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Optimization'])
    tuner_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Tuner'])

    tuner_args.out_dim = len(data_args.prediction_labels_finetune) * len(optim_args.quantiles)
    light_model_args.in_channels = data_args.in_channels
    predictor_args.w_dim = len(data_args.meta_columns_lightspec)
    # lstm_args_lc.seq_len = int(data_args.max_len_lc)
    light_encoder1 = CNNEncoder(cnn_args_lc)
    # light_encoder1 = LSTMEncoder(lstm_args_lc)
    light_encoder2 = Astroconformer(astroconformer_args_lc)
    deepnorm_init(light_encoder2, astroconformer_args_lc)
    light_backbone = DoubleInputRegressor(light_encoder1, light_encoder2, light_model_args)
    light_model = MultiTaskSimSiam(light_backbone, light_model_args)
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

    light_model_state = light_model.state_dict()
    spec_model_state = spec_model.state_dict()

    moco = PredictiveMoco(spec_model, light_model,
                            transformer_args_lightspec,
                            predictor_args.get_dict(),
                            loss_args,
                            **moco_args.get_dict()).to(local_rank)

    model = MultiTaskMoCo(moco, moco_pred_args.get_dict()).to(local_rank)

    if data_args.load_checkpoint:
        datetime_dir = os.path.basename(os.path.dirname(data_args.checkpoint_path))
        exp_num = os.path.basename(data_args.checkpoint_path).split('.')[0].split('_')[-1]
        print(datetime_dir)
        print("loading checkpoint from: ", data_args.checkpoint_path)
        moco = load_checkpoints_ddp(moco, data_args.checkpoint_path)
        print("loaded checkpoint from: ", data_args.checkpoint_path)

    if data_args.approach=='finetune':
        model = MocoTuner(model.moco_model, tuner_args.get_dict(), freeze_moco=freeze).to(local_rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    light_model.load_state_dict(light_model_state)
    spec_model.load_state_dict(spec_model_state)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainble parameters: {num_params}")
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters in model: {all_params}")

    if config is None:
        config = {}
    config.update(
        {
    "data_args": data_args.__dict__,
    "light_model_args": light_model_args.__dict__,
    "light_astroconformer_args": astroconformer_args_lc.__dict__,
    "light_cnn_args": cnn_args_lc.__dict__,
    "spec_model_args": spec_model_args.__dict__,
    "conformer_args_spec": conformer_args_spec.__dict__,
    "transformer_args_lightspec": transformer_args_lightspec.__dict__,
    "moco_args": moco_args.__dict__,
    "predictor_args": predictor_args.__dict__,
    "moco_pred_args": moco_pred_args.__dict__,
    "loss_args": loss_args.__dict__,
    "tuner_args": tuner_args.__dict__,
    "optim_args": optim_args.__dict__,
    "num_params": num_params,
    "model_structure": str(model),
    }
    )
    return model, optim_args, config, light_model, spec_model




