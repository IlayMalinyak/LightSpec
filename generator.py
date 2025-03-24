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
from nn.utils import init_model, load_checkpoints_ddp
from util.utils import *
from nn.train import *
from tests.test_unique_sampler import run_sampler_tests
from features import multimodal_umap, create_umap

MODELS = {'Astroconformer': Astroconformer, 'CNNEncoder': CNNEncoder, 'MultiEncoder': MultiEncoder,
            'MultiTaskRegressor': MultiTaskRegressor,
           'AstroEncoderDecoder': AstroEncoderDecoder, 'CNNEncoderDecoder': CNNEncoderDecoder,}

DATASETS = {'LightSpec': LightSpecDataset, 'FineTune': FineTuneDataset, 'Simulation': DualDataset, 'Kepler':
                KeplerDataset}

SIMULATION_DATA_DIR = '/data/simulations/dataset_big/lc'
SIMULATION_LABELS_PATH = '/data/simulations/dataset_big/simulation_properties.csv'



def get_data(data_args, data_generation_fn, dataset_name='FineTune', config=None):
    light_transforms = Compose([ RandomCrop(int(data_args.max_len_lc)),
                            MovingAvg(13),
                            ACF(max_lag_day=None, max_len=int(data_args.max_len_lc)),
                            FFT(seq_len=int(data_args.max_len_lc)),
                            Normalize(['mag_median', 'std']),
                            ToTensor(), ])
                            
    if dataset_name == 'Simulation':

        spec_transforms = Compose([LAMOSTSpectrumPreprocessor(rv_norm=False, continuum_norm=data_args.continuum_norm, plot_steps=False),
                            ToTensor()
                           ])
        full_dataset = DualDataset(data_dir=SIMULATION_DATA_DIR,
                            labels_names=data_args.prediction_labels_simulation,
                            labels_path=SIMULATION_LABELS_PATH,
                            lc_transforms=light_transforms,
                            spectra_transforms=spec_transforms,
                            lc_seq_len=int(data_args.max_len_lc),
                            spec_seq_len=int(data_args.max_len_spectra),
                            use_acf=data_args.use_acf,
                            use_fft=data_args.use_fft,
                            )
        indices = list(range(len(full_dataset)))
        train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
        val_indices, test_indices = train_test_split(val_indices, test_size=0.5, random_state=42)
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(train_dataset, val_indices)
        test_dataset = Subset(train_dataset, test_indices)

    else:
        spec_transforms = Compose([LAMOSTSpectrumPreprocessor(continuum_norm=data_args.continuum_norm, plot_steps=False),
                                    ToTensor()
                                ])
        train_df, val_df = data_generation_fn(meta_columns=data_args.meta_columns)
        val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=42)
        train_dataset = DATASETS[dataset_name](df=train_df, light_transforms=light_transforms,
                                        spec_transforms=spec_transforms,
                                        npy_path = '/data/lightPred/data/raw_npy',
                                        spec_path = data_args.spectra_dir,
                                        light_seq_len=int(data_args.max_len_lc),
                                        spec_seq_len=int(data_args.max_len_spectra),
                                        use_acf=data_args.use_acf,
                                        use_fft=data_args.use_fft,
                                        meta_columns=data_args.meta_columns,
                                        scale_flux=data_args.scale_flux,
                                        labels=data_args.prediction_labels
                                        )

        val_dataset =DATASETS[dataset_name](df=val_df, light_transforms=light_transforms,
                                        spec_transforms=spec_transforms,
                                        npy_path = '/data/lightPred/data/raw_npy',
                                        spec_path = data_args.spectra_dir,
                                        light_seq_len=int(data_args.max_len_lc),
                                        spec_seq_len=int(data_args.max_len_spectra),
                                        use_acf=data_args.use_acf,
                                        use_fft=data_args.use_fft,
                                        meta_columns=data_args.meta_columns,
                                        scale_flux=data_args.scale_flux,
                                        labels=data_args.prediction_labels
                                        )

        test_dataset =DATASETS[dataset_name](df=test_df, light_transforms=light_transforms,
                                        spec_transforms=spec_transforms,
                                        npy_path = '/data/lightPred/data/npy',
                                        spec_path = data_args.spectra_dir,
                                        light_seq_len=int(data_args.max_len_lc),
                                        spec_seq_len=int(data_args.max_len_spectra),
                                        use_acf=data_args.use_acf,
                                        use_fft=data_args.use_fft,
                                        meta_columns=data_args.meta_columns,
                                        scale_flux=data_args.scale_flux,
                                        labels=data_args.prediction_labels
                                        )
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
             local_rank):

    light_model_name = data_args.light_model_name
    spec_model_name = data_args.spec_model_name
    light_model_args = Container(**yaml.safe_load(open(args_dir, 'r'))[f'{light_model_name}_lc'])
    spec_model_args = Container(**yaml.safe_load(open(args_dir, 'r'))[f'{spec_model_name}_spec'])
    conformer_args_spec = Container(**yaml.safe_load(open(args_dir, 'r'))['Conformer_spec'])
    astroconformer_args_lc = Container(**yaml.safe_load(open(args_dir, 'r'))['AstroConformer_lc'])
    cnn_args_lc = Container(**yaml.safe_load(open(args_dir, 'r'))['CNNEncoder_lc'])
    lightspec_args = Container(**yaml.safe_load(open(args_dir, 'r'))['MultiEncoder_lightspec'])
    transformer_args_lightspec = Container(**yaml.safe_load(open(args_dir, 'r'))['Transformer_lightspec'])
    projector_args = Container(**yaml.safe_load(open(args_dir, 'r'))['projector'])
    predictor_args = Container(**yaml.safe_load(open(args_dir, 'r'))['predictor'])
    moco_pred_args = Container(**yaml.safe_load(open(args_dir, 'r'))['reg_predictor'])
    loss_args = Container(**yaml.safe_load(open(args_dir, 'r'))['loss'])
    moco_args = Container(**yaml.safe_load(open(args_dir, 'r'))['MoCo'])
    optim_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Optimization'])
    tuner_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Tuner'])
    tuner_args.out_dim = len(data_args.prediction_labels) * len(optim_args.quantiles)

    light_encoder1 = CNNEncoder(cnn_args_lc)
    light_encoder2 = Astroconformer(astroconformer_args_lc)
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

    moco = PredictiveMoco(spec_model.encoder, light_model.simsiam.backbone,
                            transformer_args_lightspec,
                            predictor_args.get_dict(),
                            loss_args,
                            **moco_args.get_dict()).to(local_rank)

    moco = MultiTaskMoCo(moco, moco_pred_args.get_dict()).to(local_rank)

    if data_args.load_checkpoint:
        datetime_dir = os.path.basename(os.path.dirname(data_args.checkpoint_path))
        exp_num = os.path.basename(data_args.checkpoint_path).split('.')[0].split('_')[-1]
        print(datetime_dir)
        print("loading checkpoint from: ", data_args.checkpoint_path)
        moco = load_checkpoints_ddp(moco, data_args.checkpoint_path)
        print("loaded checkpoint from: ", data_args.checkpoint_path)

    model = MocoTuner(moco.moco_model, tuner_args.get_dict(), freeze_moco=False).to(local_rank)
    model.moco_model.calc_loss = False
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

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
    return model, optim_args, config




