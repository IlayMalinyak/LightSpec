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
    nss_df = pd.read_csv('/data/lightPred/tables/nss_dataset.csv')
    lamost_kepler_df = pd.read_csv('/data/lamost/lamost_dr8_gaia_dr3_kepler_ids.csv')
    lamost_kepler_df = lamost_kepler_df[~lamost_kepler_df['KID'].isna()]
    lamost_kepler_df['KID'] = lamost_kepler_df['KID'].astype(int)
    final_df = lamost_kepler_df.merge(nss_df, on='KID', how='inner')
    train_df, val_df  = train_test_split(final_df, test_size=0.2, random_state=42)
    print("final_df columns: ", final_df.columns)
    print("number of nss: ", final_df[final_df['binary_prob'] == 1].shape[0])
    plt.hist(final_df['binary_prob'], bins=40)
    plt.savefig('/data/lightSpec/images/finetune_nss_dataset.png')
    return train_df, val_df 


current_date = datetime.date.today().strftime("%Y-%m-%d")
datetime_dir = f"inc_finetune_{current_date}"

local_rank, world_size, gpus_per_node = setup()
args_dir = '/data/lightSpec/nn/config_finetune_nss.yaml'
data_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Data'])
exp_num = data_args.exp_num
light_model_name = data_args.light_model_name
spec_model_name = data_args.spec_model_name
combined_model_name = data_args.combined_model_name
if data_args.test_run:
    datetime_dir = f"test_{current_date}"
light_model_args = Container(**yaml.safe_load(open(args_dir, 'r'))[f'{light_model_name}_lc'])
spec_model_args = Container(**yaml.safe_load(open(args_dir, 'r'))[f'{spec_model_name}_spec'])
conformer_args_spec = Container(**yaml.safe_load(open(args_dir, 'r'))['Conformer_spec'])
astroconformer_args_lc = Container(**yaml.safe_load(open(args_dir, 'r'))['AstroConformer_lc'])
cnn_args_lc = Container(**yaml.safe_load(open(args_dir, 'r'))['CNNEncoder_lc'])
lightspec_args = Container(**yaml.safe_load(open(args_dir, 'r'))['MultiEncoder_lightspec'])
transformer_args_lightspec = Container(**yaml.safe_load(open(args_dir, 'r'))['Transformer_lightspec'])
predictor_args = Container(**yaml.safe_load(open(args_dir, 'r'))['predictor'])
loss_args = Container(**yaml.safe_load(open(args_dir, 'r'))['loss'])
moco_args = Container(**yaml.safe_load(open(args_dir, 'r'))['MoCo'])
optim_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Optimization'])
tuner_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Tuner'])
tuner_args.out_dim = len(data_args.prediction_labels) * len(optim_args.quantiles)

os.makedirs(f"{data_args.log_dir}/{datetime_dir}", exist_ok=True)

light_transforms = Compose([ RandomCrop(int(data_args.max_len_lc)),
                        MovingAvg(13),
                        ACF(max_lag_day=None, max_len=int(data_args.max_len_lc)),
                        FFT(seq_len=int(data_args.max_len_lc)),
                        Normalize(['mag_median', 'std']),
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
                                use_fft=data_args.use_fft,
                                labels=data_args.prediction_labels
                                )

test_dataset =FineTuneDataset(df=test_df, light_transforms=light_transforms,
                                spec_transforms=spec_transforms,
                                npy_path = '/data/lightPred/data/npy',
                                spec_path = data_args.spectra_dir,
                                light_seq_len=int(data_args.max_len_lc),
                                spec_seq_len=int(data_args.max_len_spectra),
                                use_acf=data_args.use_acf,
                                use_fft=data_args.use_fft,
                                labels=data_args.prediction_labels
                                )

for i in range(10):
    light, spec, _, _, y, info = train_dataset[i]
    print("train dataset: ", light.shape, spec.shape, y)

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


if data_args.combined_embed:

    combined_backbone = MODELS[combined_model_name](combined_model_args, conformer_args=conformer_args_combined)

    combined_model = SimSiam(combined_backbone)

    combined_model = init_model(combined_model, combined_model_args)

    combined_encoder = combined_model.encoder

    num_params_combined = sum(p.numel() for p in combined_encoder.parameters() if p.requires_grad)
    print(f"Number of trainble parameters in combined encoder: {num_params_combined}")

else:
    combined_encoder = None
# backbone = Transformer(transformer_args_lightspec)

# model = MultimodalMoCo(spec_model.encoder, light_model.encoder, transformer_args_lightspec,  **moco_args.get_dict()).to(local_rank)
# model = MultiModalSimSiam(backbone, spec_model.encoder, light_model.backbone, sims_args).to(local_rank)
moco = PredictiveMoco(spec_model.encoder, light_model.simsiam.encoder,
                         transformer_args_lightspec,
                         predictor_args.get_dict(),
                         loss_args,
                        combined_encoder=combined_encoder,
                        **moco_args.get_dict()).to(local_rank)

moco_model = MultiTaskMoCo(moco, moco_pred_args.get_dict()).to(local_rank)

if data_args.load_checkpoint:
    datetime_dir = os.path.basename(os.path.dirname(data_args.checkpoint_path))
    exp_num = os.path.basename(data_args.checkpoint_path).split('.')[0].split('_')[-1]
    print(datetime_dir)
    print("loading checkpoint from: ", data_args.checkpoint_path)
    moco_model = load_checkpoints_ddp(moco_model, data_args.checkpoint_path)
    print("loaded checkpoint from: ", data_args.checkpoint_path)

model = MocoTuner(moco_model.moco, tuner_args.get_dict()).to(local_rank)
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
    output_dim=len(data_args.prediction_labels),
    num_quantiles=len(optim_args.quantiles),
    log_path=data_args.log_dir,
    exp_num=datetime_dir,
    exp_name=f"inc_finetune_{exp_num}",
)

# Run k-fold cross validation
# k_results = kfold_trainer.run_kfold(num_epochs=100, early_stopping=10)
      

test_dataloader = DataLoader(test_dataset, batch_size=data_args.batch_size, shuffle=False, collate_fn=kepler_collate_fn)
final_results = kfold_trainer.train_final_model_and_test(
    test_dataloader=test_dataloader,
    num_epochs=1000,
    early_stopping=15
)

train_res = final_results['train_results']
with open(f"{data_args.log_dir}/{datetime_dir}/train_results.json", 'w') as f:
    json.dump(train_res, f)
test_res = final_results['test_results']

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


