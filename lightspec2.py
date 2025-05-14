
import os
# os.system("pip install astropy statsmodels umap-learn")
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Subset
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import OneCycleLR
from astropy.io import fits, ascii
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
from dataset.dataset import LightSpecDataset, create_unique_loader
from dataset.sampler import DistinctParameterSampler
from nn.astroconf import Astroconformer, AstroEncoderDecoder
from nn.models import * 
from nn.simsiam import MultiModalSimSiam, MultiModalSimCLR
from nn.moco import MultimodalMoCo, PredictiveMoco, MultiTaskMoCo
from nn.simsiam import SimSiam, projection_MLP
from nn.utils import init_model, load_checkpoints_ddp
from util.utils import *
from nn.optim import CQR
from nn.train import ContrastiveTrainer, JEPATrainer, DualFormerTrainer
from tests.test_unique_sampler import run_sampler_tests
from features import create_umap, analyze_eigenspace_projections
import generator

META_COLUMNS = ['KID', 'Teff', 'logg', 'FeH', 'Rstar', 'Mstar', 'Lstar', 'Dist', 'kmag_abs', 'RUWE', 'Prot']

MODELS = {'Astroconformer': Astroconformer, 'CNNEncoder': CNNEncoder, 'MultiEncoder': MultiEncoder, 'MultiTaskRegressor': MultiTaskRegressor,
          'AstroEncoderDecoder': AstroEncoderDecoder, 'CNNEncoderDecoder': CNNEncoderDecoder, 'MultiTaskSimSiam': MultiTaskSimSiam,}

torch.cuda.empty_cache()

torch.manual_seed(1234)
np.random.seed(1234)


current_date = datetime.date.today().strftime("%Y-%m-%d")
datetime_dir = f"lightspec_{current_date}"

def predict_and_save_dl(dl, local_rank, name, trainer):
    rank = torch.distributed.get_rank()
    targets, kids, eigenvalues, eigenvectors, projections, \
    umap_comb, umap_proj = trainer.predict(dl, local_rank, load_best=False)

    df = pd.DataFrame({'kid': kids, 'umap_comb_X': umap_comb[:, 0], 'umap_comb_Y': umap_comb[:, 1],
                       'umap_proj_X': umap_proj[:, 0], 'umap_proj_Y': umap_proj[:, 1]})
    # top_k = top_idx_comb.shape[1]
    # for i in range(1, top_k + 1):
    #     df[f'top_idx_comb_{i}'] = top_idx_comb[:, i - 1]
    #     df[f'top_values_comb_{i}'] = top_values_comb[:, i - 1]
    
    # Save rank-specific CSV file for data predictions
    df.to_csv(f"{data_args.log_dir}/{datetime_dir}/preds_{exp_num}_{name}_{rank}.csv", index=False)
    
    # Only save eigenvalues and eigenvectors once (from rank 0)
    if rank == 0:
        np.save(f"{data_args.log_dir}/{datetime_dir}/eigenvalues_{exp_num}_{name}.npy", eigenvalues)
        np.save(f"{data_args.log_dir}/{datetime_dir}/eigenvectors_{exp_num}_{name}.npy", eigenvectors)
        np.save(f"{data_args.log_dir}/{datetime_dir}/projections_{exp_num}_{name}.npy", projections)
    
    print("results saved in", f"{data_args.log_dir}/{datetime_dir}/preds_{exp_num}_{name}_{rank}.csv")
# Function to aggregate results after all GPUs are done
def aggregate_results(name, num_gpus):
    all_dfs = []
    for rank in range(num_gpus):
        df_path = f"{data_args.log_dir}/{datetime_dir}/preds_{exp_num}_{name}_{rank}.csv"
        if os.path.exists(df_path):
            df = pd.read_csv(df_path)
            all_dfs.append(df)
    
    if all_dfs:
        # Concatenate all dataframes
        combined_df = pd.concat(all_dfs, ignore_index=True)
        # Save the combined results
        combined_df.to_csv(f"{data_args.log_dir}/{datetime_dir}/preds_{exp_num}_{name}.csv", index=False)
        print(f"Combined results saved to {data_args.log_dir}/{datetime_dir}/preds_{exp_num}_{name}.csv")

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
    prot_values = pd.DataFrame({'KID': [], 'Prot': [], 'Prot_err': [], 'Prot_ref': []})
    
    # Process dataframes in priority order
    for df in dataframes:
        print(f"Processing dataframe with {len(df)} rows. currently have {len(prot_values)} prot values")
        # Extract just the KID and Prot columns
        current = df[['KID', 'Prot', 'Prot_err', 'Prot_ref']].copy()
        
        # Only add keys that aren't already in our prot_values dataframe
        missing_keys = current[~current['KID'].isin(prot_values['KID'])]
        
        # Concatenate with existing values
        prot_values = pd.concat([prot_values, missing_keys])
    
    # Merge the aggregated Prot values into the result dataframe
    result = result.merge(prot_values, on='KID', how='left')
    
    return result


def create_train_test_dfs(meta_columns, only_main_seq=True):
    kepler_df = get_all_samples_df(num_qs=None, read_from_csv=True)
    kepler_meta = pd.read_csv('/data/lightPred/tables/berger_catalog_full.csv')
    kmag_df = pd.read_csv('/data/lightPred/tables/kepler_dr25_meta_data.csv')
    kepler_df = kepler_df.merge(kepler_meta, on='KID', how='left').merge(kmag_df[['KID', 'KMAG']], on='KID', how='left')
    kepler_df['kmag_abs'] = kepler_df['KMAG'] - 5 * np.log10(kepler_df['Dist']) + 5
    lightpred_df = pd.read_csv('/data/lightPred/tables/kepler_predictions_clean_seg_0_1_2_median.csv')
    lightpred_df['Prot_ref'] = 'lightpred'
    lightpred_df.rename(columns={'predicted period': 'Prot'}, inplace=True)
    lightpred_df['Prot_err'] = lightpred_df['observational error'] / lightpred_df['mean_period_confidence']


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
    reinhold_df['Prot_err'] = reinhold_df['Prot'] * 0.15


    p_dfs = [lightpred_df, santos_df, mcq14_df, reinhold_df]
    kepler_df = priority_merge_prot(p_dfs, kepler_df)

    kinematic_df = ascii.read('/data/lightPred/tables/kinematic_v.txt').to_pandas()
    kinematic_df['sigma'] = np.sqrt((kinematic_df['U'] ** 2 + kinematic_df['V'] ** 2 + kinematic_df['W'] ** 2) / 3)
    print("len kinematic df: ", len(kinematic_df))

    lamost_kepler_df = pd.read_csv('/data/lamost/lamost_dr8_gaia_dr3_kepler_ids.csv')
    lamost_kepler_df = lamost_kepler_df[~lamost_kepler_df['KID'].isna()]
    lamost_kepler_df['KID'] = lamost_kepler_df['KID'].astype(int)
    lamost_kepler_df = lamost_kepler_df.merge(kepler_df[META_COLUMNS], on='KID', how='inner')
    lamost_kepler_df = lamost_kepler_df.merge(kinematic_df[['KIC', 'sigma']], right_on='KIC', left_on='KID', how='left')
    lamost_kepler_df['main_seq'] = lamost_kepler_df.apply(giant_cond, axis=1)
    if only_main_seq:
        lamost_kepler_df = lamost_kepler_df[lamost_kepler_df['main_seq']==True]
    # else:
    #     lamost_kepler_df = lamost_kepler_df[lamost_kepler_df['main_seq']==False]
    # lamost_kepler_df = lamost_kepler_df.dropna(subset=norm_cols)
    train_df, val_df  = train_test_split(lamost_kepler_df, test_size=0.2, random_state=42)
    print("number of samples kepler: ", len(kepler_df),
      " lamost-kepler :", len(lamost_kepler_df), 'unique kepler ', len(lamost_kepler_df.drop_duplicates('KID')))
    print('number of unique sigmas', len(lamost_kepler_df.drop_duplicates('KID')[~lamost_kepler_df['sigma'].isna()]))
    print(lamost_kepler_df['sigma'].notna().sum())
    print(lamost_kepler_df['sigma'].isna().sum())
    print("sample sigmas: ", lamost_kepler_df['sigma'].sample(10).values)
    return train_df, val_df 

def test_dataset_samples(ds, num_iters=10):
    start = time.time()
    no_founds = 0
    start_time = time.time()
    for i in range(num_iters):
        light, spec, y,light2, spec2,info = ds[i]
        # print(light.shape, spec.shape, light2.shape, spec2.shape, y)
        # if light.sum() == 0:
        #     no_founds += 1
        # # print(light.shape, spec.shape, w.shape, info.keys())
        if i % 10 == 0:
            fig, axes = plt.subplots(1,2, figsize=(24,14))
            axes[0].plot(light[0].cpu().numpy())
            axes[1].plot(spec.cpu().numpy())
            axes[0].set_title(f"Lightcurve: {info['KID']}")
            axes[1].set_title(f"Spectrum: {info['obsid']}")
            plt.savefig(f'/data/lightSpec/images/lightspec_giants_{i}.png')
            plt.close()
    print(f"Time taken for {num_iters} iterations: {time.time() - start_time:.4f} seconds." \
        f"avg per iteration: {(time.time() - start_time)/num_iters:.6f} seconds")


if __name__ == "__main__":
    local_rank, world_size, gpus_per_node = setup()
    args_dir = '/data/lightSpec/nn/full_config_2.yaml'
    data_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Data'])
    exp_num = data_args.exp_num
    light_model_name = data_args.light_model_name
    spec_model_name = data_args.spec_model_name
    combined_model_name = data_args.combined_model_name
    if data_args.test_run:
        datetime_dir = f"test_{current_date}"

    os.makedirs(f"{data_args.log_dir}/{datetime_dir}", exist_ok=True)


    train_dataset, val_dataset, test_dataset, complete_config = generator.get_data(data_args,
                                                    data_generation_fn=create_train_test_dfs,
                                                    dataset_name='LightSpec')

    test_dataset_samples(train_dataset, num_iters=100)

    # train_dataloader = create_unique_loader(train_dataset,
    #                                       batch_size=int(data_args.batch_size), \
    #                                       num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
    #                                       collate_fn=kepler_collate_fn )

    # val_dataloader = create_unique_loader(val_dataset,
    #                                     batch_size=int(data_args.batch_size),
    #                                     num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
    #                                     collate_fn=kepler_collate_fn,
    #                                      drop_last=True
    #                                     )

    # test_dataloader = DataLoader(test_dataset,
    #                              batch_size=int(data_args.batch_size),
    #                              collate_fn=kepler_collate_fn,
    #                              num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
    #                              drop_last=True
                                #  )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
    train_dataloader = DataLoader(train_dataset,
                                batch_size=int(data_args.batch_size), \
                                num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
                                collate_fn=kepler_collate_fn,
                                sampler=train_sampler
                                )

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=int(data_args.batch_size),
                                collate_fn=kepler_collate_fn,
                                num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
                                sampler=val_sampler
                                )

    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=world_size, rank=local_rank)
    test_dataloader = DataLoader(test_dataset,
                                batch_size=int(data_args.batch_size),
                                collate_fn=kepler_collate_fn,
                                num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
                                sampler=test_sampler,
                                drop_last=True
                                )

    print("len train dataloader ", len(train_dataloader))


    model, optim_args, tuner_args, complete_config, light_model, spec_model = generator.get_model(data_args,
                                                                                    args_dir,
                                                                                    complete_config,
                                                                                    local_rank,
                                                                                    )

    # batch = next(iter(train_dataloader))
    # lc1, spec1, lc2, spec2, y, info = batch

    # A = model.module.dual_former.projection_head
    # eigevals, egivec, topk = analyze_eigenspace_projections(A, batch)

    # print(eigvals.shape, eigvecs.shape, topk.shape)

    # exit()

    # output = model(lc1, spec1)
    # print("output keys: ", output.keys())
    # print(output['lc_pred'].shape, output['spectra_pred'].shape, output['dual_pred'][0].shape, output['dual_pred'][1].shape)
    # exit()

    if data_args.approach == 'ssl':
        loss_fn = None
        num_quantiles = 1
    elif data_args.approach == 'multitask':
        loss_fn = CQR(quantiles=optim_args.quantiles, reduction='none')
        num_quantiles = len(optim_args.quantiles)
    elif data_args.approach == 'dual_former':
        loss_fn = nn.L1Loss(reduction='none')
    else:
        loss_fn = None
        num_quantiles = len(optim_args.quantiles)
    if optim_args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=float(optim_args.max_lr),
                                    momentum=float(optim_args.momentum),
                                    weight_decay=float(optim_args.weight_decay),
                                    nesterov=optim_args.nesterov)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(optim_args.max_lr),
        weight_decay=float(optim_args.weight_decay))
    scaler = GradScaler()

    accumulation_step = 1
    if data_args.approach == 'moco':
        trainer = ContrastiveTrainer(model=model, optimizer=optimizer,
                            criterion=loss_fn, output_dim=len(data_args.prediction_labels_lightspec),
                            scaler=scaler, grad_clip=True,
                        scheduler=None, train_dataloader=train_dataloader, full_input=False, only_lc=False,
                        val_dataloader=val_dataloader, device=local_rank, num_quantiles=len(optim_args.quantiles),
                                exp_num=datetime_dir, log_path=data_args.log_dir, range_update=None,
                                accumulation_step=accumulation_step, max_iter=np.inf, stack_pairs=False, use_w=True,
                            use_pred_coeff=True, pred_coeff_val=data_args.pred_coeff_val,
                            exp_name=f"{exp_num}") 
    
    elif data_args.approach == 'dual_former':
        trainer = DualFormerTrainer(model=model, optimizer=optimizer,
                            criterion=loss_fn, output_dim=len(data_args.prediction_labels_lightspec),
                            scaler=scaler, grad_clip=True, use_y_as_latent=data_args.use_latent,
                        scheduler=None, train_dataloader=train_dataloader, 
                        val_dataloader=val_dataloader, device=local_rank, num_quantiles=len(optim_args.quantiles),
                                exp_num=datetime_dir, log_path=data_args.log_dir, range_update=None,
                                accumulation_step=accumulation_step, max_iter=np.inf, print_every=20,
                                alpha=data_args.alpha,
                            exp_name=f"{exp_num}")
    
    elif data_args.approach == 'jepa':

        trainer = JEPATrainer(
                                model=model, optimizer=optimizer,
                                criterion=loss_fn, output_dim=len(data_args.prediction_labels_lightspec),
                                train_dataloader=train_dataloader,
                            val_dataloader=val_dataloader, device=local_rank, num_quantiles=len(optim_args.quantiles),
                                    exp_num=datetime_dir, log_path=data_args.log_dir, alpha=data_args.alpha,
                                    accumulation_step=accumulation_step, max_iter=np.inf, use_y_as_latent=data_args.use_latent,
                                exp_name=f"{exp_num}"
                                )

    complete_config.update(
        {"trainer": trainer.__dict__,
        "loss_fn": str(loss_fn),
        "optimizer": str(optimizer)}
    )
    
    config_save_path = f"{data_args.log_dir}/{datetime_dir}/{exp_num}_complete_config.yaml"
    with open(config_save_path, "w") as config_file:
        json.dump(complete_config, config_file, indent=2, default=str)

    print(f"Configuration (with model structure) saved at {config_save_path}.")

    fit_res = trainer.fit(num_epochs=data_args.num_epochs, device=local_rank,
                            early_stopping=20, best='loss', conf=True) 
    output_filename = f'{data_args.log_dir}/{datetime_dir}/{exp_num}_fit_res.json'
    with open(output_filename, "w") as f:
        json.dump(fit_res, f, indent=2)
    fig, axes = plot_fit(fit_res, legend=exp_num, train_test_overlay=True)
    plt.savefig(f"{data_args.log_dir}/{datetime_dir}/fit_{exp_num}.png")
    plt.clf()

    # Add before prediction calls
    torch.cuda.empty_cache()
    torch.distributed.barrier()

    # predict_and_save_dl(train_dataloader, local_rank, 'train', trainer)
    # predict_and_save_dl(val_dataloader, local_rank, 'val', trainer)
    predict_and_save_dl(test_dataloader, local_rank, 'test', trainer)

    # Make sure all processes are done before aggregating
    torch.distributed.barrier()

    # Only run aggregation on the main process
    if local_rank == 0:
        world_size = torch.distributed.get_world_size()
        aggregate_results('train', world_size)
        aggregate_results('val', world_size)
        aggregate_results('test', world_size)

    exit()




