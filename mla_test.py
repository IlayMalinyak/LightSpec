import os
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Subset
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import OneCycleLR
from astropy.io import fits
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl
import yaml
import json
from collections import OrderedDict
import warnings
import datetime
from torch.multiprocessing import freeze_support


warnings.filterwarnings("ignore")

import sys
from os import path

ROOT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print("running from ", ROOT_DIR)

from transforms.transforms import *
from dataset.dataset import KeplerDataset
from nn.astroconf import Astroconformer
from nn.models import CNNEncoder, MultiEncoder, CNNEncoderDecoder
from nn.Modules.mla import Transformer as MLATransformer, ModelArgs
# from nn.mamba import MambaEncoder
from nn.simsiam import SimSiam
from nn.train import ContrastiveTrainer, MaskedSSLTrainer
from nn.utils import deepnorm_init, load_checkpoints_ddp
from util.utils import *
from features import create_umap

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
torch.cuda.empty_cache()


def main():
    current_date = datetime.date.today().strftime("%Y-%m-%d")
    datetime_dir = f"light_{current_date}"

    local_rank, world_size, gpus_per_node = setup()
    args_dir = 'nn/mla_test.yaml'
    data_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Data'])
    exp_num = data_args.exp_num
    model_name = 'mla'
    model_args = Container(**yaml.safe_load(open(args_dir, 'r'))['MultiEncoder'])
    conformer_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Conformer'])
    optim_args = Container(**yaml.safe_load(open(args_dir, 'r'))['Optimization SSL'])
    if not os.path.exists(f"{data_args.log_dir}/{datetime_dir}"):
        os.makedirs(f"{data_args.log_dir}/{datetime_dir}")

    transforms = Compose([RandomCrop(int(data_args.max_days_lc / data_args.lc_freq)),
                          #   FillNans(interpolate=True),
                          MovingAvg(13),
                          # RandomMasking(mask_prob=0.2),
                          # RandomTransform([AddGaussianNoise(sigma=0.0001),
                          #                 RandomMasking(mask_prob=0.05),
                          #                 Shuffle(segment_len=270/data_args.lc_freq),
                          #                 Identity()]),
                          # Normalize('std'),
                          ToTensor(),
                          Resample(int(data_args.max_len_lc)),
                          ])

    kepler_df = get_all_samples_df(num_qs=None, read_from_csv=True)
    print("number of samples: ", len(kepler_df))
    train_dataset = KeplerDataset(df=kepler_df, transforms=transforms,
                                  target_transforms=transforms,
                                  npy_path='../../Kepler/data/lightPred/data/npy',
                                  seq_len=int(data_args.max_len_lc),
                                  masked_transforms=data_args.masked_transform
                                  )

    indices = list(range(len(train_dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)

    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_subset, num_replicas=world_size, rank=local_rank)
    train_dataloader = DataLoader(train_subset,
                                  batch_size=int(data_args.batch_size), \
                                  num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
                                  collate_fn=kepler_collate_fn)


    # val_sampler = torch.utils.data.distributed.DistributedSampler(val_subset, num_replicas=world_size, rank=local_rank)
    val_dataloader = DataLoader(val_subset,
                                batch_size=int(data_args.batch_size),
                                collate_fn=kepler_collate_fn,
                                num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]))

    start = time.time()
    for i, batch in enumerate(train_dataloader):
        x1, x2, _, _, info1, info2 = batch
        print(x1.shape, x2.shape)
        # plt.plot(x1[0].squeeze().float())
        # plt.show()
        if i > 10:
            break
    print("average time taken per iteration: ", (time.time() - start) / 10)
    # backbone = Astroconformer(model_args)
    # backbone.pred_layer = torch.nn.Identity()
    # backbone = CNNEncoder(model_args)
    # backbone = MultiEncoder(model_args, conformer_args=conformer_args)
    mla_args = ModelArgs()
    model = MLATransformer(mla_args).to(torch.bfloat16)
    # model = SimSiam(backbone)
    model = model.to(local_rank)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")

    if data_args.create_umap:
        umap_df = create_umap(model.module.backbone, val_dataloader, local_rank, use_w=False, dual=False)
        print("umap created: ", umap_df.shape)
        umap_df.to_csv(f"{data_args.log_dir}/{datetime_dir}/umap_{exp_num}_ssl.csv", index=False)
        print(f"umap saved at {data_args.log_dir}/{datetime_dir}/umap_{exp_num}_ssl.csv")
        exit()

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(optim_args.max_lr),
                                  weight_decay=float(optim_args.weight_decay))
    scaler = GradScaler()
    total_steps = int(data_args.num_epochs) * len(train_dataloader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=total_steps,
                                                           eta_min=float(optim_args.max_lr) / 10)
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
        trainer = MaskedSSLTrainer(model=model, optimizer=optimizer,
                                   criterion=loss_fn, output_dim=1, scaler=scaler,
                                   scheduler=scheduler, train_dataloader=train_dataloader,
                                   val_dataloader=val_dataloader, device=local_rank,
                                   exp_num=datetime_dir, log_path=data_args.log_dir, range_update=None,
                                   accumulation_step=1, max_iter=np.inf,
                                   exp_name=f"{model_name}_lc_{exp_num}")
    else:
        trainer = ContrastiveTrainer(model=model, optimizer=optimizer,
                                     criterion=loss_fn, output_dim=1, scaler=None,
                                     scheduler=scheduler, train_dataloader=train_dataloader,
                                     val_dataloader=val_dataloader, device=local_rank,
                                     exp_num=datetime_dir, log_path=data_args.log_dir, range_update=None,
                                     accumulation_step=1, max_iter=np.inf,
                                     exp_name=f"{model_name}_lc_{exp_num}")

    complete_config = {
        "model_name": model_name,
        "data_args": data_args.__dict__,
        "optim_args": optim_args.__dict__,
        "num_params": num_params,
        "model_structure": str(model),  # Add the model structure to the configuration
        "transforms": str(transforms),
        'trainer': trainer.__dict__
    }
    # config_save_path = f"{data_args.log_dir}/{datetime_dir}/{model_name}_lc_{exp_num}_complete_config.yaml"
    # with open(config_save_path, "w") as config_file:
    #     yaml.dump(complete_config, config_file, default_flow_style=False)
    # print(f"Configuration (with model structure) saved at {config_save_path}.")
    # fig, axes = plot_lr_schedule(scheduler, optim_args.steps_per_epoch, data_args.num_epochs)
    # plt.savefig(f"{data_args.log_dir}/{datetime_dir}/lr_schedule.png")
    trainer.main_process = True
    print("start training...")
    fit_res = trainer.fit(num_epochs=data_args.num_epochs, device=local_rank,
                          early_stopping=40, only_p=False, best='loss', conf=True)
    output_filename = f'{data_args.log_dir}/{datetime_dir}/{model_name}_lc_{exp_num}.json'
    with open(output_filename, "w") as f:
        json.dump(fit_res, f, indent=2)
    fig, axes = plot_fit(fit_res, legend=exp_num, train_test_overlay=True)
    plt.savefig(f"{data_args.log_dir}/{datetime_dir}/fit_{model_name}_lc_{exp_num}.png")
    plt.clf()

if __name__ == '__main__':
    freeze_support()
    main()