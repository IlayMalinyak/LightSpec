import torch
from torch import distributed as dist
from matplotlib import pyplot as plt
import numpy as np
import itertools
import os
import pandas as pd
import re
from typing import List

def setup():
    """
    Setup the distributed training environment.
    """
    world_size    = int(os.environ["WORLD_SIZE"])
    rank          = int(os.environ["SLURM_PROCID"])
    jobid         = int(os.environ["SLURM_JOBID"])
    gpus_per_node = torch.cuda.device_count()
    print('jobid ', jobid)
    print('gpus per node ', gpus_per_node)
    print(f"Hello from rank {rank} of {world_size} where there are" \
          f" {gpus_per_node} allocated GPUs per node. ", flush=True)

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)
    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)
    print(f"rank: {rank}, local_rank: {local_rank}")
    return local_rank, world_size, gpus_per_node

class Container(object):
  '''A container class that can be used to store any attributes.'''
  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)
  
  def load_dict(self, dict):
    for key, value in dict.items():
      if getattr(self, key, None) is None:
        setattr(self, key, value)

  def print_attributes(self):
    for key, value in vars(self).items():
      print(f"{key}: {value}")

def plot_fit(
    fit_res: dict,
    fig: plt.figure = None,
    log_loss: bool = False,
    legend: bool = None,
    train_test_overlay: bool = False,
):
    """
    Plot fit results.

    Args:
        fit_res (dict): The fit results.
        fig (plt.figure, optional): The figure to plot on. Defaults to None.
        log_loss (bool, optional): Whether to plot the loss on a log scale. Defaults to False.
        legend (bool, optional): The legend to use. Defaults to None.
        train_test_overlay (bool, optional): Whether to overlay the train and test results. Defaults to False.

    Returns:
        Tuple[plt.figure, plt.axes]: The figure and axes.
    """
    if fig is None:
        nrows = 1 if train_test_overlay else 2
        ncols = 1 if np.isnan(fit_res['train_acc']).any() else 2
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(8 * ncols, 5 * nrows),
            sharex="col",
            sharey=False,
            squeeze=False,
        )
        axes = axes.reshape(-1)
    else:
        axes = fig.axes

    for ax in axes:
        for line in ax.lines:
            if line.get_label() == legend:
                line.remove()
    if ncols > 1:
        p = itertools.product(enumerate(["train", "val"]), enumerate(["loss", "acc"]))
    else:
        p = itertools.product(enumerate(["train", "val"]), enumerate(["loss"]))
    for (i, traintest), (j, lossacc) in p:
        ax = axes[j if train_test_overlay else i * 2 + j]

        attr = f"{traintest}_{lossacc}"
        data =fit_res[attr]
        label = traintest if train_test_overlay else legend
        h = ax.plot(np.arange(1, len(data) + 1), data, label=label)
        ax.set_title(attr)

        if lossacc == "loss":
            ax.set_xlabel("Iteration #")
            ax.set_ylabel("Loss")
            if log_loss:
                ax.set_yscale("log")
                ax.set_ylabel("Loss (log)")
        else:
            ax.set_xlabel("Epoch #")
            ax.set_ylabel("Accuracy (%)")

        if legend or train_test_overlay:
            ax.legend()
        ax.grid(True)

    return fig, axes


def plot_lr_schedule(scheduler, steps_per_epoch, epochs):
    """
    Plot the learning rate schedule.
    
    Args:
    scheduler: The OneCycleLR scheduler.
    steps_per_epoch: The number of steps (batches) per epoch.
    epochs: The total number of epochs.
    """
    lrs = []
    total_steps = steps_per_epoch * epochs
    for _ in range(total_steps):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.optimizer.step()
        scheduler.step()
    fig, ax = plt.subplots(figsize=(14, 8))
    plt.plot(lrs)
    plt.title("Learning Rate Schedule")
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.xticks(
         rotation=90
    )
    plt.grid(True)
    return fig, ax

def create_kepler_df(kepler_path:str, table_path:str=None):
    """
    Create a DataFrame of Kepler data files.

    Args:
        kepler_path (str): The path to the Kepler data files.
        table_path (str, optional): The path to the table of Kepler data. Defaults to None.
    Returns:
        pd.DataFrame: The DataFrame of Kepler data files.
    """

    data_files_info = []
    for file in os.listdir(kepler_path):
        obj_id = extract_object_id(file)
        if obj_id:
            data_files_info.append({'KID': obj_id, 'data_file_path':os.path.join(kepler_path, file) })
    if len(data_files_info) == 0:
        print("no files found in ", kepler_path)
        return pd.DataFrame({'KID':[], 'data_file_path':'[]'})
    kepler_df = pd.DataFrame(data_files_info)
    kepler_df['KID'] = kepler_df['KID'].astype('int64')
    kepler_df['data_file_path'] = kepler_df['data_file_path'].astype('string')

    if table_path is None:
        return kepler_df
    table_df = pd.read_csv(table_path)
    final_df = table_df.merge(kepler_df, on='KID', how='inner', sort=False)
    return final_df

def kepler_collate_fn(batch:List):
    """
    Collate function for the Kepler dataset.        
    """
    # Separate the elements of each sample tuple (x, y, mask, info) into separate lists
    xs, ys, masks, masks_y, infos, infos_y = zip(*batch)

    # Convert lists to tensors
    xs_tensor = torch.stack(xs, dim=0)
    ys_tensor = torch.stack(ys, dim=0)
    masks_tensor = torch.stack(masks, dim=0)
    masks_y_tensor = torch.stack(masks_y, dim=0)
    return xs_tensor, ys_tensor, masks_tensor, masks_y_tensor, infos, infos_y

def multi_quarter_kepler_df(root_kepler_path:str, Qs:List, table_path:str=None):
    """
    Create a DataFrame of multi-quarter Kepler data files.

    Args:
        root_kepler_path (str): The root path to the Kepler data files.
        Qs (List): The list of quarters to include.
        table_path (str, optional): The path to the table of Kepler data. Defaults to None.
    Returns:
        pd.DataFrame: The DataFrame of multi-quarter Kepler data files.
    """
    
    print("creating multi quarter kepler df with Qs ", Qs, "table path " , table_path)
    dfs = []
    for q in Qs:
        kepler_path = os.path.join(root_kepler_path, f"Q{q}")
        print("kepler path ", kepler_path)
        df = create_kepler_df(kepler_path, table_path)
        print("length of df ", len(df))
        dfs.append(df)
    if 'Prot' in dfs[0].columns:
        if 'Prot_err' in dfs[0].columns:
            merged_df = pd.concat(dfs).groupby('KID').agg({'Prot': 'first', 'Prot_err': 'first', 'Teff': 'first',
            'logg': 'first', 'data_file_path': list}).reset_index()
        else:
            merged_df = pd.concat(dfs).groupby('KID').agg({'Prot': 'first', 'data_file_path': list}).reset_index()
    elif 'i' in dfs[0].columns:
        merged_df = pd.concat(dfs).groupby('KID').agg({'i': 'first', 'data_file_path': list}).reset_index()
    else:
        merged_df = pd.concat(dfs).groupby('KID')['data_file_path'].apply(list).reset_index()
    merged_df['number_of_quarters'] = merged_df['data_file_path'].apply(lambda x: len(x))
    return merged_df

def extract_object_id(file_name:str):
    """
    Extract the object ID from a file name.

    Args:
        file_name (str): The file name.

    Returns:
        str: The object ID.
    """
    match = re.search(r'kplr(\d{9})-\d{13}_llc.fits', file_name)
    return match.group(1) if match else None


def convert_to_list(string_list:str):
    """
    Convert a string representation of a list to a list.

    Args:
        string_list (str): The string representation of the list.

    Returns:
        List: The list.
    """
    # Extract content within square brackets
    matches = re.findall(r'\[(.*?)\]', string_list)
    if matches:
        # Split by comma, remove extra characters except period, hyphen, underscore, and comma, and strip single quotes
        cleaned_list = [re.sub(r'[^A-Za-z0-9\-/_,.]', '', s) for s in matches[0].split(',')]
        return cleaned_list
    else:
        return []

def convert_to_tuple(string:str):
    """
    Convert a string representation of a tuple to a tuple.

    Args:
        string (str): The string representation of the tuple.

    Returns:
        Tuple: The tuple.
    """
    values = string.strip('()').split(',')
    return tuple(int(value) for value in values)

def convert_ints_to_list(string:str):
    """
    Convert a string representation of a list of integers to a list of integers.

    Args:
        string (str): The string representation of the list of integers.

    Returns:
        List: The list of integers.
    """
    values = string.strip('()').split(',')
    return [int(value) for value in values]

def convert_floats_to_list(string:str):
    """
    Convert a string representation of a list of floats to a list of floats.

    Args:
        string (str): The string representation of the list of floats.
    Returns:
        List: The list of floats.
    """
    string = string.replace(' ', ',')
    string = string.replace('[', '')
    string = string.replace(']', '')
    numbers = string.split(',')    
    return [float(num) for num in numbers if len(num)]

def extract_qs(path:str):
    """
    Extract the quarters numbers from a string.

    Args:
        path (str): The string containing the quarter numbers.

    Returns:
        List: The list of quarter numbers.
    """
    qs_numbers = []
    for p in path:
        match = re.search(r'[\\/]Q(\d+)[\\/]', p)
        if match:
            qs_numbers.append(int(match.group(1)))
    return qs_numbers

def consecutive_qs(qs_list:List[int]):
    """
    calculate the length of the longest consecutive sequence of 'qs'
    Args:
        qs_list (List[int]): The list of quarter numbers.
    Returns:
        int: The length of the longest consecutive sequence of 'qs'.
    """

    max_length = 0
    current_length = 1
    for i in range(1, len(qs_list)):
        if qs_list[i] == qs_list[i-1] + 1:
            current_length += 1
        else:
            max_length = max(max_length, current_length)
            current_length = 1
    return max(max_length, current_length)

def find_longest_consecutive_indices(nums:List[int]):
    """
    Find the indices of the longest consecutive sequence of numbers.
    Args:
        nums (List[int]): The list of numbers.
    Returns:
        Tuple[int, int]: The start and end indices of the longest consecutive sequence.
    """
    start, end = 0, 0
    longest_start, longest_end = 0, 0
    max_length = 0

    for i in range(1, len(nums)):
        if nums[i] == nums[i-1] + 1:
            end = i
        else:
            start = i

        if end - start > max_length:
            max_length = end - start
            longest_start = start
            longest_end = end

    return longest_start, longest_end

def get_all_samples_df(num_qs:int=8, read_from_csv:bool=True):
    """
    Get all samples DataFrame.
    Args:
        num_qs (int, optional): The minimum number of quarters. Defaults to 8.
    Returns:
        pd.DataFrame: The DataFrame of all samples.
    """
    if read_from_csv:
        kepler_df = pd.read_csv('/data/lightPred/tables/all_kepler_samples.csv')
    else:
        kepler_df = multi_quarter_kepler_df('/data/lightPred/data/', table_path=None, Qs=np.arange(3,17))
    try:
        kepler_df['data_file_path'] = kepler_df['data_file_path'].apply(convert_to_list)
    except TypeError:
        pass
    kepler_df['qs'] = kepler_df['data_file_path'].apply(extract_qs)  # Extract 'qs' numbers
    kepler_df['num_qs'] = kepler_df['qs'].apply(len)  # Calculate number of quarters
    kepler_df['consecutive_qs'] = kepler_df['qs'].apply(consecutive_qs)  # Calculate length of longest consecutive sequence
    if num_qs is not None:
        # kepler_df = kepler_df[kepler_df['num_qs'] >= num_qs]
        kepler_df = kepler_df[kepler_df['consecutive_qs'] >= num_qs]
        kepler_df['longest_consecutive_qs_indices'] = kepler_df['longest_consecutive_qs_indices'].apply(convert_ints_to_list)
    return kepler_df

