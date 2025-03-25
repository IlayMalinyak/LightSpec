import torch
from torch import distributed as dist
from matplotlib import pyplot as plt
import numpy as np
import itertools
import os
import pandas as pd
import re
import random
from typing import List

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multiple GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

  def get_dict(self):
    return self.__dict__

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
            figsize=(12 * ncols, 8 * nrows),
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
    fig, ax = plt.subplots(figsize=(28, 15))
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
    x1, x2, y, x1_target, x2_target, infos = zip(*batch)

    # Convert lists to tensors
    x1_tensor = torch.stack(x1, dim=0)
    x2_tensor = torch.stack(x2, dim=0)
    y_tensor = torch.stack(y, dim=0)
    x1_target_tensor = torch.stack(x1_target, dim=0)
    x2_target_tensor = torch.stack(x2_target, dim=0)

    return x1_tensor, x2_tensor, y_tensor, x1_target_tensor, x2_target_tensor, infos

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

def giant_cond(x):
    """
    condition for red giants in kepler object.
    the criterion for red giant is given in Ciardi et al. 2011
    :param: x row in dataframe with columns - Teff, logg
    :return: boolean
    """
    logg, teff = x['logg'], x['Teff']
    if teff >= 6000:
        thresh = 3.5
    elif teff <= 4250:
        thresh = 4
    else:
        thresh = 5.2 - (2.8 * 1e-4 * teff)
    return logg >= thresh

def norm_col(df, col):
    """
    Normalize a column in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame.
        col (str): The column to normalize.

    Returns:
        pd.DataFrame: The normalized DataFrame.
    """
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df

def extract_date(p):
    match = re.search(r'\d{4}-\d{2}-\d{2}', p)
    if match:
        date = match.group(0)
        print(f"Extracted date: {date}")
    else:
        print("No date found in the path.")


def save_predictions_to_dataframe(preds, targets, info, prediction_labels, quantiles, id_name='obsid', info_keys=[],
                                save_path=None, verbose=True):
    """
    Save model predictions, including quantile predictions, to a DataFrame and optionally to CSV.
    
    Args:
        preds (np.ndarray): Predictions array of shape (batch_size, n_labels, n_quantiles)
        targets (np.ndarray): Target values array of shape (batch_size, n_labels)
        info (dict): Dictionary containing additional information about the predictions
        prediction_labels (list): List of label names
        quantiles (list): List of quantile values used in prediction
        save_path (str, optional): Path to save CSV file. If None, only returns DataFrame
        verbose (bool): Whether to print shape information
    
    Returns:
        pd.DataFrame: DataFrame containing predictions and targets
    
    Raises:
        ValueError: If input shapes are inconsistent
    """
    # Input validation
    if not isinstance(targets, np.ndarray) or not isinstance(preds, np.ndarray):
        raise ValueError("targets and preds must be numpy arrays")
    
    if targets.shape[0] != preds.shape[0]:
        raise ValueError(f"Batch size mismatch: targets {targets.shape[0]} vs preds {preds.shape[0]}")
    
    if targets.shape[1] != preds.shape[1]:
        raise ValueError(f"Number of labels mismatch: targets {targets.shape[1]} vs preds {preds.shape[1]}")
    
    if preds.shape[2] != len(quantiles):
        raise ValueError(f"Number of quantiles mismatch: preds {preds.shape[2]} vs quantiles {len(quantiles)}")
    
    if verbose:
        print(f"Targets shape: {targets.shape}, Predictions shape: {preds.shape}")
        print(f"Number of samples: {len(info['Teff'])}")

    try:
        # id_name = [k for k in info.keys() if 'obsid' in k.lower()][0]
        # Create dictionary for targets and predictions
        df_dict = {
            # Add target values - use numpy array indexing
            **{f'target_{label}': targets[:, i] for i, label in enumerate(prediction_labels)},
            # Add predictions for each quantile - use numpy array indexing
            **{
                f'pred_{label}_q{quantile:.3f}': preds[:, i, q_idx] 
                for i, label in enumerate(prediction_labels)
                for q_idx, quantile in enumerate(quantiles)
            },
            # Add info dictionary
            # id_name: info[id_name],
            # 'Teff': info['Teff']
            # Add info dictionary
            # **info
        }
            
        # if 'snrg' in info.keys():
        #     df_dict['snrg'] = info['snrg']

    except Exception as e:
        print(f"Falling back to simplified format due to error: {e}")
        # Simplified version with essential columns
        df_dict = {
            # Add target values - use numpy array indexing
            **{f'target_{label}': targets[:, i] for i, label in enumerate(prediction_labels)},
            # Add predictions for each quantile - use numpy array indexing
            **{
                f'pred_{label}_q{quantile:.3f}': preds[:, i, q_idx] 
                for i, label in enumerate(prediction_labels)
                for q_idx, quantile in enumerate(quantiles)
            },
        }


    
    # Create DataFrame
    df = pd.DataFrame(df_dict)

    for key in info_keys:
        if key in info:
            df[key] = info[key]
    
    # Save to CSV if path is provided
    if save_path is not None:
        df.to_csv(save_path, index=False)
        if verbose:
            print(f"Saved predictions to: {save_path}")
    
    return df

