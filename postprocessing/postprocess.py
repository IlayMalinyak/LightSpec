import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import umap
import seaborn as sns
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from eigenspace_analysis import transform_eigenspace, plot_diagrams

T_MIN =3483.11
T_MAX = 7500.0
LOGG_MIN = -0.22
LOGG_MAX = 4.9
FEH_MIN = -2.5
FEH_MAX = 0.964

lamost_catalog_path = r"C:\Users\Ilay\projects\kepler\data\lightPred\tables\lamost_afgkm_teff_3000_7500_catalog.csv"
lightpred_catalog_path = r"C:\Users\Ilay\projects\kepler\data\lightPred\tables\kepler_predictions_clean_seg_0_1_2_median.csv"
lightpred_full_catalog_path = r"C:\Users\Ilay\projects\kepler\data\binaries\tables\lightpred_cat.csv"
godoy_catalog_path = r"C:\Users\Ilay\projects\kepler\data\binaries\tables\GodoyRivera25_TableA1.csv"
kepler_gaia_nss_path = r"C:\Users\Ilay\projects\kepler\data\binaries\tables\gaia_nss.csv"
kepler_gaia_wide_path = r"C:\Users\Ilay\projects\kepler\data\binaries\tables\gaia_wide.csv"
ebs_path =  r"C:\Users\Ilay\projects\kepler\data\binaries\tables\ebs.csv"
berger_catalog_path = r"C:\Users\Ilay\projects\kepler\data\lightPred\tables\berger_catalog_full.csv"
lamost_kepler_path = r"C:\Users\Ilay\projects\kepler\data\lightPred\tables\lamost_dr8_gaia_dr3_kepler_ids.csv"
lamost_apogee_path = r"C:\Users\Ilay\projects\kepler\data\apogee\crossmatched_catalog_LAMOST.csv"



import torch

def convert_to_onnx(model_path, input_shapes, output_path, input_names=None, output_names=['output']):
    """
    Converts a PyTorch model (.pth) to ONNX format.

    Parameters:
        model_path (str): Path to the .pth file containing the PyTorch model.
        input_shapes (list of tuples): List of shapes for each input tensor.
        output_path (str): Path to save the ONNX model.
        input_names (list of str, optional): Names of the input tensors in ONNX. Defaults to ['input_0', 'input_1', 'input_2', 'input_3'].
        output_names (list of str, optional): Names of the output tensors in ONNX. Defaults to ['output'].
    """

    # Load the PyTorch model
    model = torch.load(model_path, map_location=torch.device('cpu'))
    # model.eval()  # Set to evaluation mode

    # Create example input tensors based on input shapes
    example_inputs = tuple(torch.randn(shape) for shape in input_shapes)

    # Default input names if not provided
    if input_names is None:
        input_names = [f'input_{i}' for i in range(len(input_shapes))]

    # Export the model to ONNX
    torch.onnx.export(model,                             # PyTorch model
                      example_inputs,                    # Tuple of input tensors
                      output_path,                       # Output ONNX file path
                      input_names=input_names,           # Input names in ONNX model
                      output_names=output_names,         # Output names in ONNX model
                      dynamic_axes={name: {0: 'batch_size'} for name in input_names})  # Make batch size dynamic

    print(f"ONNX model saved at {output_path}")


def get_modal_type(parent_dir):
    if 'lightspec' in parent_dir:
        modal_type = 'lightspec'
    elif 'light' in parent_dir:
        modal_type = 'light'
    elif 'spec' in parent_dir:
        modal_type = 'spec'
    elif 'combined' in parent_dir:
        modal_type = 'combined'
    else:
        modal_type = 'none'
    return modal_type

def epoch_avg(values, num_epochs):
    """
    Average values within each epoch.

    Args:
        values (np.array): Array of loss/accuracy values from all iterations
        num_epochs (int): Total number of epochs

    Returns:
        np.array: Averaged values per epoch
    """
    # Calculate number of iterations per epoch
    iters_per_epoch = len(values) // num_epochs

    if iters_per_epoch == 0:
        return values  # Return original values if there are fewer values than epochs

    # Reshape the array to (num_epochs, iters_per_epoch)
    # If there are extra iterations, they are truncated
    values_reshaped = values[:num_epochs * iters_per_epoch].reshape(num_epochs, -1)

    # Calculate mean for each epoch
    return np.mean(values_reshaped, axis=1)


def plot_training_loss(log_path, exp_name='',
                       axes=None, plot_acc=False,
                       scale_type='none', use_epoch_avg=True,
                       plot_aux=False, aux_names=['Vic-Reg', 'Constrastive']):
    """
    Plots training and validation loss from structured JSON log files with improved scaling options
    that preserve comparability between experiments.

    Args:
        log_path (str): Path to directory containing log files
        exp_name (str): Name of experiment for legend
        axes (tuple of matplotlib.axes, optional): Axes to plot on
        plot_acc (bool): Whether to plot accuracy instead of separate losses
        scale_type (str): Scaling method to use:
            - 'shifted_log': log(x + 1) transformation that preserves relative differences
            - 'raw': No scaling
    """
    if axes is None:
        fig, axes = plt.subplots(nrows=1, ncols=2)
    if os.path.isfile(log_path):
        log_files = [log_path]
        exp_dir = log_path.split('/')[-2]
    else:
        log_files = [os.path.join(log_path, f) for f in os.listdir(log_path) if f.endswith('.json')
                 and 'config' not in f]
        exp_dir = log_path.split('/')[-1]
    modal_type = get_modal_type(exp_dir)

    if not len(log_files):
        return axes

    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    aux_train_loss_1 = []
    aux_val_loss_1 = []
    aux_train_loss_2 = []
    aux_val_loss_2 = []

    for log in log_files:
        print(f"Processing file: {log}")
        try:
            with open(log, 'r') as f:
                log_dict = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Skipping file {log} due to JSON error: {e}")
            continue

        for key, target_list in [('train_loss', train_loss),
                                 ('val_loss', val_loss),
                                 ('train_acc', train_acc),
                                 ('val_acc', val_acc),
                                 ('train_aux_loss_1', aux_train_loss_1),
                                 ('val_aux_loss_1', aux_val_loss_1),
                                 ('train_aux_loss_2', aux_train_loss_2),
                                 ('val_aux_loss_2', aux_val_loss_2)]:
            if key in log_dict:
                try:
                    target_list.extend([entry for entry in log_dict[key] if isinstance(entry, (int, float))])
                except Exception as e:
                    print(f"Error processing {key} in {log}: {e}")
                    continue

        if 'aux_loss_1' in log_dict and 'aux_loss_2' in log_dict:
            (aux_train_loss_1, aux_train_loss_2,
             aux_val_loss_1, aux_val_loss_2) = seperate_aux_loss(log, log_dict,
                                                                 train_loss, val_loss,
                                                                 aux_train_loss_1, aux_train_loss_2,
                                                                 aux_val_loss_1, aux_val_loss_2)

    if not train_loss or not val_loss:
        print("No valid data found.")
        return axes

    num_epochs = len(val_acc) if val_acc else len(val_loss)

    # Apply epoch averaging if requested
    if use_epoch_avg:
        train_loss = epoch_avg(np.array(train_loss), num_epochs)
        val_loss = epoch_avg(np.array(val_loss), num_epochs)

        # Apply epoch averaging to auxiliary losses if they exist
        if aux_train_loss_1 and aux_val_loss_1:
            aux_train_loss_1 = epoch_avg(np.array(aux_train_loss_1), num_epochs)
            aux_val_loss_1 = epoch_avg(np.array(aux_val_loss_1), num_epochs)

        if aux_train_loss_2 and aux_val_loss_2:
            aux_train_loss_2 = epoch_avg(np.array(aux_train_loss_2), num_epochs)
            aux_val_loss_2 = epoch_avg(np.array(aux_val_loss_2), num_epochs)

        if plot_acc:
            train_acc = epoch_avg(np.array(train_acc), num_epochs)
            val_acc = epoch_avg(np.array(val_acc), num_epochs)
    else:
        train_loss = np.array(train_loss)
        val_loss = np.array(val_loss)

        # Convert auxiliary losses to numpy arrays
        if aux_train_loss_1 and aux_val_loss_1:
            aux_train_loss_1 = np.array(aux_train_loss_1)
            aux_val_loss_1 = np.array(aux_val_loss_1)

        if aux_train_loss_2 and aux_val_loss_2:
            aux_train_loss_2 = np.array(aux_train_loss_2)
            aux_val_loss_2 = np.array(aux_val_loss_2)

        if plot_acc:
            train_acc = np.array(train_acc)
            val_acc = np.array(val_acc)

    def scale_values(values, method):
        if method == 'shifted_log':
            # Add 1 before taking log to handle small values while preserving relative differences
            return np.log1p(values)
        else:  # 'raw'
            return values

    # Apply the selected scaling
    scaled_train_loss = scale_values(train_loss, scale_type)
    scaled_val_loss = scale_values(val_loss, scale_type)

    # Determine y-axis label based on scaling method
    y_label = 'log(1 + Loss)' if scale_type == 'shifted_log' else 'Loss'

    # Create x-axis values based on whether we're using epoch averaging
    x_train = range(len(scaled_train_loss))
    x_val = range(len(scaled_val_loss))

    if not plot_acc:
        axes[0].plot(x_train, scaled_train_loss, label=exp_name)
        axes[0].set_xlabel('Epoch' if use_epoch_avg else 'Iteration')
        axes[0].set_ylabel(f'Train {y_label}')
        axes[0].legend()

        axes[1].plot(x_val, scaled_val_loss, label=exp_name)
        axes[1].set_xlabel('Epoch' if use_epoch_avg else 'Iteration')
        axes[1].set_ylabel(f'Validation {y_label}')
        axes[1].legend()
        if plot_aux and len(aux_train_loss_1) > 0:
            axes[0].plot(x_train, aux_train_loss_1, label=aux_names[0], linestyle='--')
            axes[0].plot(x_train, aux_train_loss_2, label=aux_names[1], linestyle=':')
            axes[1].plot(x_val, aux_val_loss_1,  label=aux_names[0], linestyle='--')
            axes[1].plot(x_val, aux_val_loss_2, label=aux_names[1], linestyle=':')
            axes[0].legend()
    else:
        axes[0].plot(x_train, scaled_train_loss, label=exp_name + " train")
        axes[0].plot(x_val, scaled_val_loss, label=exp_name + ' validation')
        axes[0].set_xlabel('Epoch' if use_epoch_avg else 'Iteration')
        axes[0].set_ylabel(y_label)
        axes[0].legend()

        x_acc_train = range(len(train_acc))
        x_acc_val = range(len(val_acc))
        axes[1].plot(x_acc_train, train_acc, label=exp_name + " train")
        axes[1].plot(x_acc_val, val_acc, label=exp_name + ' validation')
        axes[1].set_xlabel('Epoch' if use_epoch_avg else 'Iteration')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'figs', modal_type, f'{exp_dir}_loss.png'))
    plt.show()

    return axes, fig


def seperate_aux_loss(log, log_dict,
                      train_loss,
                      val_loss,
                      aux_train_loss_1,
                      aux_train_loss_2,
                      aux_val_loss_1,
                      aux_val_loss_2):

    try:
        # Filter out any non-numeric entries
        aux_loss_1 = [entry for entry in log_dict['aux_loss_1'] if isinstance(entry, (int, float))]
        aux_loss_2 = [entry for entry in log_dict['aux_loss_2'] if isinstance(entry, (int, float))]

        # Calculate how many epochs we have for train and val
        num_train_epochs = len(train_loss)
        num_val_epochs = len(val_loss)

        print(f"Train epochs: {num_train_epochs}, Val epochs: {num_val_epochs}")
        print(f"Aux loss 1 length: {len(aux_loss_1)}, Aux loss 2 length: {len(aux_loss_2)}")

        # Separate auxiliary losses into train and validation
        # Assuming the auxiliary losses alternate: train, val, train, val, ...
        if len(aux_loss_1) >= num_train_epochs + num_val_epochs:
            # Extract train and val auxiliary losses
            aux_train_loss_1.extend(aux_loss_1[::2][:num_train_epochs])  # Even indices (0, 2, 4...) for train
            aux_val_loss_1.extend(aux_loss_1[1::2][:num_val_epochs])  # Odd indices (1, 3, 5...) for val

            aux_train_loss_2.extend(aux_loss_2[::2][:num_train_epochs])  # Even indices for train
            aux_val_loss_2.extend(aux_loss_2[1::2][:num_val_epochs])  # Odd indices for val
        else:
            print("Warning: Auxiliary loss data is shorter than expected")

    except Exception as e:
        print(f"Error processing auxiliary losses in {log}: {e}")

    return aux_train_loss_1, aux_train_loss_2, aux_val_loss_1, aux_val_loss_2


def compare_losses(log_path, exp_names=[], axes=None,
                   scale_type='none', use_epoch_avg=True, plot_acc=True,
                   plot_aux=False, aux_names=[],
                   suffix=''):
    if axes is None:
        fig, axes = plt.subplots(nrows=1, ncols=2)

    log_files = [os.path.join(log_path, f) for f in os.listdir(log_path) if f.endswith('.json') and
                 ('config' not in f)]
    exp_dir = log_path.split('/')[-1]
    modal_type = get_modal_type(exp_dir)

    if not len(log_files):
        return axes

    colors = plt.cm.get_cmap('tab10', len(log_files))


    for i, log in enumerate(log_files):
        color = colors(i)
        train_loss = []
        val_loss = []
        train_acc = []
        val_acc = []

        aux_train_loss_1 = []
        aux_val_loss_1 = []
        aux_train_loss_2 = []
        aux_val_loss_2 = []

        name = exp_names[i] if i < len(exp_names) else 'unknown'
        print(f"Processing file: {log}, exp-name: {name}")
        try:
            with open(log, 'r') as f:
                log_dict = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Skipping file {log} due to JSON error: {e}")
            continue

        for key, target_list in [('train_loss', train_loss),
                                 ('val_loss', val_loss),
                                 ('train_acc', train_acc),
                                 ('val_acc', val_acc),
                                 ('train_aux_loss_1', aux_train_loss_1),
                                 ('val_aux_loss_1', aux_val_loss_1),
                                 ('train_aux_loss_2', aux_train_loss_2),
                                 ('val_aux_loss_2', aux_val_loss_2)]:
            if key in log_dict:
                try:
                    target_list.extend([entry for entry in log_dict[key] if isinstance(entry, (int, float))])
                except Exception as e:
                    print(f"Error processing {key} in {log}: {e}")
                    continue

        if 'epochs' in log_dict:
            num_epochs = len(log_dict['epochs'])
        else:
            num_epochs = len(val_acc) if val_acc else len(val_loss)
        print("Number of epochs: ", num_epochs)
        print('arrays length: ', len(train_loss), len(val_loss), len(train_acc), len(val_acc))
        # Apply epoch averaging if requested
        if use_epoch_avg:
            train_loss = epoch_avg(np.array(train_loss), num_epochs)
            val_loss = epoch_avg(np.array(val_loss), num_epochs)
            aux_train_loss_1 = epoch_avg(np.array(aux_train_loss_1), num_epochs)
            aux_val_loss_1 = epoch_avg(np.array(aux_val_loss_1), num_epochs)
            aux_train_loss_2 = epoch_avg(np.array(aux_train_loss_2), num_epochs)
            aux_val_loss_2 = epoch_avg(np.array(aux_val_loss_2), num_epochs)
        else:
            train_loss = np.array(train_loss)
            val_loss = np.array(val_loss)
            aux_train_loss_1 = np.array(aux_train_loss_1)
            aux_val_loss_1 = np.array(aux_val_loss_1)
            aux_train_loss_2 = np.array(aux_train_loss_2)
            aux_val_loss_2 = np.array(aux_val_loss_2)


        # Determine y-axis label based on scaling method
        y_label = 'Loss'

        # Create x-axis values based on whether we're using epoch averaging
        x_train = range(len(train_loss))
        x_val = range(len(val_loss))
        print("min aux losses: train1 ", np.min(aux_train_loss_1), " val1 ", np.min(aux_val_loss_1),
              "train2 ", np.min(aux_train_loss_2), "val2 ", np.min(aux_val_loss_2))
        print("min losses: train", np.min(train_loss), " val", np.min(val_loss))
        print("max acc: train", np.max(train_acc), " val", np.max(val_acc))
        axes[0].plot(x_train, train_loss, label=name, color=color)
        axes[0].set_xlabel('Epoch' if use_epoch_avg else 'Iteration')
        axes[0].set_ylabel(f'Train {y_label}')
        axes[0].legend()
        if plot_acc:
            axes[0].plot(x_val, val_loss, linestyle='--', color=color)
            axes[1].plot(x_train, train_acc, label=name, color=color)
            axes[1].plot(x_val, val_acc,  linestyle='--', color=color)
            axes[1].set_ylabel(f'avg. Accuracy {y_label}')
        else:
            if plot_aux:
                axes[0].plot(x_train, aux_train_loss_1, linestyle='--', color=color)
                axes[0].plot(x_val, aux_train_loss_2, linestyle='--', color=color)
                axes[1].plot(x_val, aux_val_loss_1, linestyle='--', color=color)
                axes[1].plot(x_val, aux_val_loss_2, linestyle='--', color=color)
            axes[1].plot(x_val, val_loss, label=name, color=color)
            axes[1].set_ylabel(f'Validation {y_label}')
        axes[1].set_xlabel('Epoch' if use_epoch_avg else 'Iteration')
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'figs', modal_type, f'{exp_dir}_loss_comparison{suffix}.png'))
    plt.show()

    return axes, fig

def compare_with_catalog(cat_path, df, modal_type='spec', merge_on='KID', color=None, snr_cut=None):
    catalog = pd.read_csv(cat_path)
    df_merged = pd.merge(df, catalog, on=merge_on, how='left', suffixes=['_orig', ''])
    target_cols = [c for c in df.columns if 'target' in c]
    quantiles = ['0.100', '0.250', '0.500', '0.750', '0.900']
    fraction = 100
    for col in target_cols:
        print(col)
        if 'teff' in col.lower():
            mult_fact = 5778
        elif 'vsini' in col.lower():
            mult_fact = 100
        else:
            mult_fact = 1
        if 'feh' not in col.lower():
            df_col = df[df[col] > 0]
        else:
            df_col = df

        label = col.split('_')[-1]
        target_col = [c for c in catalog.columns if c.lower() == label.lower()][0]

        plot_qunatile_property(df_merged,
                               target_col, label,
                               color,
                               fraction,
                               modal_type, mult_fact,
                               snr_cut, scale_target=False,
                               suffix='_cat_compare')



def unnormalize_df(df):
    teff_cols = [c for c in df.columns if 'teff' in c.lower()]
    feh_cols = [c for c in df.columns if 'feh' in c.lower()]
    logg_cols = [c for c in df.columns if 'logg' in c.lower()]
    df[teff_cols] = df[teff_cols] * (T_MAX - T_MIN) + T_MIN
    df[feh_cols] = df[feh_cols] * (FEH_MAX - FEH_MIN) + FEH_MIN
    df[logg_cols] = df[logg_cols] * (LOGG_MAX - LOGG_MIN) + LOGG_MIN
    return df

def plot_quantile_predictions(df_path, merge_cat_path=None, left_on='obsid',
                              right_on='combined_obsid', target_prefix=None,
                              color=None, snr_cut=None, snr_max=np.inf, restrict_teff=False, suffix=''):
    df = pd.read_csv(df_path)
    df = unnormalize_df(df)
    if 'obsid' in df.columns:
        df['obsid'] = df['obsid'].apply(lambda x: int(x.split('.')[0]))
    if merge_cat_path is not None:
        if merge_cat_path == lamost_catalog_path:
            merge_cat = pd.read_csv(merge_cat_path, sep='|')
        else:
            merge_cat = pd.read_csv(merge_cat_path)
        df = df.merge(merge_cat, left_on=left_on, right_on=right_on)
    if 'snrg' in df.columns:
        df['snrg'] *= 1000
    if 'RUWE' in df.columns:
        df = df[df['RUWE'] < df['RUWE'].mean() + 3 * df['RUWE'].std()]
    orig_len = len(df)
    modal_type = get_modal_type(os.path.dirname(df_path))
    if snr_cut is not None:
        df = df[(df['snrg'] > snr_cut) & (df['snrg'] < snr_max)]
        print(f'fraction samples after snr cut of {snr_cut} :', len(df)/orig_len)

    fraction = len(df)/orig_len * 100

    target_cols = [c for c in df.columns if 'target' in c]
    quantiles = ['0.100', '0.250', '0.500', '0.750', '0.900']

    for col in target_cols:
        if 'teff' in col.lower():
            mult_fact = 1
            # mult_fact = 5778
            df = df[df[col] * mult_fact < 7500]
        elif 'vsini' in col.lower():
            mult_fact = 100
        elif 'predicted period' in col.lower() or 'prot' in col.lower():
            mult_fact = 67
        else:
            mult_fact = 1
        if 'feh' not in col.lower():
            df_col = df[df[col] > 0]
        else:
            df_col = df

        label = col.split('target_')[-1]
        print(label)

        scale_target = True
        if target_prefix is not None:
            col = f"{target_prefix}_{col.split('_')[-1]}"
            if target_prefix == 'APOGEE':
                col = col.upper()
            if 'TEFF' in col:
                scale_target = False
                if restrict_teff:
                    df_col = df_col[df[col] < 7500]
            if col not in df_col.columns:
                print(f'{col} not in df_col')
                continue

        snr_str = f'{snr_cut}-{snr_max}'
        plot_qunatile_property(df_col,
                               col, label,
                               color,
                               fraction,
                               modal_type, mult_fact,
                               snr_str,  scale_target=scale_target, suffix=suffix)


def plot_qunatile_property(df_col, target_col, label,
                           color, fraction, modal_type,
                           mult_fact, snr_cut,
                           scale_target=True, suffix=''):
    print(label, len(df_col))
    df_col.dropna(subset=[target_col], inplace=True)
    targets = df_col[target_col].values
    # if scale_target:
    #     targets = targets * mult_fact

    # Get predictions for all quantiles
    preds_median = df_col[f'pred_{label}_q0.500'].values * mult_fact
    preds_lower = df_col[f'pred_{label}_q0.100'].values * mult_fact
    preds_upper = df_col[f'pred_{label}_q0.900'].values * mult_fact
    # Calculate metrics using median predictions
    avg_err = np.mean(np.abs(targets - preds_median))
    acc_10p = (np.abs(targets - preds_median) < (np.abs(targets) * 0.1)).sum() / len(targets)
    coverage = ((targets >= preds_lower) & (targets <= preds_upper)).sum() / len(targets)
    # Create figure with a specific size
    plt.figure(figsize=(10, 8))
    sort_idx = np.argsort(preds_median)
    preds_median = preds_median[sort_idx]
    preds_lower = preds_lower[sort_idx]
    preds_upper = preds_upper[sort_idx]
    preds_interval = preds_upper - preds_lower
    targets = targets[sort_idx]
    if color is None:
        # For hexbin plot
        plt.hexbin(preds_median, targets, mincnt=1, cmap='viridis')
    else:
        # Check if color column contains strings
        if df_col[color].dtype == object:
            # Group by unique values in color column
            unique_colors = df_col[color].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_colors)))  # Get different colors for each group
            
            for group_val, plot_color in zip(unique_colors, colors):
                group_mask = (df_col[color] == group_val).values
                group_preds = preds_median[group_mask[sort_idx]]
                group_targets = targets[group_mask[sort_idx]]
                
                # Calculate metrics for this group
                group_avg_err = np.mean(np.abs(group_targets - group_preds))
                group_acc_10p = (np.abs(group_targets - group_preds) < (np.abs(group_targets) * 0.1)).sum() / len(group_targets)
                group_coverage = ((group_targets >= preds_lower[group_mask[sort_idx]]) & 
                                (group_targets <= preds_upper[group_mask[sort_idx]])).sum() / len(group_targets)
                
                # Create new figure for each group
                plt.figure(figsize=(10, 8))
                plt.scatter(group_preds, group_targets, 
                          color=plot_color, 
                          alpha=0.5,
                          label=str(group_val))
                
                # Add confidence interval for this group
                plt.fill_between(preds_median[sort_idx][group_mask[sort_idx]],
                               preds_lower[sort_idx][group_mask[sort_idx]],
                               preds_upper[sort_idx][group_mask[sort_idx]],
                               alpha=0.2,
                               color='lightsalmon',
                               label='80% confidence interval')
                
                plt.xlabel(f'Predicted {label}')
                plt.ylabel(f'Target {label}')
                plt.title(f'{label} - {group_val}\nMAE: {group_avg_err:.3f},'
                        f' Accuracy: {group_acc_10p:.3f},'
                        f' Coverage: {group_coverage:.3f},'
                        f' SNR cut: {snr_cut}')
                
                lims = [
                    np.min([plt.xlim()[0], plt.ylim()[0]]),  # min of both axes
                    np.max([plt.xlim()[1], plt.ylim()[1]]),  # max of both axes
                ]
                plt.plot(lims, lims, 'k--', alpha=0.5, zorder=0)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'figs', modal_type, 
                          f'{label}_targets_with_confidence_cut_{snr_cut}_{group_val}{suffix}.png'))
                plt.show()
        else:
            # Create figure with a specific size
            plt.figure(figsize=(10, 8))
            # Original numeric color handling
            color_values = np.log(df_col[color].values)[sort_idx]
            scatter = plt.scatter(preds_median, targets, c=color_values,
                                alpha=0.5)
            plt.colorbar(scatter, label=f'log({color})')
            # Add shaded confidence region
            plt.fill_between(preds_median,
                           preds_lower,
                           preds_upper,
                           alpha=0.2,
                           color='lightsalmon',
                           label='80% confidence interval')
    plt.xlabel(f'Predicted {label}')
    plt.ylabel(f'Target {label}')
    plt.title(f'{label}\nMAE: {avg_err:.3f},'
            f' Accuracy: {acc_10p:.3f},'
            f' Coverage: {coverage:.3f},'
            f' SNR cut: {snr_cut}')
    lims = [
        np.min([plt.xlim()[0], plt.ylim()[0]]),  # min of both axes
        np.max([plt.xlim()[1], plt.ylim()[1]]),  # max of both axes
    ]
    plt.plot(lims, lims, 'k--', alpha=0.5, zorder=0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'figs', modal_type,
              f'{label}_targets_with_confidence_cut_{snr_cut}{suffix}.png'))
    plt.show()

    # plt.figure(figsize=(10, 8))
    # plt.hexbin(targets, preds_interval, mincnt=1, cmap='viridis')
    # plt.xlabel(f'Prediction interval {label}')
    # plt.ylabel(f'Target {label}')
    # plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'figs', modal_type, f'{label}_intervals.png'))
    # plt.show()

def plot_error_vs_prediction_interval(df_path, snr_cut=0):
    df = pd.read_csv(df_path)
    modal_type = get_modal_type(os.path.dirname(df_path))
    target_cols = [c for c in df.columns if 'target' in c]
    df['snrg'] *= 1000
    df = df[df['snrg'] > snr_cut]
    for col in target_cols:
        # Apply same multiplication factors as in plot_quantile_predictions
        if 'teff' in col:
            mult_fact = 5778
        elif 'vsini' in col:
            mult_fact = 100
        else:
            mult_fact = 1
        # Filter data
        if 'feh' not in col.lower():
            df_col = df[df[col] > 0]
        else:
            df_col = df
        label = col.split('_')[-1]
        targets = df_col[col].values * mult_fact
        preds_median = df_col[f'pred_{label}_q0.500'].values * mult_fact
        preds_lower = df_col[f'pred_{label}_q0.100'].values * mult_fact
        preds_upper = df_col[f'pred_{label}_q0.900'].values * mult_fact
        prediction_interval = preds_upper - preds_lower
        error = np.abs(targets - preds_median)

        plt.hexbin(prediction_interval, error, mincnt=1, cmap='viridis')
        plt.xlabel(f'80% Prediction Interval')
        plt.ylabel(f'Absolute Error')
        plt.title(label)
        plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'figs', modal_type, f'{label}_error_vs_interval.png'))
        plt.show()

        plt.hexbin(prediction_interval, np.log(df_col['snrg']), mincnt=1, cmap='viridis')
        plt.xlabel(f'80% Prediction Interval')
        plt.ylabel(f'log(S/N)')
        plt.title(label)
        plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'figs', modal_type, f'{label}_snr_vs_interval.png'))
        plt.show()

def plot_error_vs_snr(df_path, bins=10):
    """
    Plot error rate vs SNR ratio using scatter and box plots

    Parameters:
    df_path (str): Path to the predictions DataFrame
    bins (int): Number of bins for the box plot (default=10)
    """
    # Load and preprocess data similar to plot_quantile_predictions
    lamost_cat = pd.read_csv(lamost_catalog_path, sep='|')
    df = pd.read_csv(df_path)
    df['obsid'] = df['obsid'].apply(lambda x: int(x.split('.')[0]))
    df = df.merge(lamost_cat, left_on='obsid', right_on='combined_obsid')
    df['snrg'] *= 1000
    modal_type = get_modal_type(os.path.dirname(df_path))


    # Get target columns
    target_cols = [c for c in df.columns if 'target' in c]

    for col in target_cols:
        # Apply same multiplication factors as in plot_quantile_predictions
        if 'teff' in col:
            mult_fact = 5778
        elif 'vsini' in col:
            mult_fact = 100
        else:
            mult_fact = 1

        # Filter data
        if 'feh' not in col.lower():
            df_col = df[df[col] > 0]
        else:
            df_col = df

        label = col.split('_')[-1]
        targets = df_col[col].values * mult_fact
        preds_median = df_col[f'pred_{label}_q0.500'].values * mult_fact

        # Calculate absolute errors
        log_abs_errors = np.log10(np.abs(targets - preds_median))
        log_snr = np.log10(df_col['snrg'].values)

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Scatter plot
        scatter = ax1.hexbin(log_snr, log_abs_errors, mincnt=1, cmap='YlGnBu')
        ax1.set_xlabel('log(SNR)')
        ax1.set_ylabel(f'log(Absolute Error) ({label})')
        # ax1.set_title('Error vs SNR (Scatter)')

        # Calculate and plot rolling mean
        sorted_idx = np.argsort(log_snr)
        window = len(log_snr) // 20  # 5% of data points
        rolling_mean = np.convolve(log_abs_errors[sorted_idx],
                                   np.ones(window) / window,
                                   mode='valid')
        ax1.plot(log_snr[sorted_idx][window - 1:],
                 rolling_mean,
                 'r-',
                 label='Rolling mean',
                 linewidth=2)
        ax1.legend()

        # Box plot
        quantiles = np.linspace(0, 1, bins + 1)
        log_snr_bins = np.quantile(log_snr, quantiles)

        # Create bin labels
        bin_labels = [f'{log_snr_bins[i]:.1f}-{log_snr_bins[i + 1]:.1f}'
                      for i in range(len(log_snr_bins) - 1)]

        # Assign each point to a bin
        binned_data = []
        for i in range(len(log_snr_bins) - 1):
            mask = (log_snr >= log_snr_bins[i]) & (log_snr < log_snr_bins[i + 1])
            # For the last bin, include the right edge
            if i == len(log_snr_bins) - 2:
                mask = (log_snr >= log_snr_bins[i]) & (log_snr <= log_snr_bins[i + 1])
            binned_data.append(log_abs_errors[mask])

        # Create box plot without outliers
        bp = ax2.boxplot(binned_data,
                         labels=bin_labels,
                         whis=1.5,
                         showfliers=False)  # This line hides the outliers
        ax2.set_xlabel('log(SNR) bins')
        ax2.set_ylabel(f'log(Absolute Error) ({label})')
        # ax2.set_title('Error vs SNR (Box Plot)')
        ax2.tick_params(axis='x', rotation=45)

        # Add median values on top of each box
        # medians = [np.median(bin_data) for bin_data in binned_data]
        # for i, median in enumerate(medians, 1):
        #     ax2.text(i, median, f'{median:.2f}',
        #              horizontalalignment='center',
        #              verticalalignment='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'figs', modal_type, f'{label}_error_vs_snr.png'))
        plt.show()

        # Print some statistics
        print(f"\nStatistics for {label}:")
        print("Correlation coefficient (log(SNR) vs Error):",
              np.corrcoef(log_snr, log_abs_errors)[0, 1])
        print("Mean error per SNR bin:")
        # for i, bin_label in enumerate(bin_labels):
        #     print(f"SNR bin {bin_label}: {np.mean(binned_data[i]):.3f}")

def plot_umap(log_path, color_cols=['Teff'], merge_cat=None, left_on=None, right_on=None, suffix=''):
    umap_files = [os.path.join(log_path, f) for f in os.listdir(log_path) if f.endswith('.csv') and 'umap' in f]
    exp_dir = log_path.split('/')[-1]
    modal_type = get_modal_type(exp_dir)
    if len(color_cols) > 4:
        fig, axs = plt.subplots(2, len(color_cols)//2, figsize=(30, 24))
    else:
        fig, axs = plt.subplots(1, len(color_cols), figsize=(30, 12))
    # divider = make_axes_locatable(axs)
    # cax = divider.append_axes('right', size='5%', pad=0.05)

    axs = axs.ravel()
    for umap_file in umap_files:
        df = pd.read_csv(umap_file)
        df = df[df['Teff'] <= 7500]
        if ('Teff' in df.columns) and (df['Teff'].max() <= 1):
            df['Teff'] *= 5778
        if ('snrg' in df.columns) and (df['snrg'].max() <= 1):
            df['snrg'] *= 1000
        if merge_cat is not None:
            for cat in merge_cat:
                if cat == lamost_catalog_path:
                    catalog = pd.read_csv(cat, sep='|')
                else:
                    catalog = pd.read_csv(cat)
                df = df.drop(columns=[col for col in df.columns if col.endswith('_umap')], errors='ignore')
                df = df.merge(catalog, left_on=left_on, right_on=right_on, suffixes=['_umap', ''])
        if 'RUWE' in df.columns:
            df['RUWE'] = np.log(df['RUWE'])
        print(f'umap plot of {umap_file} with {len(df)} points')
        for i, color_col in enumerate(color_cols):
            label = color_col if color_col != 'RUWE' else f'log({color_col})'
            print(color_col, label)
            sc = axs[i].scatter(df['umap_x'], df['umap_y'], c=df[color_col], cmap='magma')
            axs[i].set_xlabel('UMAP X', fontsize=20)
            axs[i].set_ylabel('UMAP Y', fontsize=20)
            cbar = fig.colorbar(sc,  orientation='vertical', label=label)
            cbar.ax.yaxis.set_tick_params(labelsize=18)  # Set colorbar tick label size
            cbar.set_label(label, fontsize=20)  # Set colorbar label size
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'figs', modal_type, f'umap{suffix}.png'))
        plt.show()


def postprocess_dir(dir_name, predicition_name,
                    plot_acc=True,
                    plot_aux=False,
                    color_cols=['Teff', 'FeH', 'logg', 'Prot'],
                    exp_name=''):
    df_path = os.path.join(dir_name, predicition_name)
    plot_quantile_predictions(df_path)
    plot_training_loss(dir_name, plot_acc=plot_acc,
                       use_epoch_avg=True, plot_aux=plot_aux,
                       aux_names=['ssl loss', 'reg loss'], exp_name=exp_name)
    if 'compare' in os.listdir(dir_name):
        compare_losses(os.path.join(dir_name, 'compare'),
                       plot_acc=plot_acc,
                       plot_aux=plot_aux, aux_names=['ssl', 'reg'],
                       suffix=exp_name)
    plot_umap(dir_name,
              color_cols=color_cols,
              suffix='')

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



def analyze_projection_distributions(projections, n_bins=50, top_n=10):
    """Analyze how data is distributed across eigenspace"""

    # Take absolute values of projections to measure magnitude
    projection_magnitudes = np.abs(projections)

    # Calculate statistics for each eigenvector
    mean_magnitudes = np.mean(projection_magnitudes, axis=0)
    median_magnitudes = np.median(projection_magnitudes, axis=0)
    std_magnitudes = np.std(projection_magnitudes, axis=0)

    # Plot overall magnitude distribution across eigenvectors
    plt.figure(figsize=(12, 8))

    # Plot 1: Mean magnitude by eigenvector index
    plt.subplot(2, 2, 1)
    plt.bar(range(len(mean_magnitudes)), mean_magnitudes)
    plt.title('Mean Projection Magnitude by Eigenvector')
    plt.xlabel('Eigenvector Index')
    plt.ylabel('Mean Magnitude')
    plt.yscale('log')  # Often helpful to see exponential decay

    # Plot 2: Distribution of all projection magnitudes
    plt.subplot(2, 2, 2)
    plt.hist(projection_magnitudes.flatten(), bins=n_bins, alpha=0.7)
    plt.title('Distribution of All Projection Magnitudes')
    plt.xlabel('Magnitude')
    plt.ylabel('Frequency')

    # Plot 3: Box plots for top eigenvectors
    plt.subplot(2, 2, 3)
    plot_data = [projection_magnitudes[:, i] for i in range(min(top_n, projection_magnitudes.shape[1]))]
    plt.boxplot(plot_data, labels=[f'Eig{i + 1}' for i in range(min(top_n, projection_magnitudes.shape[1]))])
    plt.title(f'Magnitude Distribution for Top {top_n} Eigenvectors')
    plt.ylabel('Magnitude')

    # Plot 4: Histogram of top eigenvectors
    plt.subplot(2, 2, 4)
    for i in range(min(top_n, projection_magnitudes.shape[1])):
        if i < 5:  # Limit to avoid overcrowding
            plt.hist(projection_magnitudes[:, i], bins=30, alpha=0.4,
                     label=f'Eigenvector {i + 1}')
    plt.title(f'Distribution of Top Eigenvector Projections')
    plt.xlabel('Magnitude')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return mean_magnitudes, median_magnitudes, std_magnitudes

def plot_eigenspace(dir_name, file_name):
    berger = pd.read_csv(berger_catalog_path)
    lightpred = pd.read_csv(lightpred_full_catalog_path)
    godoy = pd.read_csv(godoy_catalog_path)
    eigvec_path = f'{dir_name}\eigenvectors_{file_name}.npy'
    eigvecs = np.load(eigvec_path)
    eigvals_path = f'{dir_name}\eigenvalues_{file_name}.npy'
    eigvals = np.abs(np.load(eigvals_path))
    proj_path = f'{dir_name}\projections_{file_name}.npy'
    projections = np.load(proj_path)
    reducer = umap.UMAP(n_components=2)
    reduced_data_comb = reducer.fit_transform(projections)
    # analyze_projection_distributions(projections, n_bins=30)
    df = pd.read_csv(f'{dir_name}\preds_{file_name}.csv')
    df = df.merge(berger, right_on='KID', left_on='kid',  how='left')
    age_cols = [c for c in lightpred.columns if 'age' in c]
    df = df.merge(lightpred[['KID', 'predicted period', 'mean_period_confidence'] + age_cols], right_on='KID', left_on='kid',  how='left')
    df = df.merge(godoy, right_on='KIC', left_on='kid', how='left')
    df['main_seq'] = df.apply(giant_cond, axis=1)
    df = df[df['main_seq'] == True]
    df['subgiant'] = (df['flag_CMD'] == 'Subgiant').astype(int)
    print("ploting eigenspace of ", file_name, ' with ', len(df), ' samples')
    cols_to_plot = ['Teff', 'Rstar', 'Lstar', 'logg',  'subgiant', 'BPmRP_0',
                    'predicted period', 'age_gyrointerp_model', 'age_angus23', 'flag_Binary_Union']
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(42, 24), sharex=True, sharey=True)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        color = df[cols_to_plot[i]] if cols_to_plot[i] != 'Teff' else np.log(df[cols_to_plot[i]])
        label = cols_to_plot[i] if cols_to_plot[i] != 'Teff' else f'log({cols_to_plot[i]})'
        sc = ax.scatter(reduced_data_comb[:, 0], reduced_data_comb[:, 1], c=color)
        cbar = fig.colorbar(sc,  orientation='vertical', label=label)
        cbar.ax.yaxis.set_tick_params(labelsize=18)  # Set colorbar tick label size
        cbar.set_label(label, fontsize=20)  # Set colorbar label size
    fig.supxlabel('UMAP X', fontsize=20)
    fig.supylabel('UMAP Y', fontsize=20)
    fig.suptitle('UMAP of Eigenspace', fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_name, f'umap_eigenspace_{file_name}.png'))
    plt.show()

    # for vec_idx in range(5):
    #     vec = eigvecs[:, vec_idx]
    #     plt.plot(vec)
    # plt.show()
    # x = np.linspace(0, len(df), len(df))
    # plt.scatter(df['umap_comb_X'], df['umap_comb_Y'], c=df['Teff'])
    # plt.show()
    # x = np.linspace(0, len(df), len(df))
    # for val_idx in range(1, 6, 2):
    #     plt.scatter(df[f'top_idx_comb_{val_idx}'], df[f'top_values_comb_{val_idx}'], c=df['Teff'])
    #     plt.title(f'{val_idx}')
    #     plt.show()
    #     fig, axes = plt.subplots(1, 2)
    #     sc0 = axes[0].scatter(df['Teff'], df['logg'], c=df[f'top_values_comb_{val_idx}'])
    #     axes[0].invert_xaxis()
    #     axes[0].invert_yaxis()
    #     cbar = fig.colorbar(sc0, orientation='vertical', label='eigenvalues idx')
    #     cbar.ax.yaxis.set_tick_params(labelsize=18)  # Set colorbar tick label size
    #     sc1 = axes[1].scatter(df['Teff'], df['Lstar'], c=df[f'top_values_comb_{val_idx}'])
    #     axes[1].invert_xaxis()
    #     # plt.gca().invert_yaxis()
    #     cbar = fig.colorbar(sc1, orientation='vertical', label='eigenvalues idx')
    #     cbar.ax.yaxis.set_tick_params(labelsize=18)
    #     fig.suptitle(f'{val_idx}', fontsize=20)
    #     plt.tight_layout()
    #     plt.show()





if __name__ == '__main__':

    root_dir = r'C:\Users\Ilay\projects\multi_modal\LightSpec\logs'
    spec_df_path = os.path.join(root_dir, 'spec', 'spec_decode2_2025-02-11',
                                'MultiTaskRegressor_spectra_decode2_5_cqr.csv')
    lc_df_path = os.path.join(root_dir, 'light', 'light_2025-03-19',
                              'DoubleInputRegressor_lc_decode2_3_1_ssl_cqr.csv')
    lightspec_dir = os.path.join(root_dir, 'lightspec', 'lightspec_2025-04-22')
    files = [f for f in os.listdir(lightspec_dir) if f.endswith('csv')]
    for file in files:
        if '6_latent' in file:
            file_name = file.replace('preds_', '').replace('.csv', '')
            plot_diagrams(lightspec_dir, file_name)
            # plot_eigenspace(lightspec_dir, file_name)
    exit()
    lightspec_best_dir = os.path.join(root_dir, 'lightspec', 'lightspec_2025-03-28',)
    spec_dir = os.path.join(root_dir, 'spec', 'spec_decode2_2025-04-07',)
    postprocess_dir(spec_dir,'MultiTaskRegressor_spectra_decode2_spec_minmax_targets_2_cqr.csv')
    lightspec_df_path = os.path.join(lightspec_best_dir,
                                     'lightspec_ssl_cqr.csv')
    model_path = os.path.join(lightspec_best_dir,
                                     'lightspec_5_mlp_non_freeze.pth')
    output_path = model_path.replace('.pth', '.onnx')
    # convert_to_onnx(model_path,
    #                 [(4,34560), (1,4096), (4,34560), (1,4096)],
    #                 output_path)
    # plot_quantile_predictions(lc_df_path, merge_cat_path=berger_cat, left_on='KID', right_on='KID',
    #                           )
    spec_df = pd.read_csv(spec_df_path)
    lamost_kepler = pd.read_csv(lamost_kepler_path)
    # lamost_kepler['obsid'] = lamost_kepler['obsid'].astype(np.int64)
    # spec_df['obsid'] = spec_df['obsid'].apply(lambda x: x.split('.')[0]).astype(np.int64)
    # spec_kepler = spec_df.merge(lamost_kepler, on='obsid')
    # spec_kepler.rename(columns={'kepid': 'KID'}, inplace=True)
    # compare_with_catalog(berger_cat, spec_kepler)
    # plot_quantile_predictions(spec_df_path)
    # for cut in [None, 5, 10, 20, 50]:
        # snr_max = 20 if (cut is not None and cut < 20) else np.inf
        # snr_max = np.inf
        # plot_quantile_predictions(spec_df_path, merge_cat_path=lamost_apogee_path,
        #                           snr_cut=cut, snr_max=snr_max, target_prefix='APOGEE', suffix='_apogee')
        # plot_quantile_predictions(spec_df_path, merge_cat_path=lamost_apogee_path,
        #                           snr_cut=cut, snr_max=snr_max, target_prefix='APOGEE', restrict_teff=True, suffix='_apogee_restriced')
        # plot_quantile_predictions(spec_df_path, merge_cat_path=lamost_catalog_path,
        #                           snr_cut=cut, snr_max=snr_max, target_prefix=None, suffix='_lasp')
    # plot_error_vs_snr(spec_df_path, bins=20)
    # plot_error_vs_prediction_interval(spec_df_path, snr_cut=0)
    # plot_training_loss(os.path.join(os.path.dirname(__file__), '..', 'logs',
    #                                 'light', 'light_2025-02-28'), plot_acc=True)
    # plot_training_loss(os.path.join(os.path.dirname(__file__), '..', 'logs', 'spec', 'spec_decode2_2025-02-11'), plot_acc=True)
    # plot_training_loss(os.path.join(os.path.dirname(__file__), '..', 'logs', 'light', 'light_2025-02-11'), plot_acc=False)
    # plot_training_loss(os.path.join(os.path.dirname(__file__), '..', 'logs', 'lightspec', 'lightspec_2025-03-02'), plot_acc=True, use_epoch_avg=True)
    plot_training_loss(os.path.join(root_dir, 'lightspec', 'lightspec_2025-03-23'), plot_acc=False,
                       use_epoch_avg=True, plot_aux=True,
                       aux_names=['ssl loss', 'reg loss'], exp_name='_multitask')
    # plot_training_loss(os.path.join(os.path.dirname(__file__), '..', 'logs', 'light', 'light_2025-03-03', 'MultiEncoder_lc_3.json'), plot_acc=False,
    #                    use_epoch_avg=True, plot_aux=True,
    #                    aux_names=['ssl loss', 'reg loss'], exp_name='_multitask')
    # plot_training_loss(os.path.join(os.path.dirname(__file__), '..', 'logs', 'light', 'light_2025-03-12', 'MultiEncoder_lc_1.json'), plot_acc=False,
    #                    use_epoch_avg=True, plot_aux=True,
    #                    aux_names=['ssl loss', 'reg loss'], exp_name='_multitask')
    # plot_training_loss(os.path.join(os.path.dirname(__file__), '..', 'logs', 'lightspec', 'lightspec_2025-03-03'),
    #                    plot_acc=True, plot_aux=True, aux_names=['ssl', 'reg'])
    compare_losses(os.path.join(root_dir, 'lightspec', 'lightspec_2025-03-23', 'compare'),
                   exp_names=['shared projector', 'mlp freeze', 'mlp non freeze'],
                   plot_acc=False,
                   plot_aux=False, aux_names=['ssl', 'reg'],
                   suffix='_multitask_simsiam')
    # compare_losses(os.path.join(os.path.dirname(__file__), '..', 'logs', 'lightspec', 'lightspec_2025-02-24'),
    #                exp_names=['Vic Reg + Contrastive', 'Vic Reg'], suffix='_vic_contrastive')
    # compare_losses(os.path.join(os.path.dirname(__file__), '..', 'logs', 'lightspec', 'lightspec_2025-03-03'),
    #                exp_names=['1', '2'], suffix='_test', plot_acc=True)
    # compare_losses(os.path.join(os.path.dirname(__file__), '..', 'logs', 'light', 'light_2025-03-03'),
    #                exp_names=['logg', 'teff-logg-M-L', 'only ssl'], suffix='_preds', plot_acc=True)
    # plot_umap(os.path.join(os.path.dirname(__file__), '..', 'logs', 'light', 'light_2025-03-03'),
    #           color_cols=['mean_period_confidence', 'predicted period', 'logg'],
    #           merge_cat=[lightpred_catalog_path, berger_catalog_path], left_on='KID', right_on='KID')
    # plot_umap(os.path.join(os.path.dirname(__file__), '..', 'logs', 'light', 'light_2025-03-19'),
    #           color_cols=['Prot', 'mean_period_confidence', 'RUWE'],
    #           merge_cat=[berger_catalog_path, lightpred_catalog_path], left_on='KID', right_on='KID')
    # plot_umap(os.path.join(os.path.dirname(__file__), '..', 'logs', 'light', 'light_2025-02-11'),
    #           color_cols=['mean_period_confidence', 'predicted period', 'RUWE'],
    #           merge_cat=[lightpred_catalog_path, berger_catalog_path], left_on='KID', right_on='KID')
    # plot_umap(os.path.join(os.path.dirname(__file__), '..', 'logs', 'spec', 'spec_decode2_2025-02-11'),
    #           color_cols=['Teff', 'logg', 'FeH'])
    plot_umap(os.path.join(root_dir, 'lightspec', 'lightspec_2025-03-26'),
              color_cols=['Teff', 'logg', 'Rstar', 'RUWE', 'Prot', 'FeH'],
              suffix='')
