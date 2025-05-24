import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
import matplotlib as mlp
from sklearn.neural_network import MLPRegressor

from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import procrustes # Correct import for procrustes

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
mist_path = r"C:\Users\Ilay\projects\kepler\data\binaries\tables\ks_mist_catalog_kepler_1e9_feh_interp_all.csv"
kepler_meta_path = r"C:\Users\Ilay\projects\kepler\data\lightPred\tables\kepler_dr25_meta_data.csv"

units = {'M_G_0': 'mag' ,'BPmRP_0': 'mag', 'Teff' :'K', 'age_gyrointerp_model': 'Myr', 'age_angus23': 'Myr',
         'kmag_abs': 'mag', 'kmag_diff': 'mag', 'predicted period': 'Days',
         'FeH': 'dex', 'i': 'Deg', 'i_err': 'Deg', 'final_age_norm': 'Gyr',
         'age_error_norm': 'Gyr'}


latex_names = {'M_G_0': 'Absolute $M_G$' ,'BPmRP_0': 'de-reddened $(BP-RP)$', 'Teff' :'$T_{eff}$',
               'Lstar': r'$log(\frac{L}{L_\odot})$',
               'Mstar': r'$\frac{M}{M_\odot}$',
               'Rstar':r'$\frac{R}{R_\odot}$',
               'age_gyrointerp_model': '$Age_{gyro}$',
               'age_angus23': '$Age_{iso}$',
               'kmag_abs': 'Absolute $M_K$',
               'kmag_diff': r'$\Delta K_{iso}$',
               'RUWE': 'RUWE',
               'predicted period': '$P_{rot}$',
               'FeH': '$FeH$',
               'flag_CMD_numeric': 'CMD Category',
               'i': 'inclination',
               'i_err': 'inclination error',
               'final_age_norm': 'Age',
               'age_error_norm': 'Age Error',
               'moco': 'MoCo',
               'moco_uniqueLoader': 'MoCo Clean',
               'jepa': 'VicReg',
               'unimodal_light': 'Only LC',
               'unimodal_spec': 'Only Spec',
               'dual_former': 'DualFormer (ours)',}

# --- Helper Function (from your example) ---
def giant_cond(row):
    # This is just a placeholder based on your usage, replace with actual logic if needed
    # Example: Assuming 'logg' column exists and main sequence is logg > 4.0
    if pd.isna(row['logg']):
        return False # Exclude if logg is missing
    return row['logg'] > 4.0

# --- 1. Data Loading and Preparation ---

def get_mag_data(df):
    kepler_meta = pd.read_csv(kepler_meta_path)
    mist_data = pd.read_csv(mist_path)
    mag_cols = [c for c in kepler_meta.columns if ('MAG' in c) or ('COLOR' in c)]
    meta_columns = mag_cols + ['KID', 'EBMINUSV']
    df = df.merge(kepler_meta[meta_columns], on='KID', how='left')
    df = df.merge(mist_data[['KID', 'Kmag_MIST']], on='KID', how='left')
    if 'Dist' not in df.columns:
        berger_cat = pd.read_csv(berger_catalog_path)
        df = df.merge(berger_cat, on='KID', how='left',
                      suffixes=['_old', ''])
    for c in mag_cols:
        df[f'{c}_abs'] = df.apply(lambda x: x[c] - 5 * np.log10(x['Dist']) + 5, axis=1)
    df['kmag_abs'] = df.apply(lambda x: x['KMAG'] - 5 * np.log10(x['Dist']) + 5, axis=1)
    df['kmag_diff'] = df['kmag_abs'] - df['Kmag_MIST']
    return df

def prepare_data(dir_name, file_name, umap_input='projections'):
    """Loads projections, merges catalogs, filters data, and prepares inputs."""

    print(f"--- Preparing data for {file_name} ---")

    # Load base data
    proj_path = os.path.join(dir_name, f'embedding_projections_{file_name}.npy')
    projections = np.load(proj_path)
    print(f"Loaded projections with shape: {projections.shape}")

    features_path = os.path.join(dir_name, f'final_features_{file_name}.npy')
    features = np.load(features_path)
    print(f"Loaded features with shape: {features.shape}")

    # Load main prediction file which should correspond to projections
    preds_path = os.path.join(dir_name, f'preds_{file_name}.csv')
    df = pd.read_csv(preds_path)

    # Load and merge catalogs
    berger = pd.read_csv(berger_catalog_path)
    lightpred = pd.read_csv(lightpred_full_catalog_path)
    godoy = pd.read_csv(godoy_catalog_path)

    df = df.merge(berger, right_on='KID', left_on='kid', how='left')
    df = get_mag_data(df)
    age_cols = [c for c in lightpred.columns if 'age' in c] # Robustly find age columns
    df = df.merge(lightpred[['KID', 'predicted period', 'mean_period_confidence'] + age_cols], right_on='KID', left_on='kid', how='left')
    df = df.merge(godoy, right_on='KIC', left_on='kid', how='left', suffixes=['', '_godoy'])
    df['subgiant'] = (df['flag_CMD'] == 'Subgiant').astype(int)
    df['main_seq'] = df.apply(giant_cond, axis=1)
    codes, uniques = pd.factorize(df['flag_CMD'])  # No na_sentinel parameter here
    df['flag_CMD_numeric'] = codes
    mapping_dict = {i: label for i, label in enumerate(uniques)}
    mapping_dict[-1] = np.nan

    # --- Generate Standard UMAP Coordinates (U_standard) ---
    print("Calculating standard UMAP...")
    reducer_standard = umap.UMAP(n_components=2, random_state=42) # Added random_state for reproducibility
    if umap_input == 'projections':
        U_standard = reducer_standard.fit_transform(projections)
    elif umap_input == 'features':
        U_standard = reducer_standard.fit_transform(features)
    else:
        raise ValueError(f'Invalid umap_input: {umap_input}')
    print(f"Calculated U_standard with shape: {U_standard.shape}")
    return projections, U_standard, df, mapping_dict


def create_hr_coords(df, projections, diagram_coords):
    # Extract and clean HR coordinates
    hr_coords = df[diagram_coords].copy()
    hr_coords = hr_coords.replace([np.inf, -np.inf], np.nan)

    # Find valid (non-NaN) indices
    valid_mask = hr_coords.notna().all(axis=1)
    initial_count = len(hr_coords)
    print(f"Removed {initial_count - valid_mask.sum()} rows with NaN")

    # Clean both Y_hr and corresponding X (projections)
    Y_hr = hr_coords.loc[valid_mask].values
    X = projections[valid_mask.values]

    print(f"Prepared Y_hr (HR coordinates) with shape: {Y_hr.shape}")
    print(f"Prepared X (projections) with shape: {X.shape}")
    return Y_hr, X

# --- 2. Transformation Methods ---

def apply_linear_regression(X, Y, fraction_scale=0.2, name=''):
    """Applies Linear Regression to predict Y from X."""
    print("\n--- Applying Linear Regression ---")
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    scaler_Y = StandardScaler()
    Y_scaled = scaler_Y.fit_transform(Y)

    x_indices = np.random.choice(len(X_scaled), size=int(len(X_scaled)*fraction_scale), replace=False)
    mask = np.ones(len(X_scaled), dtype=bool)
    mask[x_indices] = False
    if mask.sum() == 0:
        mask = np.ones(len(X_scaled), dtype=bool)
    X_train_scaled = X_scaled[x_indices, :]
    X_test_scaled = X_scaled[mask, :]
    y_train_scaled = Y_scaled[x_indices, :]
    y_test_scaled = Y_scaled[mask, :]

    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train_scaled)
    score = lr.score(X_test_scaled, y_test_scaled)

    Y_pred_scaled = lr.predict(X_scaled)
    # Inverse transform to get predictions in original Temp/LogLum scale
    Y_pred_lr = scaler_Y.inverse_transform(Y_pred_scaled)
    Y_lr = scaler_Y.inverse_transform(Y_scaled)
    acc = np.mean(np.abs(Y_lr - Y_pred_lr) < Y_lr*0.1, axis=0)
    loss = np.abs(Y_lr - Y_pred_lr).mean(axis=-1).mean(axis=-1)
    print("acc: ", acc, "loss: ", loss)

    # fig, axes = plt.subplots(nrows=2, ncols=1)
    # axes[0].hexbin(Y_pred_lr[:, 0], Y_scaled[:, 0], mincnt=1)
    # axes[1].hexbin(Y_pred_lr[:, 1], Y_scaled[:, 1], mincnt=1)
    # fig.suptitle(f"{name} accuracy: {acc[0]:.3f}, {acc[1]:.3f}")
    #
    # plt.show()

    print(f"Generated Linear Regression predictions shape: {Y_pred_lr.shape}, score: {score}")
    return Y_pred_lr, score


def plot_umap(df, umap_coords, cmd_mapping_dict, dir_name, file_name):
    # df.loc[df['kmag_diff'].abs() > 2, 'kmag_diff'] = np.nan
    cols_to_plot = ['Lstar', 'flag_CMD_numeric', 'age_gyrointerp_model', 'RUWE']
    # cols_to_plot = ['Lstar', 'flag_CMD_numeric', 'age_gyrointerp_model', 'RUWE', 'FeH', 'predicted period', 'kmag_diff', 'Teff']
    # df[df['kmag_diff'].abs() > 2] = np.nan
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(40, 24), sharex=True, sharey=True)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        label = latex_names[cols_to_plot[i]]
        unit = f'({units[cols_to_plot[i]]})' if cols_to_plot[i] in units else ''
        if ((cols_to_plot[i] == 'Teff') or (cols_to_plot[i] == 'predicted period') or
                (cols_to_plot[i] == 'RUWE') or (cols_to_plot[i]) == 'Rstar' or (cols_to_plot[i]) == 'Mstar'):
            color = np.log(df[cols_to_plot[i]])
            label = f'log({label})'
        else:
            color = df[cols_to_plot[i]]
        if cols_to_plot[i] == 'flag_CMD_numeric':
            # Define a colormap with distinct colors (excluding -1 for NaN values)
            # Using a tab20 colormap which provides 20 distinct colors
            cmap = plt.cm.get_cmap('hot', len(cmd_mapping_dict))

            # Initialize empty handles list for legend
            legend_handles = []

            # Plot each category with its own color and add to legend
            for value, name in cmd_mapping_dict.items():
                if value != -1:  # Skip NaN values (-1)
                    mask = df['flag_CMD_numeric'] == value
                    color = cmap(value % len(cmd_mapping_dict))  # Cycle through colors if needed
                    scatter = ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                                         c=[color], label=name)
                    legend_handles.append(scatter)

            # Plot NaN values with a distinct style (gray and smaller)
            mask_nan = df['flag_CMD_numeric'] == -1
            if mask_nan.any():
                nan_scatter = ax.scatter(umap_coords[mask_nan, 0], umap_coords[mask_nan, 1],
                                         c='lightgray', alpha=0.5, s=20, label='NaN')
                legend_handles.append(nan_scatter)

            # Add a legend to this subplot
            ax.legend(handles=legend_handles, title="CMD Classification",
                      loc='upper left', fontsize=26, title_fontsize=30)
        else:
            sc = ax.scatter(umap_coords[:, 0], umap_coords[:, 1], c=color, cmap='OrRd')
            cbar = fig.colorbar(sc, orientation='vertical', label=f'{label} {unit}')
            cbar.ax.yaxis.set_tick_params(labelsize=50)  # Set colorbar tick label size
            cbar.set_label(f'{label} {unit}', fontsize=50)  # Set colorbar label size
    fig.supxlabel('UMAP X', fontsize=50)
    fig.supylabel('UMAP Y', fontsize=50)
    # fig.suptitle('UMAP of Eigenspace')
    plt.tight_layout()
    plt.savefig(os.path.join(dir_name, f'umap_eigenspace_{file_name}.png'))
    plt.show()

    # binary_cols = ['flag_Binary_Union', 'flag_RUWE', 'flag_RVvariable', 'flag_NSS', 'flag_EB_Kepler',
    #                'flag_EB_Gaia', 'flag_SB9', 'kmag_diff']
    # fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(40, 24), sharex=True, sharey=True)
    # axes = axes.flatten()
    # for i, ax in enumerate(axes):
    #     if i < 7:
    #         color = df[binary_cols[i]]
    #         label = binary_cols[i]
    #         sc = ax.scatter(umap_coords[:, 0], umap_coords[:, 1], c=color, cmap='OrRd')
    #         cbar = fig.colorbar(sc, orientation='vertical', label=label)
    #         cbar.ax.yaxis.set_tick_params(labelsize=40)  # Set colorbar tick label size
    #         cbar.set_label(label)  # Set colorbar label size
    # fig.supxlabel('UMAP X')
    # fig.supylabel('UMAP Y')
    # fig.suptitle('UMAP of Eigenspace')
    # plt.tight_layout()
    # plt.savefig(os.path.join(dir_name, f'umap_eigenspace_{file_name}_binaries.png'))
    # plt.show()


def plot_transformation(Y_hr, Y_pred_lr, score_lr, hr_coords_df,
                        dir_name, file_name, diagram_coords=['Teff', 'Lstar']):
    """Plots the results of all methods."""
    print("\n--- Plotting Results ---")
    # Create figure without axes first
    fig = plt.figure(figsize=(40, 24))

    # Plotting parameters
    # cmap = 'viridis'
    s = 20  # Scatter point size

    # Create a GridSpec layout with 2x2 grid
    gs = fig.add_gridspec(2, 2)

    # Create individual axes - with no sharing for the top row
    ax_top_left = fig.add_subplot(gs[0, 0])
    ax_top_right = fig.add_subplot(gs[0, 1])

    # Create bottom row axes with shared x and y axes
    # Note: We'll set up shared axes AFTER inversion if needed
    ax_bottom_left = fig.add_subplot(gs[1, 0])
    ax_bottom_right = fig.add_subplot(gs[1, 1])

    # Organize axes in a way that matches the original code's structure
    axes = [[ax_top_left, ax_top_right],
            [ax_bottom_left, ax_bottom_right]]

    # Now proceed with the plotting
    for row_idx in range(2):
        if row_idx == 0:
            for col_idx in range(2):
                ax = axes[row_idx][col_idx]
                sc = ax.hexbin(Y_hr[:, col_idx], Y_pred_lr[:, col_idx], mincnt=1)
                label = latex_names[diagram_coords[col_idx]]
                unit = f'({units[diagram_coords[col_idx]]})' if diagram_coords[col_idx] in units else ''
                ax.set_xlabel(f'True {label} {unit}', fontsize=50)
                ax.set_ylabel(f'Predicted {label} {unit}', fontsize=50)
        else:
            label_0 = latex_names[diagram_coords[0]]
            unit_0 = f'({units[diagram_coords[0]]})' if diagram_coords[0] in units else ''
            label_1 = latex_names[diagram_coords[1]]
            unit_1 = f'({units[diagram_coords[1]]})' if diagram_coords[1] in units else ''  # Fixed bug here

            # a) Original HR Diagram
            ax = axes[row_idx][0]
            sc = ax.scatter(Y_hr[:, 0], Y_hr[:, 1], s=s, c='sandybrown')
            ax.set_xlabel(f'{label_0} {unit_0}', fontsize=50)
            ax.set_ylabel(f'{label_1} {unit_1}', fontsize=50)

            # Linear Regression Prediction
            ax2 = axes[row_idx][1]
            mae = np.abs(Y_hr - Y_pred_lr).mean(axis=-1)
            sc = ax2.scatter(Y_pred_lr[:, 0], Y_pred_lr[:, 1], s=s, c=np.log(mae), cmap='OrRd')
            ax2.set_xlabel(f"Predicted {label_0} {unit_0}", fontsize=50)
            ax2.set_ylabel(f"Predicted {label_1} {unit_1}", fontsize=50)


            # Apply inversions AFTER plotting but BEFORE linking axes
            if diagram_coords[0] == 'Teff':
                ax.invert_xaxis()  # Standard HR diagram convention
                ax2.invert_xaxis()  # Match original HR convention
            if diagram_coords[1] == 'logg':
                ax.invert_yaxis()
                ax2.invert_yaxis()

            # NOW link the axes for sharing limits
            ax2.sharex(ax)
            ax2.sharey(ax)

            # Add colorbar
            cbar = fig.colorbar(sc, ax=ax2)
            cbar.ax.yaxis.set_tick_params()
            cbar.set_label('log(MAE)')

    # Remove tick labels from the right subplot for cleaner appearance
    # Only show y-axis labels on the left plot for bottom row
    axes[1][1].tick_params(labelleft=False)

    # Make sure both plots share the same view limits after inversion
    axes[1][0].set_xlim(axes[1][0].get_xlim())
    axes[1][0].set_ylim(axes[1][0].get_ylim())
    fig.suptitle(f'$R^2$ - {score_lr:.4f}')

    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to prevent title overlap
    save_path = os.path.join(dir_name, f'hr_transformations_{file_name}_{diagram_coords[0]}_{diagram_coords[1]}.png')
    plt.savefig(save_path)
    print(f"Saved comparison plot to {save_path}")
    plt.show()
def transform_eigenspace(df, projections, dir_name, file_name, diagram_coords=['Teff', 'Lstar']):
    # 1. Prepare data
    Y_hr, X = create_hr_coords(df, projections, diagram_coords)
    # 2. Apply transformations
    Y_pred_lr, score = apply_linear_regression(X, Y_hr, name=f'{diagram_coords[0]}, {diagram_coords[1]}')
    # 3. Plot results
    plot_transformation(Y_hr, Y_pred_lr, score, df, dir_name, file_name, diagram_coords)


def combined_projection_analysis(df, dir_name, target_coords=['i', 'i_err']):
    files = os.listdir(dir_name)
    projections = []
    kids = []
    for file in files:  # Changed variable name to avoid shadowing the outer 'file' variable
        print(file)
        if 'projections' in file:
            split = file.split('_')[1]
            projections.append(np.load(os.path.join(dir_name, file)))
            kid_df = pd.read_csv(os.path.join(dir_name, f'{split}_kids.csv'))
            print(len(projections[-1]), len(kid_df))
            kids.append(kid_df)
    projections = np.concatenate(projections)
    kids = pd.concat(kids)

    # Keep track of the original order with a temporary index
    kids = kids.reset_index(drop=True)
    kids['original_index'] = kids.index

    # Remove duplicates from df before merging (if needed)
    # This ensures each KID matches only once
    df_unique = df.drop_duplicates(subset=['KID'])

    # Merge while keeping the original order
    target_df = kids.merge(df_unique, on='KID', how='left')

    # Sort back to original order and drop the temporary index
    target_df = target_df.sort_values('original_index').drop('original_index', axis=1)

    # Verify the counts match
    assert len(target_df) == len(kids), f"Length mismatch: target_df ({len(target_df)}) != kids ({len(kids)})"

    Y_hr, X = create_hr_coords(target_df, projections, target_coords)
    Y_pred_lr, score = apply_linear_regression(X, Y_hr, name=f'{target_coords[0]}, {target_coords[1]}')
    # 3. Plot results
    plot_transformation(Y_hr, Y_pred_lr, score, target_df, dir_name, 'combined_projection', target_coords)
def plot_diagrams(dir_name, file_name, umap_input='projections'):
    projections, U_standard, df, cmd_mapping_dict = prepare_data(
        dir_name, file_name, umap_input=umap_input
    )
    if umap_input == 'projections':
        for coords in [['BPmRP_0', 'M_G_0'], ['Teff', 'Lstar'], ['Rstar', 'FeH']]:
            transform_eigenspace(df, projections, dir_name, file_name, coords)

    plot_umap(df, U_standard, cmd_mapping_dict, dir_name, file_name)

def compare_umaps(dir_name, inputs, property):
    dirs = [f for f in os.listdir(dir_name)]
    fig, axes = plt.subplots(ncols=len(inputs),nrows=1, figsize=(40, 24), sharex=True, sharey=True)
    axes = axes.flatten()
    for i, dir in enumerate(dirs):
        files = os.listdir(os.path.join(dir_name, dir))
        for file in files:
            if file.startswith('preds'):
                file_name = file.removesuffix('.csv').replace('preds_', '')
                projections, umap_coords, df, cmd_mapping_dict = prepare_data(
                    os.path.join(dir_name, dir), file_name, umap_input=inputs[i]
                )
                ax = axes[i]
                label = latex_names[property]
                unit = f'({units[property]})' if property in units else ''
                if ((property == 'Teff') or (property == 'predicted period') or
                        (property == 'RUWE') or (property) == 'Rstar' or (property) == 'Mstar'):
                    color = np.log(df[property])
                    label = f'log({label})'
                else:
                    color = df[property]
                if property == 'flag_CMD_numeric':
                    # Define a colormap with distinct colors (excluding -1 for NaN values)
                    # Using a tab20 colormap which provides 20 distinct colors
                    cmap = plt.cm.get_cmap('hot', len(cmd_mapping_dict))

                    # Initialize empty handles list for legend
                    legend_handles = []

                    # Plot each category with its own color and add to legend
                    for value, name in cmd_mapping_dict.items():
                        if value != -1:  # Skip NaN values (-1)
                            mask = df['flag_CMD_numeric'] == value
                            color = cmap(value % len(cmd_mapping_dict))  # Cycle through colors if needed
                            scatter = ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                                                 c=[color], label=name)
                            legend_handles.append(scatter)

                    # Plot NaN values with a distinct style (gray and smaller)
                    mask_nan = df['flag_CMD_numeric'] == -1
                    if mask_nan.any():
                        nan_scatter = ax.scatter(umap_coords[mask_nan, 0], umap_coords[mask_nan, 1],
                                                 c='lightgray', alpha=0.5, s=20, label='NaN')
                        legend_handles.append(nan_scatter)

                    # Add a legend to this subplot
                    if i == len(inputs) - 1:
                        ax.legend(handles=legend_handles, title="CMD Classification",
                                  loc='upper right', fontsize=24, title_fontsize=30)
                else:
                    sc = ax.scatter(umap_coords[:, 0], umap_coords[:, 1], c=color, cmap='OrRd')
                    cbar = fig.colorbar(sc, orientation='vertical', label=f'{label} {unit}')
                    cbar.ax.yaxis.set_tick_params(labelsize=18)  # Set colorbar tick label size
                    # cbar.set_label(f'{label} {unit}')  # Set colorbar label size
                ax.set_title(dir)
    fig.supxlabel('UMAP X')
    fig.supylabel('UMAP Y')
    fig.tight_layout()
    save_path = os.path.join(os.path.dirname(dir_name), f'compare_umaps_{property}.png')
    plt.savefig(save_path)
    print(f"Saved comparison plot to {save_path}")
    plt.show()


