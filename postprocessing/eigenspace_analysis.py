import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
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

def prepare_data(dir_name, file_name):
    """Loads projections, merges catalogs, filters data, and prepares inputs."""

    print(f"--- Preparing data for {file_name} ---")

    # Load base data
    proj_path = os.path.join(dir_name, f'projections_{file_name}.npy')
    projections = np.load(proj_path)
    print(f"Loaded projections with shape: {projections.shape}")

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

    # --- Generate Standard UMAP Coordinates (U_standard) ---
    print("Calculating standard UMAP...")
    reducer_standard = umap.UMAP(n_components=2, random_state=42) # Added random_state for reproducibility
    U_standard = reducer_standard.fit_transform(projections)
    print(f"Calculated U_standard with shape: {U_standard.shape}")

    return projections,  U_standard, df

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

def plot_umap(df, umap_coords, dir_name, file_name):
    df.loc[df['kmag_diff'].abs() > 2, 'kmag_diff'] = np.nan
    cols_to_plot = ['Teff', 'Rstar', 'Lstar', 'logg',  'main_seq', 'subgiant', 'FeH',
                    'predicted period', 'age_gyrointerp_model', 'age_angus23',]
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(42, 24), sharex=True, sharey=True)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if ((cols_to_plot[i] == 'Teff') or (cols_to_plot[i] == 'predicted period') or
                (cols_to_plot[i] == 'RUWE') or (cols_to_plot[i]) == 'Rstar'):
            color = np.log(df[cols_to_plot[i]])
            label = f'log({cols_to_plot[i]})'
        else:
            color = df[cols_to_plot[i]]
            label = cols_to_plot[i]

        sc = ax.scatter(umap_coords[:, 0], umap_coords[:, 1], c=color)
        cbar = fig.colorbar(sc,  orientation='vertical', label=label)
        cbar.ax.yaxis.set_tick_params(labelsize=18)  # Set colorbar tick label size
        cbar.set_label(label, fontsize=20)  # Set colorbar label size
    fig.supxlabel('UMAP X', fontsize=20)
    fig.supylabel('UMAP Y', fontsize=20)
    fig.suptitle('UMAP of Eigenspace', fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_name, f'umap_eigenspace_{file_name}.png'))
    plt.show()

    binary_cols = ['flag_Binary_Union', 'flag_RUWE', 'flag_RVvariable', 'flag_NSS', 'flag_EB_Kepler',
                   'flag_EB_Gaia', 'flag_SB9', 'kmag_diff']
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(42, 24), sharex=True, sharey=True)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < 7:
            color = df[binary_cols[i]]
            label = binary_cols[i]
            sc = ax.scatter(umap_coords[:, 0], umap_coords[:, 1], c=color)
            cbar = fig.colorbar(sc, orientation='vertical', label=label)
            cbar.ax.yaxis.set_tick_params(labelsize=18)  # Set colorbar tick label size
            cbar.set_label(label, fontsize=20)  # Set colorbar label size
    fig.supxlabel('UMAP X', fontsize=20)
    fig.supylabel('UMAP Y', fontsize=20)
    fig.suptitle('UMAP of Eigenspace', fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_name, f'umap_eigenspace_{file_name}_binaries.png'))
    plt.show()


def plot_transformation(Y_hr,  Y_pred_lr, score_lr, hr_coords_df,
                 dir_name, file_name, diagram_coords=['Teff', 'Lstar']):
    """Plots the results of all methods."""
    print("\n--- Plotting Results ---")
    # Determine grid size
    ncols = 2
    nrows = 1

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20,12), squeeze=False,
                             sharey=True, sharex=True)
    axes = axes.flatten()
    plot_idx = 0

    # Plotting parameters
    cmap = 'viridis'
    s = 5 # Scatter point size

    # a) Original HR Diagram
    ax = axes[plot_idx]
    sc = ax.scatter(Y_hr[:, 0], Y_hr[:, 1], s=s, c=Y_hr[:, 0], cmap=cmap) # Color by Temp
    ax.set_xlabel(diagram_coords[0], fontsize=20)
    ax.set_ylabel(diagram_coords[1], fontsize=20)
    ax.invert_xaxis() # Standard HR diagram convention
    if diagram_coords[1] == 'logg':
        ax.invert_yaxis()
    # cbar = fig.colorbar(sc, ax=ax, label=diagram_coords[0])
    # cbar.ax.yaxis.set_tick_params(labelsize=18)  # Set colorbar tick label size
    # cbar.set_label(diagram_coords[0], fontsize=20)  # Set colorbar label size
    plot_idx += 1

    # Linear Regression Prediction
    ax = axes[plot_idx]
    sc = ax.scatter(Y_pred_lr[:, 0], Y_pred_lr[:, 1], s=s, c=Y_hr[:, 0], cmap=cmap)
    ax.set_xlabel(f"Predicted {diagram_coords[0]}", fontsize=20)
    ax.set_ylabel(f"Predicted {diagram_coords[1]}", fontsize=20)
    ax.set_title(f'$R^2$ - {score_lr:.4f}', fontsize=20)
    ax.invert_xaxis() # Match original HR convention
    if diagram_coords[1] == 'logg':
        ax.invert_yaxis()
    cbar = fig.colorbar(sc, ax=ax, label=diagram_coords[0])
    cbar.ax.yaxis.set_tick_params(labelsize=18)  # Set colorbar tick label size
    cbar.set_label(diagram_coords[0], fontsize=20)  # Set colorbar label size
    plot_idx += 1

    # fig.suptitle(f"HR Diagram Transformations from Projections ({file_name})", fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to prevent title overlap
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

def plot_diagrams(dir_name, file_name):
    projections, U_standard, df = prepare_data(
        dir_name, file_name,
    )
    for coords in [['M_G_0','BPmRP_0'], ['Teff', 'Lstar'], ['Mstar', 'Rstar']]:
        transform_eigenspace(df, projections, dir_name, file_name, coords)

    plot_umap(df, U_standard, dir_name, file_name)