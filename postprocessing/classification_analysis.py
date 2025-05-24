import pandas as pd
import numpy as np
import os
from eigenspace_analysis import transform_eigenspace, plot_diagrams, plot_umap, giant_cond, get_mag_data, latex_names, units
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import umap
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize



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


def get_df(results_dir, exp_name):
    df = pd.read_csv(os.path.join(results_dir, f'preds_{exp_name}.csv'))
    berger = pd.read_csv(berger_catalog_path)
    lightpred = pd.read_csv(lightpred_full_catalog_path)
    godoy = pd.read_csv(godoy_catalog_path)

    df = df.merge(berger, right_on='KID', left_on='kid', how='left')
    df = get_mag_data(df)
    age_cols = [c for c in lightpred.columns if 'age' in c]  # Robustly find age columns
    df = df.merge(lightpred[['KID', 'predicted period', 'mean_period_confidence'] + age_cols], right_on='KID',
                  left_on='kid', how='left')
    df = df.merge(godoy, right_on='KIC', left_on='kid', how='left', suffixes=['', '_godoy'])
    df['subgiant'] = (df['flag_CMD'] == 'Subgiant').astype(int)
    df['main_seq'] = df.apply(giant_cond, axis=1)
    return df

def plot_confusion_mat(y_true, y_pred, results_dir, file_name):
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    # Labeling
    num_classes = len(np.unique(y_true))
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Display counts
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'confusion_{file_name}.png'))
    plt.show()

def plot_roc_curve(df, cls_names, results_dir, file_name):
    num_classes = len([col for col in df.columns if col.startswith('pred_')])
    assert num_classes == len(cls_names)
    df['target'] = df['target'].astype(int)

    # Binarize ground truth labels for multiclass ROC/PR curves
    if num_classes == 2:
        # Special case: binary classification -> manually build two columns
        y_true_bin = np.zeros((len(df), 2))
        y_true_bin[np.arange(len(df)), df['target']] = 1
    else:
        y_true_bin = label_binarize(df['target'], classes=np.arange(num_classes))

    # Stack predicted probabilities
    y_score = df[[f'pred_{i}' for i in range(num_classes)]].values

    # --- ROC CURVES ---
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])


    # --- PLOTTING ROC CURVES ---
    # plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f'{cls_names[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(os.path.join(results_dir, f'roc_{file_name}.png'))
    plt.show()

def plot_precision_recall_curve(df, cls_names, results_dir, file_name):
    num_classes = len([col for col in df.columns if col.startswith('pred_')])
    assert num_classes == len(cls_names)

    # Binarize ground truth labels for multiclass ROC/PR curves
    if num_classes == 2:
        # Special case: binary classification -> manually build two columns
        y_true_bin = np.zeros((len(df), 2))
        y_true_bin[np.arange(len(df)), df['target']] = 1
    else:
        y_true_bin = label_binarize(df['target'], classes=np.arange(num_classes))

    # Stack predicted probabilities
    y_score = df[[f'pred_{i}' for i in range(num_classes)]].values

    # --- PRECISION-RECALL CURVES ---
    precision = dict()
    recall = dict()
    average_precision = dict()

    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_true_bin[:, i], y_score[:, i])

    # --- PLOTTING PRECISION-RECALL CURVES ---
    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        plt.plot(recall[i], precision[i], label=f'{cls_names[i]} (AP = {average_precision[i]:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc='lower left')
    plt.grid()
    plt.savefig(os.path.join(results_dir, f'precision_recall_{file_name}.png'))
    plt.show()

def plot_cls_umap(df, umap_coords, results_dir, file_name):
    # Define your color mapping manually
    color_dict = {-1: 'plum', 0: 'gray', 1: 'lightskyblue', 2: 'moccasin',  4: 'gold'}
    label_dict = {-1: 'RV', 0: 'Spectroscopic', 1: 'EB', 2: 'Astrometric', 4: 'Singles'}

    # Initialize binary_color
    df['binary_color'] = np.ones(len(df), dtype=int) * -999

    # Assign labels
    df.loc[df['flag_RVvariable'], 'binary_color'] = -1
    df.loc[df['NSS_Binary_Type'] == 'spectroscopic', 'binary_color'] = 0
    df.loc[(df['flag_EB_Kepler'] == 1)
           | (df['flag_EB_Gaia'] == 1)
           | (df['NSS_Binary_Type'] == 'eclipsing')
           | (df['NSS_Binary_Type'] == 'eclipsing+spectroscopic'), 'binary_color'] = 1
    df.loc[(df['NSS_Binary_Type'] == 'astrometric')
           | (df['NSS_Binary_Type'] == 'spectroscopic+astrometric'), 'binary_color'] = 2
    df.loc[df['flag_Binary_Union'] == 0, 'binary_color'] = 4

    print("min value binary color: ", np.min(df['binary_color']))

    # Create the plot
    fig, ax = plt.subplots()

    # Plot each group separately
    for value, color in color_dict.items():
        mask = df['binary_color'] == value
        ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                   c=color, label=label_dict[value], s=50)

    # Build custom legend automatically
    ax.legend(fontsize=20, loc='best')

    ax.set_xlabel('UMAP X', fontsize=20)
    ax.set_ylabel('UMAP Y', fontsize=20)
    # ax.set_title('UMAP of Eigenspace', fontsize=20)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'umap_{file_name}_binaries_color.png'))
    plt.show()

def plot_classification_results(results_dir, exp_name, plot_umap=False):

    df = get_df(results_dir, exp_name)
    if plot_umap:
        projections = np.load(os.path.join(results_dir, f'projections_{exp_name}.npy'))
        features = np.load(os.path.join(results_dir, f'features_{exp_name}.npy'))
        reducer_standard = umap.UMAP(n_components=2, random_state=42)  # Added random_state for reproducibility
        U_proj = reducer_standard.fit_transform(projections)
        U_feat = reducer_standard.fit_transform(features)
        # plot_umap(df, U_proj, results_dir, exp_name + '_proj')
        plot_cls_umap(df, U_feat, results_dir, exp_name + '_feat')

    y_true = df['target']
    y_pred = df['preds_cls']
    accuracy = accuracy_score(y_true, y_pred)

    # Calculate precision, recall, and F1 score
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')

    plot_confusion_mat(y_true, y_pred, results_dir, exp_name)
    plot_roc_curve(df, ['singles', 'binaries'], results_dir, exp_name)
    plot_precision_recall_curve(df, ['singles', 'binaries'], results_dir, exp_name)


def compare_cls_experiments(dir):
    files_nss = [f for f in os.listdir(dir) if f.endswith('csv') and f.startswith('preds_')]

    # Define colors for different experiments
    colors = plt.cm.tab10(np.linspace(0, 1, len(files_nss)))

    # Create subplots for ROC and PR curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    cls_names = ['singles', 'binaries']

    colors = plt.cm.get_cmap('Paired', 8)

    for i, file in enumerate(files_nss):
        filename = file.replace('preds_', '').replace('.csv', '')
        words = filename.split('_')
        if words[-1] == 'spec' or words[-1] == 'light' or words[-1] == 'former' or words[-1] == 'uniqueLoader':
            start_idx = -2
            model_name = '_'.join(words[start_idx:])
        else:
            model_name = words[-1]

        # Load the predictions file
        df = pd.read_csv(os.path.join(dir, file))

        # Get number of classes and prepare data
        num_classes = len([col for col in df.columns if col.startswith('pred_')])
        assert num_classes == len(cls_names)
        df['target'] = df['target'].astype(int)

        # Binarize ground truth labels for multiclass ROC/PR curves
        if num_classes == 2:
            y_true_bin = np.zeros((len(df), 2))
            y_true_bin[np.arange(len(df)), df['target']] = 1
        else:
            y_true_bin = label_binarize(df['target'], classes=np.arange(num_classes))

        # Stack predicted probabilities
        y_score = df[[f'pred_{j}' for j in range(num_classes)]].values

        color = colors(i)
        model_name = latex_names[model_name]

        # --- ROC CURVES ---
        for class_idx in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, class_idx], y_score[:, class_idx])
            roc_auc = auc(fpr, tpr)

            # linestyle = '--' if cls_names[class_idx] == 'singles' else '-'
            if cls_names[class_idx] == 'binaries':
                ax1.plot(fpr, tpr,
                         color=color,
                         label=f'{model_name} - (AUC = {roc_auc:.2f})',
                         )
            # else:
            #     ax1.plot(fpr, tpr,
            #              color=colors(i),
            #              label=f'{model_name} - (AUC = {roc_auc:.2f})',
            #              )



        # --- PRECISION-RECALL CURVES ---
        for class_idx in range(num_classes):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, class_idx], y_score[:, class_idx])
            average_precision = average_precision_score(y_true_bin[:, class_idx], y_score[:, class_idx])

            # linestyle = '--' if cls_names[class_idx] == 'singles' else '-'
            if cls_names[class_idx] == 'binaries':
                ax2.plot(recall, precision,
                         color=color,
                         label=f'{model_name} (AP = {average_precision:.2f})')

    # Configure ROC plot
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    # ax1.set_title('ROC Curves Comparison', fontsize=14)
    ax1.legend(loc='lower right', fontsize=24)
    ax1.grid(True, alpha=0.3)

    # Configure Precision-Recall plot
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    # ax2.set_title('Precision-Recall Curves Comparison', fontsize=14)
    ax2.legend(loc='lower left', fontsize=24)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(dir, 'experiments_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary statistics for each experiment
    print("\nExperiment Summary:")
    print("-" * 80)
    for file in files_nss:
        file_name = file.replace('preds_', '').replace('.csv', '')
        df = pd.read_csv(os.path.join(dir, file))

        y_true = df['target']
        y_pred = df['preds_cls']

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        print(f"{file_name:20} | Acc: {accuracy:.3f} | Prec: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")