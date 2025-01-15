import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import umap
import pandas as pd
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

def create_umap(model,
                dl,
                device,
                stack_pairs=False,
                temperature=1,
                use_w=True,
                dual=True,
                max_iter=np.inf):
    # Initialize UMAP reducer
    reducer = umap.UMAP(n_components=2)
    
    # Lists to collect data
    data_list = []
    metadata_list = []
    print('Extracting UMAP coordinates...')
    for i, batch in enumerate(tqdm(dl)):
        if i >= max_iter:
            break
        
        x1, x2, w, _, info1, info2 = batch
        x1, x2 = x1.to(device), x2.to(device)
        
        # Process model outputs
        with torch.no_grad():
            if stack_pairs:
                x = torch.cat((x1, x2), dim=0)
                out = model(x, temperature=temperature)
            else:
                if use_w:
                    out = model(x1, x2, w)
                elif dual:
                    out = model(x1, x2)
                else:
                    out = model(x1)
        
        # Extract logits and apply UMAP
        logits = out['logits'] if dual else out
        if isinstance(logits, tuple):
            logits = logits[0]
        reduced_data = reducer.fit_transform(logits.cpu().numpy())
        
        # Aggregate metadata
        batch_metadata = []
        for sample_info in info1:
            
            # Flatten dictionary, handling nested structures if needed
            flat_info = {}
            for key, value in sample_info.items():
                # If value is a tensor, convert to scalar or list
                if torch.is_tensor(value):
                    value = value.cpu().numpy().tolist()
                flat_info[key] = value
            batch_metadata.append(flat_info)
        
        # Combine metadata with UMAP coordinates
        for metadata, umap_coords in zip(batch_metadata, reduced_data):
            metadata['umap_x'] = umap_coords[0]
            metadata['umap_y'] = umap_coords[1]
            data_list.append(metadata)
    
    # Create DataFrame
    df = pd.DataFrame(data_list)
    
    return df

def draw_umap(log_path, color_col='Teff'):
    umap_files = [os.path.join(log_path, f) for f in os.listdir(log_path) if 'umap' in f]
    for umap_file in umap_files:
        print(umap_file)
        df = pd.read_csv(umap_file)
        plt.scatter(df['umap_x'], df['umap_y'], c=df[color_col], cmap='viridis')
        plt.xlabel('UMAP X')
        plt.ylabel('UMAP Y')
        plt.colorbar(label=color_col)
        plt.savefig(f'figs/{os.path.basename(log_path).split(".")[0]}_umap_{color_col}.png')
        plt.show()
