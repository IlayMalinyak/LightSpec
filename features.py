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
                combine=False,
                return_predictions=False,
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
        if combine:
            lc, spec, lc2, spec2, info1, info2 = batch 
            lc, lc2, spec, spec2 = lc.to(device), lc2.to(device), spec.to(device), spec2.to(device)
            spec = torch.nn.functional.pad(spec, (0, lc.shape[-1] - spec.shape[-1], 0,0))
            spec2 = torch.nn.functional.pad(spec2, (0, lc2.shape[-1] - spec2.shape[-1], 0,0))
            x1 = torch.cat((lc, spec.unsqueeze(1)), dim=1)
            x2 = torch.cat((lc2, spec2.unsqueeze(1)), dim=1)
        else:
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
        if logits.dim() > 2:
            logits = logits.mean(dim=1)
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
        for metadata, umap_coords, logit in zip(batch_metadata, reduced_data, logits):
            metadata['umap_x'] = umap_coords[0]
            metadata['umap_y'] = umap_coords[1]
            metadata['logits_std'] = logit.squeeze().cpu().numpy().std()
            metadata['logits_mean'] = logits.squeeze().cpu().numpy().mean()
            data_list.append(metadata)
    
    # Create DataFrame
    df = pd.DataFrame(data_list)
    print(df.columns)
    
    return df

def multimodal_umap(model,
                dl,
                lc_encoder,
                spec_encoder,
                device,
                stack_pairs=False,
                temperature=1,
                use_w=False,
                dual=True,
                return_predictions=False,
                max_iter=np.inf):
    # Initialize UMAP reducer
    reducer = umap.UMAP(n_components=2)
    
    # Lists to collect data
    data_dual = []
    data_lc = []
    data_spec = []
    metadata_list = []
    
    print('Extracting UMAP coordinates...')
    for i, batch in enumerate(tqdm(dl)):
        if i >= max_iter:
            break
        
        lc, spec, w, _, info1, info2 = batch
        lc, spec = lc.to(device), spec.to(device)
        
        # Process model outputs
        with torch.no_grad():
            lc_out, _ = lc_encoder(lc)
            spec_out, _ = spec_encoder(spec)
            dual_out = model(lc, spec)
            if use_w:
                dual_out = model(lc, spec, w)
            else:
                dual_out = model(lc, spec)
           
        # Extract logits and apply UMAP
        dual_logits = dual_out['logits']
       
        reduced_dual = reducer.fit_transform(dual_logits.cpu().numpy())
        reduced_lc = reducer.fit_transform(lc_out.cpu().numpy())
        reduced_spec = reducer.fit_transform(spec_out.cpu().numpy())

        print(reduced_lc.shape, reduced_spec.shape, reduced_dual.shape)
        
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
        for metadata, dual_coords in zip(batch_metadata, reduced_dual):
            print(dual_coords.shape)
            metadata['umap_x'] = dual_coords[0]
            metadata['umap_y'] = dual_coords[1]
            data_dual.append(metadata)
        
        for metadata, lc_coords in zip(batch_metadata, reduced_lc):
            metadata['umap_x'] = lc_coords[0]
            metadata['umap_y'] = lc_coords[1]
            data_lc.append(metadata)
        
        for metadata, spec_coords in zip(batch_metadata, reduced_spec):
            metadata['umap_x'] = spec_coords[0]
            metadata['umap_y'] = spec_coords[1]
            data_spec.append(metadata)
        print(data_spec[0]['umap_x'],data_lc[0]['umap_x'],data_dual[0]['umap_x'])
    # Create DataFrame
    df_dual = pd.DataFrame(data_dual)
    df_lc = pd.DataFrame(data_lc)
    df_spec = pd.DataFrame(data_spec)
    
    return df_dual, df_lc, df_spec

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
