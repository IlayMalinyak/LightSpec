import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import umap
import pandas as pd
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


def features_analysis(model, device, linear_layer, dataloader, top_k=3, use_y_as_latent=True):
    model.eval()

    weight = linear_layer.weight.detach()
    
    # Calculate eigenvalues and eigenvectors
    # For a non-symmetric matrix, we need to use torch.linalg.eig
    # For a symmetric matrix, we could use torch.linalg.eigh
    if torch.allclose(weight, weight.t(), atol=1e-6):
        # Symmetric case - eigenvalues are real
        eigenvalues, eigenvectors = torch.linalg.eigh(weight)
        # Sort in descending order of eigenvalues
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
    else:
        # Non-symmetric case - eigenvalues may be complex
        eigenvalues_complex, eigenvectors_complex = torch.linalg.eig(weight)
        # For simplicity, we'll take the magnitude of complex eigenvalues
        eigenvalues_mag = torch.abs(eigenvalues_complex)
        idx = torch.argsort(eigenvalues_mag, descending=True)
        eigenvalues = eigenvalues_complex[idx]
        eigenvectors = eigenvectors_complex[:, idx]
        
        # Note: In practice, you might want to handle complex eigenvectors differently
        # Here we'll just take the real part for demonstration
        if torch.is_complex(eigenvectors):
            print("Warning: Complex eigenvectors detected. Using real part for projections.")
            eigenvectors = eigenvectors.real
    
    # Normalize eigenvectors
    eigenvectors = F.normalize(eigenvectors, dim=0)
    
    for batch in dataloader:
        lc, spectra, y, lc_target, spectra_target, info = batch
        lc, spectra, y, lc_target, spectra_target = lc.to(device), spectra.to(device), y.to(device), lc_target.to(device), spectra_target.to(device)
        latent = y.nan_to_num(-1) if self.use_y_as_latent else None
        out = self.model(lc ,spectra, latent=latent)
        dual_pred = out['dual_pred']
        lc_proj, spec_proj = dual_pred['proj1'], dual_pred['proj2']
        lc_emb, spec_emb = dual_pred['emb1'], dual_pred['emb2']
        eigenvalues, eigenvectors, top_projections = analyze_eigenspace_projections(linear_layer, lc_emb, top_k=top_k)
        eigenvalues2, eigenvectors2, top_projections2 = analyze_eigenspace_projections(linear_layer, spec_emb, top_k=top_k)

        print(f"Eigenvalues: {eigenvalues}")
        print(f"Eigenvectors: {eigenvectors}")
        print(f"Top {top_k} projections: {top_projections}")


def analyze_eigenspace_projections(linear_layer, batch_vectors, top_k=3):
    """
    Calculate eigenvectors and eigenvalues of a linear layer's weight matrix,
    then project input vectors onto this eigenspace and identify the most significant components.
    
    Args:
        linear_layer (nn.Linear): PyTorch linear layer
        batch_vectors (torch.Tensor): Batch of vectors with shape [batch_size, embed_dim]
        top_k (int): Number of top projections to return
        
    Returns:
        tuple: (eigenvalues, eigenvectors, top_projections)
            - eigenvalues: Sorted eigenvalues (largest first)
            - eigenvectors: Corresponding eigenvectors
            - top_projections: Dict with 'values' and 'indices' of top-k projections for each vector
    """
    # Get the weight matrix from the linear layer
    weight = linear_layer.weight.detach()
    
    # Calculate eigenvalues and eigenvectors
    # For a non-symmetric matrix, we need to use torch.linalg.eig
    # For a symmetric matrix, we could use torch.linalg.eigh
    if torch.allclose(weight, weight.t(), atol=1e-6):
        # Symmetric case - eigenvalues are real
        eigenvalues, eigenvectors = torch.linalg.eigh(weight)
        # Sort in descending order of eigenvalues
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
    else:
        # Non-symmetric case - eigenvalues may be complex
        eigenvalues_complex, eigenvectors_complex = torch.linalg.eig(weight)
        # For simplicity, we'll take the magnitude of complex eigenvalues
        eigenvalues_mag = torch.abs(eigenvalues_complex)
        idx = torch.argsort(eigenvalues_mag, descending=True)
        eigenvalues = eigenvalues_complex[idx]
        eigenvectors = eigenvectors_complex[:, idx]
        
        # Note: In practice, you might want to handle complex eigenvectors differently
        # Here we'll just take the real part for demonstration
        if torch.is_complex(eigenvectors):
            print("Warning: Complex eigenvectors detected. Using real part for projections.")
            eigenvectors = eigenvectors.real
    
    # Normalize eigenvectors
    eigenvectors = F.normalize(eigenvectors, dim=0)
    
    # Project batch vectors onto eigenvectors
    # Shape: [batch_size, embed_dim] @ [embed_dim, embed_dim] -> [batch_size, embed_dim]
    projections = batch_vectors @ eigenvectors
    
    # Get the top-k projections for each vector
    projection_magnitudes = torch.abs(projections)
    top_values, top_indices = torch.topk(projection_magnitudes, k=min(top_k, projections.shape[1]), dim=1)
    
    # Create a result dictionary with the top projections
    top_projections = {
        'values': projections.gather(1, top_indices),  # Get actual projection values (with sign)
        'indices': top_indices,  # Eigenvalue indices (sorted by magnitude)
        'eigenvalues': eigenvalues[top_indices]  # Corresponding eigenvalues
    }
    
    return eigenvalues, eigenvectors, top_projections


def create_umap(model,
                dl,
                device,
                stack_pairs=False,
                temperature=1,
                use_w=True,
                full_input=True,
                logits_key='logits',
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
        # if combine:
        #     lc, spec, lc2, spec2, info1, info2 = batch 
        #     lc, lc2, spec, spec2 = lc.to(device), lc2.to(device), spec.to(device), spec2.to(device)
        #     spec = torch.nn.functional.pad(spec, (0, lc.shape[-1] - spec.shape[-1], 0,0))
        #     spec2 = torch.nn.functional.pad(spec2, (0, lc2.shape[-1] - spec2.shape[-1], 0,0))
        #     x1 = torch.cat((lc, spec.unsqueeze(1)), dim=1)
        #     x2 = torch.cat((lc2, spec2.unsqueeze(1)), dim=1)
        # else:
        #     x1, x2, _, _, info1, info2 = batch
        #     x1, x2 = x1.to(device), x2.to(device)
        lc, spectra, y, lc_target, spectra_target, info = batch
        lc, spectra, lc_target, spectra_target = lc.to(device), spectra.to(device), lc_target.to(device), spectra_target.to(device)
        
        # Process model outputs
        with torch.no_grad():
            if stack_pairs:
                x = torch.cat((lc, lc_target), dim=0)
                out = model(x, temperature=self.temperature)
            elif use_w:
                w = torch.stack([i['w'] for i in info]).to(device)
                if full_input:
                    out = model(lightcurves=lc, spectra=spectra,
                                 lightcurves2=lc_target, spectra2=spectra_target,
                                  w=w)
                else:
                    out = model(lc, spectra, w=w)
            else:
                out = model(lc, lc_target)
        
        # Extract logits and apply UMAP
        if isinstance(out, dict):
            logits = out[logits_key]
        elif isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out
        if isinstance(logits, tuple):
            logits = logits[0]
        if logits.dim() > 2:
            logits = logits.mean(dim=1)
        # print(logits.shape, logits[:, 10])
        reduced_data = reducer.fit_transform(logits.cpu().numpy())

        norm = torch.norm(logits, dim=1, keepdim=True)
        norm_logits = logits / norm
        # Aggregate metadata
        batch_metadata = []
        for sample_info in info:
            
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
            metadata['logits_std'] = norm_logits.squeeze().cpu().numpy().std()
            metadata['logits_mean'] = norm_logits.squeeze().cpu().numpy().mean()
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
