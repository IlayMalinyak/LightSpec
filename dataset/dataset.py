import os
import torch
from torch.utils.data import Dataset
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl
import pandas as pd
from typing import List
import copy
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import traceback

import sys
from os import path
ROOT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print("running from ", ROOT_DIR) 

from transforms.transforms import *
from dataset.sampler import DistributedUniqueLightCurveSampler, DistributedBalancedSampler, UniqueIDDistributedSampler

mpl.rcParams['axes.linewidth'] = 4
plt.rcParams.update({'font.size': 30, 'figure.figsize': (14,10), 'lines.linewidth': 4})
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["gray", "r", "c", 'm', 'brown'])
plt.rcParams.update({'xtick.labelsize': 22, 'ytick.labelsize': 22})
plt.rcParams.update({'legend.fontsize': 22})

T_sun = 5778
T_MIN =3483.11
T_MAX = 7500.0
LOGG_MIN = -0.22
LOGG_MAX = 4.9
FEH_MIN = -2.5
FEH_MAX = 0.964
VSINI_MAX = 100
P_MAX = 70
MAX_AGE = 10

def pad_with_last_element(tensor, seq_len):
    """
    Pad the last dimension of a tensor from size T to size seq_len, 
    where the padding values for each dimension are the last elements of that dimension.
    
    Args:
        tensor: Input tensor of shape (dims, T)
        seq_len: Target sequence length for the last dimension
        
    Returns:
        Padded tensor of shape (dims, seq_len)
    """
    dims, T = tensor.shape
    
    if T >= seq_len:
        # If the tensor is already longer than or equal to seq_len, truncate it
        return tensor[:, :seq_len]
    
    # Create the padded tensor
    padded = torch.zeros((dims, seq_len), dtype=tensor.dtype, device=tensor.device)
    
    # Copy the original data
    padded[:, :T] = tensor
    
    # For each dimension, fill the padding with the last element of that dimension
    for i in range(dims):
        padded[i, T:] = tensor[i, -1]
    
    return padded

def create_boundary_values_dict(df):
  boundary_values_dict = {}
  for c in df.columns:
    if isinstance(df[c].values[0], str):
      continue
    if c not in boundary_values_dict.keys():
      if c == 'Butterfly':
        boundary_values_dict[c] = bool(df[c].values[0])
      else:
        min_val, max_val = df[c].min(), df[c].max()
        boundary_values_dict[f'min {c}'] = float(min_val)
        boundary_values_dict[f'max {c}'] = float(max_val)
  return boundary_values_dict

class SimulationDataset(Dataset):
    def __init__(self,
                 df,
                labels=['Period', 'Inclination'],
                light_transforms=None,
                spec_transforms=None,
                npy_path = None,
                spec_path = None,
                use_acf=False,
                use_fft=False,
                scale_flux=False,
                meta_columns=None,
                spec_seq_len=4096,
                light_seq_len=34560,
                example_wv_path='/data/lamost/example_wv.npy'
                ):
        self.df = df
        self.spec_seq_len = spec_seq_len
        self.lc_seq_len = light_seq_len
        self.use_acf = use_acf
        self.use_fft = use_fft
        self.range_dict = dict()
        self.labels = labels
        # self.labels_df = self.df[labels]
        self.update_range_dict()
        # self.lc_dir = os.path.join(data_dir, 'lc')
        # self.spectra_dir = os.path.join(data_dir, 'lamost')
        self.lc_transforms = light_transforms
        self.spectra_transforms = spec_transforms
        self.example_wv = np.load(example_wv_path)
        self.Nlc = len(self.df)
        self.scale_flux = scale_flux
        self.meta_columns = meta_columns
        self.boundary_values_dict = create_boundary_values_dict(self.df)

    def fill_nan_inf_np(self, x: np.ndarray, interpolate: bool = True):
        """
        Fill NaN and Inf values in a numpy array

        Args:
            x (np.ndarray): array to fill
            interpolate (bool): whether to interpolate or not

        Returns:
            np.ndarray: filled array
        """
        # Create a copy to avoid modifying the original array
        x_filled = x.copy()
        
        # Identify indices of finite and non-finite values
        finite_mask = np.isfinite(x_filled)
        non_finite_indices = np.where(~finite_mask)[0]

        finite_indices = np.where(finite_mask)[0]
        
        # If there are non-finite values and some finite values
        if len(non_finite_indices) > 0 and len(finite_indices) > 0:
            if interpolate:
                # Interpolate non-finite values using linear interpolation
                interpolated_values = np.interp(
                    non_finite_indices, 
                    finite_indices, 
                    x_filled[finite_mask]
                )
                # Replace non-finite values with interpolated values
                x_filled[non_finite_indices] = interpolated_values
            else:
                # Replace non-finite values with zero
                x_filled[non_finite_indices] = 0
        
        return x_filled

    def update_range_dict(self):
        for name in self.labels:
            min_val = self.df[name].min()
            max_val = self.df[name].max()
            self.range_dict[name] = (min_val, max_val)
        
    def __len__(self):
        return len(self.df)

    def _normalize(self, x, label):
        # min_val = float(self.range_dict[key][0])
        # max_val = float(self.range_dict[key][1])
        # return (x - min_val) / (max_val - min_val)
        if 'period' in label.lower():
            x = x / P_MAX
        elif 'age' in label.lower():
            x = x / MAX_AGE
        return x

        # min_val, max_val = self.boundary_values_dict[f'min {label}'], self.boundary_values_dict[f'max {label}']
        # return (x - min_val)/(max_val - min_val)

    def transform_lc_flux(self, flux, info_lc):
        if self.lc_transforms:
            flux,_,info_lc = self.lc_transforms(flux, info=info_lc)
            if self.use_acf:
                acf = torch.tensor(info_lc['acf']).nan_to_num(0)
                flux = torch.cat((flux, acf), dim=0)
            if self.use_fft:
                fft = torch.tensor(info_lc['fft']).nan_to_num(0)
                flux = torch.cat((flux, fft), dim=0)
        if flux.shape[-1] == 1:
            flux = flux.squeeze(-1)
        if len(flux.shape) == 1:
            flux = flux.unsqueeze(0)
        flux = pad_with_last_element(flux, self.lc_seq_len)
        return flux, info_lc

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        padded_idx = f'{idx:d}'.zfill(int(np.log10(self.Nlc))+1)
        s = time.time()
        try:
            spec = pd.read_parquet(row['spec_data_path']).values
            # spec = self.fill_nan_np(spec)
        except (FileNotFoundError, OSError, IndexError) as e:   
            # print("Error reading file ", idx, e)
            spec = np.zeros((3909, 1))
        try: 
            # print(idx, row['lc_data_path'])
            lc = pd.read_parquet(row['lc_data_path']).values
            # lc[:, 1] = self.fill_nan_inf_np(lc[:, 1])
            # max_val = np.max(np.abs(lc[:, 1]))
            # if max_val > 1e2:
            #     lc[np.abs(lc[:, 1]) > 1e2, 1] = np.random.uniform(0, 2, size=lc[np.abs(lc[:, 1]) > 1e2, 1].shape)
            
            # lc[:, 1] = lc[:, 1] / lc[:, 1].max()
        except (FileNotFoundError, OSError, IndexError) as e:
            print("Error reading file ", idx, e)
            lc = np.zeros((48000, 2))
        spectra = spec[:,-1]
        target_spectra = spectra.copy()
        flux = lc[:,-1]
        target_flux = flux.copy()
        info_s = dict()
        info_lc = dict()
        try:
            # label = row[self.labels].to_dict()
            # print(label)
            # label = {k: self._normalize(v, k) for k, v in label.items()}
            y = torch.tensor([self._normalize(row[k], k) for k in self.labels], dtype=torch.float32)

        except IndexError:
            y = torch.zeros(len(self.labels))
        s1 = time.time()
        info_s['wavelength'] = self.example_wv
        if self.spectra_transforms:
            spectra, _,info_s = self.spectra_transforms(spectra, info=info_s)
        spectra = pad_with_last_element(spectra, self.spec_seq_len)
        spectra = torch.nan_to_num(spectra, nan=0)
        s2 = time.time()
        # print("nans in spectra: ", np.sum(np.isnan(spectra)))
        # spectra = torch.tensor(spectra).float()
        flux, info_lc = self.transform_lc_flux(flux, info_lc)
        target_flux, _ = self.transform_lc_flux(target_flux, info_lc)
        info = {'spectra': info_s, 'lc': info_lc}
        if 'L' in row.keys():
            info['KMAG'] = row['L']
        else:
            info['KMAG'] = 1
        s3 = time.time()
        flux = flux.nan_to_num(0).float()
        spectra = spectra.nan_to_num(0).float()
        target_flux = target_flux.nan_to_num(0).float()
        # print(flux.shape, target_flux.shape, spectra.shape, y.shape)
        # print(s1-s, s2-s1, s3-s2)
        return flux, spectra, y , target_flux,  spectra, info

class SpectraDataset(Dataset):
    """
    dataset for spectra data
    Args:
        data_dir: path to the data directory
        transforms: transformations to apply to the data
        df: dataframe containing the data paths
        max_len: maximum length of the spectra
        use_cache: whether to use a cache file
        id: column name for the observation id
    """
    def __init__(self, data_dir,
                     transforms=None,
                     df=None,
                    max_len=3909,
                    use_cache=True,
                    id='combined_obsid',
                    target_norm='solar'
                    ):
        self.data_dir = Path(data_dir)
        self.transforms = transforms
        self.target_norm = target_norm
        self.df = df
        self.id = id
        if df is None:
            cache_file = os.path.join(self.data_dir, '.path_cache.txt')
            
            if use_cache and os.path.exists(cache_file):
                print("Loading cached file paths...")
                with open(cache_file, 'r') as f:
                    self.path_list = np.array([line.strip() for line in f])
            else:
                print("Creating files list...")
                self.path_list = self._file_listing()
                if use_cache:
                    with open(cache_file, 'w') as f:
                        f.write('\n'.join(self.path_list))
        else:
            self.path_list = None
        self.max_len = max_len
        self.mask_transform = RandomMasking()
    
    def _file_listing(self):
        
        def process_chunk(file_names):
            return [self.data_dir / name for name in file_names]
        
        file_names = os.listdir(self.data_dir)
        chunk_size = 100000  
        chunks = [file_names[i:i + chunk_size] for i in range(0, len(file_names), chunk_size)]
        
        with ThreadPoolExecutor() as executor:
            paths = []
            for chunk_paths in executor.map(process_chunk, chunks):
                paths.extend(chunk_paths)
        
        return np.array(paths)
        
    def __len__(self):
        if self.df is not None:
            return len(self.df)
        return len(self.path_list) if self.path_list is not None else 0

    def read_lamost_spectra(self, filename):
        with fits.open(filename) as hdulist:
          binaryext = hdulist[1].data
          header = hdulist[0].header
        x = binaryext['FLUX'].astype(np.float32)
        wv = binaryext['WAVELENGTH'].astype(np.float32)
        rv = header['HELIO_RV']
        meta = {'RV': rv, 'wavelength': wv}
        return x, meta
    
    def read_apogee_spectra(self, filename):
        with fits.open(filename) as hdul:
            data = hdul[1].data.astype(np.float32).squeeze()[None]
        meta = {}
        return data, meta
    
    def create_lamost_target(self, row, info):
        info['Teff'] = row['combined_teff']
        info['rv2'] = row['combined_rv']
        info['logg'] = row['combined_logg']
        info['FeH'] = row['combined_feh']
        info['snrg'] = row['combined_snrg'] / 1000
        info['snri'] = row['combined_snri'] / 1000
        info['snrr'] = row['combined_snrr'] / 1000
        info['snrz'] = row['combined_snrz'] / 1000
        
        # Add joint probability information if available
        if 'joint_prob' in row:
            info['joint_prob'] = row['joint_prob']
            info['joint_weight'] = row['joint_weight']
            
        if self.target_norm =='solar':
            target = torch.tensor([-999, info['Teff'] / T_sun, info['logg'], info['FeH']], dtype=torch.float32)
        elif self.target_norm == 'minmax':
            teff = (info['Teff'] - T_MIN) / (T_MAX - T_MIN)
            logg = (info['logg'] - LOGG_MIN) / (LOGG_MAX - LOGG_MIN)
            feh = (info['FeH'] - FEH_MIN) / (FEH_MAX - FEH_MIN)
            target = torch.tensor([-999, teff, logg, feh], dtype=torch.float32)
        else:
            raise ValueError("Unknown target normalization method")
        return target, info
    
    def create_apogee_target(self, row, info):
        info['Teff'] = row['TEFF']
        info['logg'] = row['LOGG']
        info['FeH'] = row['FE_H']
        info['snrg'] = row['SNR'] / 1000
        info['snri'] = row['SNR'] / 1000
        info['snrr'] = row['SNR'] / 1000
        info['snrz'] = row['SNR'] / 1000
        info['vsini'] = row['VSINI']
        target = torch.tensor([info['vsini'] / VSINI_MAX, info['Teff'] / T_sun, info['logg'], info['FeH']], dtype=torch.float32)
        return target, info

    def create_empty_info(self, info):
        info['Teff'] = np.nan
        info['logg'] = np.nan
        info['FeH'] = np.nan
        info['snrg'] = 1e-4
        info['snri'] = 1e-4
        info['snrr'] = 1e-4
        info['snrz'] = 1e-4
        info['vsini'] = np.nan
        return info

    def __getitem__(self, idx):
        start = time.time()
        obsid = int(self.df.iloc[idx]['combined_obsid'])
        obsdir = str(obsid)[:4]
        spectra_filename = os.path.join(self.data_dir, f'{obsdir}/{obsid}.fits')
        try:
            if not self.df.iloc[idx]['APOGEE_ID'].isnan():     
                filepath = f"{self.data_dir}/aspcapStar-dr17-{self.df.iloc[idx][self.id]}.fits"
                spectra, meta = self.read_apogee_spectra(filepath)
            else:
                obsid = int(self.df.iloc[idx]['combined_obsid'])
                obsdir = str(obsid)[:4]
                spectra_filename = os.path.join(self.data_dir, f'{obsdir}/{obsid}.fits')
                spectra, meta = self.read_lamost_spectra(spectra_filename)
                spec_time = time.time() - start
                meta['obsid'] = obsid
        except OSError as e:
            print("Error reading file ", obsid, e)
            # spectra = np.zeros((self.spec_seq_len))
            # meta = {'RV': 0, 'wavelength': np.zeros(self.spec_seq_len)}
            return (torch.zeros(self.max_len),
                    torch.zeros(self.max_len),
                    torch.zeros(4),
                    torch.zeros(self.max_len, dtype=torch.bool),
                    torch.zeros(self.max_len, dtype=torch.bool),
                    {'obsid': obsid, 'snrg': 1e-4, 'snri': 1e-4, 'snrr': 1e-4, 'snrz': 1e-4})
            
        # if self.df is not None:
        #     if 'APOGEE' in self.id:     
        #         filepath = f"{self.data_dir}/aspcapStar-dr17-{self.df.iloc[idx][self.id]}.fits"
        #     else:
        #         filepath = self.df.iloc[idx]['data_path']
        #         # filepath = filepath.replace('data', 'storage', 1)
        #     target_size = 4
        # else: 
        #     filepath = self.path_list[idx]
        #     target_size = self.max_len    
        # obsid = os.path.basename(filepath)
        # try:
        #     if 'APOGEE' in self.id:
        #         spectra, meta = self.read_apogee_spectra(filepath)
        #     else:
        #         spectra, meta = self.read_lamost_spectra(filepath)
        # except (OSError, FileNotFoundError) as e:
        #     info = self.create_empty_info({self.id: obsid})
        #     print("Error reading file ", filepath, "\n", e)
        meta[self.id] = obsid
        t1 = time.time()
        if self.transforms:
            spectra, _, info = self.transforms(spectra, None, meta)
        spectra_masked, mask, _ = self.mask_transform(spectra, None, info)
        t2 = time.time()
        if self.df is not None:
            row = self.df.iloc[idx]
            if 'APOGEE' in self.id:
                target, info = self.create_apogee_target(row, meta)
            else:
                target, info = self.create_lamost_target(row, meta)
        else:
            print("no df!")
            target = torch.zeros_like(mask)
        t3 = time.time()
        if spectra_masked.shape[-1] < self.max_len:
            pad = torch.zeros(1, self.max_len - spectra_masked.shape[-1])
            spectra_masked = torch.cat([spectra_masked, pad], dim=-1)
            pad_mask = torch.zeros(1, self.max_len  - mask.shape[-1], dtype=torch.bool)
            mask = torch.cat([mask, pad_mask], dim=-1)
            pad_spectra = torch.zeros(1, self.max_len - spectra.shape[-1])
            spectra = torch.cat([spectra, pad_spectra], dim=-1)
        spectra = torch.nan_to_num(spectra, nan=0)
        spectra_masked = torch.nan_to_num(spectra_masked, nan=0)
        t4 = time.time()
        # print("read spectra: ", t1-start, "transform: ", t2-t1, "target: ", t3-t2, "pad: ", t4-t3)
        return (spectra_masked.float().squeeze(0), spectra.float().squeeze(0),\
         target.float(), mask.squeeze(0), mask.squeeze(0), info)


class KeplerDataset():
    """
    A dataset for Kepler data.
    """
    def __init__(self,
                df:pd.DataFrame=None,
                prot_df:pd.DataFrame=None,
                npy_path:str=None,
                transforms:object=None,
                seq_len:int=34560,
                target_transforms:object=None,
                masked_transforms:bool=False,
                use_acf:bool=False,
                use_fft:bool=False,
                scale_flux:bool=True,
                labels:object=None,
                 dims:int=1
                ):
        """
        dataset for Kepler data
        Args:
            df (pd.DataFrame): DataFrame containing Kepler paths
            prot_df (pd.DataFrame): DataFrame containing rotation periods
            npy_path (str): Path to numpy files
            transforms (object): Transformations to apply to the data
            seq_len (int): Sequence length
            target_transforms (object): Transformations to apply to the target
            masked_transforms (bool): Whether to apply masking transformations

        """
        self.df = df
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.npy_path = npy_path
        self.prot_df = prot_df
        self.seq_len = seq_len
        if df is not None and 'predicted period' not in df.columns:
            if prot_df is not None:
                self.df = pd.merge(df, prot_df[['KID', 'predicted period']], on='KID')
        self.mask_transform = RandomMasking(mask_prob=0.2) if masked_transforms else None
        self.use_acf = use_acf
        self.use_fft = use_fft
        self.scale_flux = scale_flux
        self.labels = labels
        self.dims = dims

    def __len__(self):
        return len(self.df)

    def read_lc(self, filename: str):
        """
        Reads a FITS file and returns the PDCSAP_FLUX and TIME columns as numpy arrays.

        Args:
            filename (str): The path to the FITS file.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The PDCSAP_FLUX and TIME columns as numpy arrays.
        """
        with fits.open(filename) as hdulist:
                binaryext = hdulist[1].data
                meta = hdulist[0].header
        df = pd.DataFrame(data=binaryext)
        x = df['PDCSAP_FLUX']
        time = df['TIME'].values
        return x,time, meta

    def read_row(self, idx):
        """
        Reads a row from the DataFrame.
        """
        row = self.df.iloc[idx]
        if 'prot' in row.keys():
            y_val = row['prot']
        elif 'Prot' in row.keys():
            y_val = row['Prot']
        else:
            y_val = np.nan

        if self.npy_path is not None:
            try:
                file_path = os.path.join(self.npy_path, f"{int(row['KID'])}.npy")
                x = np.load(file_path)            
                if not isinstance(x, np.ndarray) or x.size == 0:
                    print(f"Warning: Empty or invalid numpy array for {row['KID']}")
                    x = np.zeros((self.seq_len, 1))
            except FileNotFoundError as e:
                # print(f"Error: File not found for {row['KID']}", e)
                x = np.zeros((self.seq_len, 1))
            except Exception as e:
                print(f"Error loading file for {row['KID']}: {str(e)}")
                x = np.zeros((self.seq_len, 1))
            if 'Teff' in row.keys():
                meta = {'TEFF': row['Teff'],
                            'RADIUS': row['Rstar'],
                        'LOGG': row['logg'],
                        'M':row['Mstar'],
                        'FeH': row['FeH'],
                        'Dist': row['Dist']}
            else:
                meta = {'TEFF': None, 'RADIUS': None,
                        'LOGG': None, 'M': None, 'FeH':None,
                        'Dist': None}
            meta['KMAG'] = row['kmag_abs'] if 'kmag_abs' in row.keys() else None
            if self.labels is not None:
                for label in self.labels:
                    meta[label] = row[label]
            
            if 'predicted period' in row.keys():
                meta['predicted period'] = row['predicted period'] / P_MAX
            if 'Prot' in row.keys():
                meta['Prot'] = row['Prot'] / P_MAX
            else:
                meta['Prot'] = np.nan
            if 'Prot_ref' in row.keys():
                meta['Prot_ref'] = row['Prot_ref']
            else:
                meta['Prot_ref'] = np.nan
            conf_cols = [c for c in row.keys() if 'confidence' in c]
            prob_cols = [c for c in row.keys() if 'prob' in c]
            inc_cols = [c for c in row.keys() if 'inc' in c]
            age_cols = [c for c in row.keys() if 'age' in c]
            for c in conf_cols:
                meta[c] = row[c]
            for c in prob_cols:
                meta[c] = row[c]
            for c in inc_cols:
                meta[c] = row[c]
            for c in age_cols:
                meta[c] = row[c]
            if 'Pmax' in row.keys() and 'Pmin' in row.keys():
                meta['Pmax'] = row['Pmax']
                meta['Pmin'] = row['Pmin']
            if 'RUWE' in row.keys():
                meta['RUWE'] = row['RUWE']
            if 'sigma' in row.keys():
                meta['sigma'] = row['sigma']
            return x, meta, None, y_val
        else:
            paths = row['data_file_path']
            meta = {}
            for i in range(len(paths)):
                x, time, meta = self.read_lc(paths[i])
                meta = dict(meta.items())
                meta['M'] = row['Mstar'] if 'Mstar' in row.keys() else None
                meta['Dist'] = row['Dist'] if 'Dist' in row.keys() else None
                if i == 0:
                    x_tot = x.copy()
                else:
                    border_val = np.nanmean(x) - np.nanmean(x_tot)
                    x -= border_val
                    x_tot = np.concatenate((x_tot, np.array(x)))
                self.cur_len = len(x)
            return x_tot, meta, None, y_val

    def fill_nan_np(self, x:np.ndarray, interpolate:bool=True):
        """
        fill nan values in a numpy array

        Args:
                x (np.ndarray): array to fill
                interpolate (bool): whether to interpolate or not

        Returns:
            np.ndarray: filled array
        """
        non_nan_indices = np.where(~np.isnan(x))[0]
        nan_indices = np.where(np.isnan(x))[0]
        if len(nan_indices) and len(non_nan_indices):
            if interpolate:
                # Interpolate NaN values using linear interpolation
                interpolated_values = np.interp(nan_indices, non_nan_indices, x[non_nan_indices])
                # Replace NaNs with interpolated values
                x[nan_indices] = interpolated_values
            else:
                x[nan_indices] = 0	
        return x
    
    def transform_data(self, x, transforms, info, idx):
        try:
            if transforms is not None:
                x, mask, info = self.transforms(x, mask=None, info=info)
            else:
                x = torch.tensor(x)
                mask = torch.zeros(1, self.seq_len)
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            elif (len(x.shape) == 2) and (x.shape[-1] == 1):
                x = x.squeeze(-1).unsqueeze(0)
            if mask is not None and len(mask.shape) == 1:
                mask = mask.unsqueeze(0)
            x = pad_with_last_element(x, self.seq_len)
            if mask is not None:
                mask = pad_with_last_element(mask, self.seq_len)
            else:
                mask = torch.zeros(1, self.seq_len)
            if self.use_acf:
                acf = torch.tensor(info['acf']).nan_to_num(0)
                x = torch.cat((x, acf), dim=0)
            if self.use_fft:
                fft = torch.tensor(info['fft']).nan_to_num(0)
                x = torch.cat((x, fft), dim=0)
        except Exception as e:
            print(f"Error in transforms for index {idx}: {str(e)}")
            traceback.print_exc()
            x = torch.zeros(self.dims, self.seq_len)
            mask = torch.zeros(1, self.seq_len)
        return x, mask, info



    def __getitem__(self, idx):
        tic = time.time()
        x, info, qs, p_val = self.read_row(idx)
        x = self.fill_nan_np(x, interpolate=True)
        info['idx'] =  idx
        info['qs'] = qs
        info['period'] = p_val
        info_y = copy.deepcopy(info)
        if self.scale_flux:
            x = (x - x.min()) / (x.max() - x.min())
        target = x.copy()
        mask = None
        x, mask, info= self.transform_data(x, self.transforms, info, idx) 
        target, mask_y, info_y = self.transform_data(target, self.target_transforms, info_y, idx)       
        x = x.nan_to_num(0)
        target = target.nan_to_num(0)
        info['Teff'] = info['TEFF']
        info['Mstar'] = info['M']
        info['logg'] = info['LOGG']
        info['Rstar'] = info['RADIUS']
        info['kmag_abs'] = info['KMAG']
       
        info['KID'] = self.df.iloc[idx]['KID']
        toc = time.time()
        info['time'] = toc - tic
        if mask is None:
            mask = torch.zeros_like(x)
        if mask_y is None:
            mask_y = torch.zeros_like(target)
        if self.labels is not None:
            if 'Teff' in self.labels:
                info['Teff'] /= T_sun
            y = torch.tensor([info[label] for label in self.labels], dtype=torch.float32)
        else:
            y = target
        # Ensure info and info_y are always dictionaries
        info = info if isinstance(info, dict) else {}
        info_y = info_y if isinstance(info_y, dict) else {}
        result = (x.float(), mask, y, target.float(), mask, info)
        if any(item is None for item in result):
            print(f"Warning: None value in result for index {idx}")
            print(f"x: {x.shape if x is not None else None}")
            print(f"target: {target.shape if target is not None else None}")
            print(f"mask: {mask.shape if mask is not None else None}")
            print(f"mask_y: {mask_y.shape if mask_y is not None else None}")
            print(f"mask copy: {mask.shape if mask is not None else None}")
            print(f"info_y: {info_y}")
        return result


class LightSpecDataset(KeplerDataset):
    """
    A Multimodal dataset for spectra and lightcurve.
    Args:
        df (pd.DataFrame): DataFrame containing paths
        prot_df (pd.DataFrame): DataFrame containing rotation periods
        npy_path (str): Path to numpy files
        spec_path (str): Path to spectra files
        light_transforms (object): Transformations to apply to the lightcurve
        spec_transforms (object): Transformations to apply to the spectra
        light_seq_len (int): Sequence length for lightcurve
        spec_seq_len (int): Sequence length for spectra
        meta_columns (List[str]): Columns to use as metadata weights

    """
    def __init__(self, df:pd.DataFrame=None,
                prot_df:pd.DataFrame=None,
                npy_path:str=None,
                spec_path:str=None,
                light_transforms:object=None,
                spec_transforms:object=None,
                light_seq_len:int=34560,
                use_acf:bool=False,
                use_fft:bool=False,
                scale_flux:bool=True,
                spec_seq_len:int=3909,
                meta_columns = ['Teff', 'Mstar', 'logg'], labels=None
                ):
        super().__init__(df, prot_df, npy_path, light_transforms,
                light_seq_len, target_transforms=light_transforms, use_acf=use_acf, use_fft=use_fft, scale_flux=scale_flux)
        self.spec_path = spec_path
        self.spec_transforms = spec_transforms
        self.spec_seq_len = spec_seq_len
        self.meta_columns = meta_columns
        self.masked_transform = RandomMasking()
        self.labels = labels


    def read_spectra(self, filename):
        with fits.open(filename) as hdulist:
          binaryext = hdulist[1].data
          header = hdulist[0].header
        x = binaryext['FLUX'].astype(np.float32)
        wv = binaryext['WAVELENGTH'].astype(np.float32)
        rv = header['HELIO_RV']
        meta = {'RV': rv, 'wavelength': wv}
        return x, meta

    def __getitem__(self, idx):
        start = time.time()
        light, _, _, light_target, _ ,info = super().__getitem__(idx)
        t1 = time.time()
        kid = int(info['KID'])
        obsid = int(self.df.iloc[idx]['ObsID'])
        info['obsid'] = obsid
        obsdir = str(obsid)[:4]
        spectra_filename = os.path.join(self.spec_path, f'{obsdir}/{obsid}.fits')
        try:
            spectra, meta = self.read_spectra(spectra_filename)
            spec_time = time.time() - start
        except OSError as e:
            print("Error reading file ", obsid, e)
            spectra = np.zeros((self.spec_seq_len))
            wv = np.load('/data/lightSpec/lamost_wv.npy')
            to_pad = spectra.shape[0] - wv.shape[0]
            if to_pad > 0:
                wv = np.pad(wv, ((0, to_pad)), mode='constant')
            meta = {'RV': 0, 'wavelength': wv}
        t2 = time.time()
        if self.spec_transforms:
            spectra, _, spec_info = self.spec_transforms(spectra, None, meta)
            spec_transform_time = time.time() - start
        if spectra.shape[-1] < self.spec_seq_len:
            spectra = F.pad(spectra, ((0, self.spec_seq_len - spectra.shape[-1],0,0)), "constant", value=0)
        t3 = time.time()
        spectra = torch.nan_to_num(spectra, nan=0)
        masked_spectra, _, spec_info = self.masked_transform(spectra, None, spec_info)
        info.update(spec_info)
        t4 = time.time()
        if self.meta_columns is not None:
            w = torch.tensor([info[c] for c in self.meta_columns], dtype=torch.float32)
            info['w'] = w
        # print(f"Light time: {light_time}, Spec time: {spec_time}, Spec transform time: {spec_transform_time}")
        if self.labels is not None:
            y = torch.tensor([info[label] for label in self.labels], dtype=torch.float32)
        else:
            y = light_target
        t5 = time.time()
        # print(f"Read lc: {t1-start:.4f}, read spec: {t2-t1:.4f}, spec transform: {t3-t2:.4f}, mask: {t4-t3:.4f}, target: {t5-t4:.4f}")
        return (light.float().squeeze(0), spectra.float().squeeze(0), y,
         light_target.float().squeeze(0), masked_spectra.float().squeeze(0), info)
    

class LightSpecDatasetV2(KeplerDataset):
    def __init__(self, lc_df:pd.DataFrame=None,
                 lc_data_dir:str=None,
                 spec_df:pd.DataFrame=None,
                 spec_data_dir:str=None,
                 shared_df:pd.DataFrame=None,
                 main_type:str='spectra',
                 spec_col:str='combined_obsid',
                 lc_col:str='KID',
                 light_transforms:object=None,
                 spec_transforms:object=None,
                 light_seq_len:int=13506,
                 spec_seq_len:int=4096,
                 **kwargs):
        self.lc_df = lc_df
        self.lc_data_dir = lc_data_dir
        self.spec_df = spec_df
        self.spec_data_dir = spec_data_dir
        self.shared_df = shared_df
        self.spec_col = spec_col
        self.lc_col = lc_col
        self.main_type = main_type
        self.spec_transforms = spec_transforms
        self.light_transforms = light_transforms
        self.light_seq_len = light_seq_len
        self.spec_seq_len = spec_seq_len
        super().__init__(df=lc_df,
                        transforms=light_transforms,
                         target_transforms=light_transforms,
                        seq_len=light_seq_len,
                         **kwargs)
        self.mask_transform = RandomMasking()
    def read_lamost_spectra(self, filename):
        with fits.open(filename) as hdulist:
          binaryext = hdulist[1].data
          header = hdulist[0].header
        x = binaryext['FLUX'].astype(np.float32)
        wv = binaryext['WAVELENGTH'].astype(np.float32)
        rv = header['HELIO_RV']
        meta = {'RV': rv, 'wavelength': wv}
        return x, meta
    
    def create_spectra_sample(self, spectra, meta):
        spectra_masked = spectra.copy()
        if self.spec_transforms:
            spectra, _, meta = self.spec_transforms(spectra, None, meta)
            spectra_masked, mask, _ = self.mask_transform(spectra, None, meta)
 
        if spectra_masked.shape[-1] < self.spec_seq_len:
            pad = torch.zeros(1, self.spec_seq_len - spectra_masked.shape[-1])
            spectra_masked = torch.cat([spectra_masked, pad], dim=-1)
            pad_mask = torch.zeros(1, self.spec_seq_len  - mask.shape[-1], dtype=torch.bool)
            mask = torch.cat([mask, pad_mask], dim=-1)
            pad_spectra = torch.zeros(1, self.spec_seq_len - spectra.shape[-1])
            spectra = torch.cat([spectra, pad_spectra], dim=-1)
        spectra = torch.nan_to_num(spectra, nan=0)
        spectra_masked = torch.nan_to_num(spectra_masked, nan=0)
        return spectra, spectra_masked, mask, meta        

    def __getitem__(self, idx):
        if self.main_type == 'spectra':
            spec_id = self.spec_df.iloc[idx][self.spec_col]
            spec_dir = str(spec_id)[:4]
            filepath = f"{self.spec_data_dir}/{spec_dir}/{spec_id}.fits"
            spectra, meta = self.read_lamost_spectra(filepath)
            spectra, spectra_masked, mask, info_spec = self.create_spectra_sample(spectra, meta)
            if spec_id in self.shared_df[self.spec_col].values:
                shared_idx = self.shared_df[self.shared_df[self.spec_col] == spec_id].index[0]
                lc , lc_target, _, _, info_lc, _ = super().__getitem__(shared_idx)
            else: 
                lc = torch.zeros(1, self.seq_len) if not self.use_acf else torch.zeros(2, self.seq_len)
                lc_target = torch.zeroslike(lc)
                info_lc = {'data_dir': self.data_dir}
        else:
            lc, lc_target, _, _, info_lc, _ = super().__getitem__(idx)
            lc_id = self.lc_df.iloc[idx][self.lc_col]
            if lc_id in self.shared_df[self.lc_col].values:
                shared_idx = self.shared_df[self.shared_df[self.lc_col] == lc_id].index[0]
                filepath = f"{self.lc_data_dir}/{lc_id}.fits"
                spectra, meta = self.read_lamost_spectra(filepath)
                spectra, spectra_masked, mask, info_spec = self.create_spectra_sample(spectra,meta)
            else:
                spectra = torch.zeros(1, self.spec_seq_len)
                spectra_masked = torch.zeros(1, self.spec_seq_len)
                mask = torch.zeros(1, self.spec_seq_len)
                info_spec = {}
        return (lc.float().squeeze(0), spectra.float().squeeze(0), lc_target.float().squeeze(0),
            spectra_masked.float().squeeze(0), info_lc, info_spec)

class FineTuneDataset(LightSpecDataset):
    """
    A dataset for fine-tuning lightcurve and spectra models
    """
    def __init__(self,
                labels = ['inc'],
                **kwargs
                ):
        super().__init__(**kwargs)
        self.labels = labels
    
    def __getitem__(self, idx):
        light, spectra, _, light_target, spectra_target, info = super().__getitem__(idx)
        y = torch.tensor([info[label] for label in self.labels], dtype=torch.float32)
        # print(f"Light time: {light_time}, Spec time: {spec_time}, Spec transform time: {spec_transform_time}")
        return (light, spectra, y, light_target, spectra_target, info)

def create_unique_loader(dataset, batch_size, num_workers=4, **kwargs):
    """
    Create a distributed data loader with the custom sampler
    """
    # sampler = DistributedUniqueLightCurveSampler(
    #     dataset, 
    #     batch_size=batch_size
    # )
    sampler = UniqueIDDistributedSampler(
        dataset, 
        )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        **kwargs
    )
