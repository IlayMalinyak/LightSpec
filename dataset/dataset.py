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
VSINI_MAX = 100


class DualDataset(Dataset):
    def __init__(self, data_dir,
                    labels_path,
                    labels_names=['Period', 'Inclination'],
                    lc_transforms=None,
                    spec_seq_len=4096,
                    spectra_transforms=None,
                    example_wv_path='/data/lamost/example_wv.npy'
                ):
        self.data_dir = data_dir
        self.labels = pd.read_csv(labels_path)
        self.labels_names = labels_names
        self.spec_seq_len = spec_seq_len
        self.range_dict = dict()
        self.update_range_dict()
        self.lc_dir = os.path.join(data_dir, 'lc')
        self.spectra_dir = os.path.join(data_dir, 'spectra')
        self.lc_transforms = lc_transforms
        self.spectra_transforms = spectra_transforms
        self.path_list = os.listdir(self.lc_dir)
        self.example_wv = np.load(example_wv_path)
        
    def update_range_dict(self):
        for name in self.labels_names:
            min_val = self.labels[name].min()
            max_val = self.labels[name].max()
            self.range_dict[name] = (min_val, max_val)
        
    def __len__(self):
        return len(self.path_list)

    def _normalize(self, x, key):
        min_val = float(self.range_dict[key][0])
        max_val = float(self.range_dict[key][1])
        return (x - min_val) / (max_val - min_val)

    def get_rv(self, row):
        r_km = row['radius'] * 6.96e5
        p_sec = row['Period'] * 24 * 3600 
        return np.sin(np.radians(row['Inclination'])) * (2*np.pi*r_km / p_sec)

    def __getitem__(self, idx):
        try:
            spec = pd.read_parquet(os.path.join(self.spectra_dir, f'{idx}.pqt')).values
            lc = pd.read_parquet(os.path.join(self.lc_dir, f'{idx}.pqt')).values
        except (FileNotFoundError, OSError) as e:
            # print("Error reading file ", idx, e)
            lc = np.zeros((48000, 2))
            spec = np.zeros((3909, 1))
        spectra = spec[:,-1]
        flux = lc[:,-1]
        info_s = dict()
        info_lc = {'data_dir': self.data_dir}
        label = self.labels.iloc[idx].to_dict()
        info_s['RV'] = self.get_rv(label)
        info_s['wavelength'] = self.example_wv
        if self.spectra_transforms:
            spectra, _,info_s = self.spectra_transforms(spectra, info=info_s)
        if self.lc_transforms:
            flux,_,info_lc = self.lc_transforms(flux, info=info_lc)
        if spectra.shape[-1] < self.spec_seq_len:
            spectra = F.pad(spectra, ((0, self.spec_seq_len - spectra.shape[-1],0,0)), "constant", value=0)
        spectra = torch.nan_to_num(spectra, nan=0)
        info = {'spectra': info_s, 'lc': info_lc}
        y = torch.tensor([self._normalize(label[name], name) for name in self.labels_names], dtype=torch.float32)
        return flux.squeeze().unsqueeze(0), spectra.squeeze().unsqueeze(0), y.squeeze() , info

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
                    id='combined_obsid'):
        self.data_dir = Path(data_dir)
        self.transforms = transforms
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
        target = torch.tensor([-999, info['Teff'] / T_sun, info['logg'], info['FeH']], dtype=torch.float32)
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
        if self.df is not None:
            if 'APOGEE' in self.id:     
                filepath = f"{self.data_dir}/aspcapStar-dr17-{self.df.iloc[idx][self.id]}.fits"
            else:
                filepath = f"{self.data_dir}/{self.df.iloc[idx][self.id]}.fits"
            target_size = 4
        else: 
            filepath = self.path_list[idx]
            target_size = self.max_len    
        obsid = os.path.basename(filepath)
        try:
            if 'APOGEE' in self.id:
                spectra, meta = self.read_apogee_spectra(filepath)
            else:
                spectra, meta = self.read_lamost_spectra(filepath)
        except (OSError, FileNotFoundError) as e:
            info = self.create_empty_info({self.id: obsid})
            # print("Error reading file ", filepath, "\n", e)
            return (torch.zeros(self.max_len),
                    torch.zeros(self.max_len),
                    torch.zeros(target_size),\
                    torch.zeros(self.max_len, dtype=torch.bool),
                    info,
                    info)
        meta[self.id] = obsid
        if self.transforms:
            spectra, _, info = self.transforms(spectra, None, meta)
        spectra_masked, mask, _ = self.mask_transform(spectra, None, info)

        if self.df is not None:
            row = self.df.iloc[idx]
            if 'APOGEE' in self.id:
                target, info = self.create_apogee_target(row, meta)
            else:
                target, info = self.create_lamost_target(row, meta)
        else:
            target = torch.zeros_like(mask)
            
        if spectra_masked.shape[-1] < self.max_len:
            pad = torch.zeros(1, self.max_len - spectra_masked.shape[-1])
            spectra_masked = torch.cat([spectra_masked, pad], dim=-1)
            pad_mask = torch.zeros(1, self.max_len  - mask.shape[-1], dtype=torch.bool)
            mask = torch.cat([mask, pad_mask], dim=-1)
            pad_spectra = torch.zeros(1, self.max_len - spectra.shape[-1])
            spectra = torch.cat([spectra, pad_spectra], dim=-1)
        spectra = torch.nan_to_num(spectra, nan=0)
        spectra_masked = torch.nan_to_num(spectra_masked, nan=0)
        
        return (spectra_masked.float().squeeze(0), spectra.float().squeeze(0),\
         target.float(), mask.squeeze(0), info, info)


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
                use_acf:bool=False
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
            except FileNotFoundError:
                print(f"Error: File not found for {row['KID']}")
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
                        'KMAG': row['kmag_abs']}
            else:
                meta = {'TEFF': None, 'RADIUS': None, 'LOGG': None, 'M': None, 'FeH':None, 'KMAG': None}
            
            if 'predicted period' in row.keys():
                meta['predicted period'] = row['predicted period']
                meta['mean_period_confidence'] = row['mean_period_confidence']
            if 'Pmax' in row.keys() and 'Pmin' in row.keys():
                meta['Pmax'] = row['Pmax']
                meta['Pmin'] = row['Pmin']
            return x, meta, None, y_val
        else:
            paths = row['data_file_path']
            for i in range(len(paths)):
                x, time, meta = self.read_lc(paths[i])
                meta['M'] = None
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


    def __getitem__(self, idx):
        tic = time.time()
        x, meta, qs, p_val = self.read_row(idx)
        x = self.fill_nan_np(x, interpolate=True)
        info = {'idx': idx}
        info['qs'] = qs
        info['period'] = p_val
        info_y = copy.deepcopy(info)
        x /= x.max()
        target = x.copy()
        mask = None
        if self.transforms is not None:
            try:
                x, mask, info = self.transforms(x, mask=None, info=info)
                if self.seq_len > x.shape[0]:
                    x = F.pad(x, ((0, 0,0, self.seq_len - x.shape[-1],)), "constant", value=0)
                    if mask is not None:
                        mask = F.pad(mask, ((0,0,0, self.seq_len - mask.shape[-1])), "constant", value=0)
                    else:
                        mask = torch.zeros_like(x)
                x = x[:self.seq_len,:].nan_to_num(0).squeeze().unsqueeze(0)
                mask = mask[:self.seq_len,:].squeeze().unsqueeze(0)
                if self.use_acf:
                    acf = torch.tensor(info['acf']).nan_to_num(0)
                    x = torch.cat((x, acf), dim=0)
            except Exception as e:
                print(f"Error in transforms for index {idx}: {str(e)}")
                # traceback.print_exc()
                x = torch.zeros(1, self.seq_len) if not self.use_acf else torch.zeros(2, self.seq_len)
                mask = torch.zeros(1, self.seq_len)
        else:
            x = torch.tensor(x)
        if self.target_transforms is not None:
            try:
                target, mask_y, info_y = self.target_transforms(target, mask=None, info=info_y)
                if self.seq_len > target.shape[0]:
                    target = F.pad(target, ((0, 0,0, self.seq_len - target.shape[-1],)), "constant", value=0)
                    if mask_y is not None:
                        mask_y = F.pad(mask_y, ((0, 0,0, self.seq_len - mask_y.shape[-1],)), "constant", value=0)
                    else:
                        mask_y = torch.zeros_like(target)
                target = target[:self.seq_len,:].nan_to_num(0).squeeze().unsqueeze(0)
                mask_y = mask_y[:self.seq_len,:].squeeze().unsqueeze(0)
                if self.use_acf:
                    acf = torch.tensor(info_y['acf']).nan_to_num(0)
                    target = torch.cat((target, acf), dim=0)
            except Exception as e:
                print(f"Error in target transforms for index {idx}: {str(e)}")
                target = torch.zeros(1, self.seq_len) if not self.use_acf else torch.zeros(2, self.seq_len)
                mask_y = torch.zeros(1, self.seq_len)
        elif 'predicted period' in meta.keys():
            target = torch.tensor([np.log10(meta['predicted period'])])
            mask_y = torch.zeros_like(target)
        elif 'Pmax' in meta.keys():
            target = torch.tensor([np.log10(meta['Pmax']), np.log10(meta['Pmin'])])
            mask_y = torch.zeros_like(target)
        else:
            target = x.clone()
            mask_y = torch.zeros_like(target)
        if len(meta):
            info['Teff'] = meta['TEFF'] if meta['TEFF'] is not None else 0
            info['R'] = meta['RADIUS'] if meta['RADIUS'] is not None else 0
            info['logg'] = meta['LOGG'] if meta['LOGG'] is not None else 0
            info['kmag_abs'] = meta['KMAG'] if meta['KMAG'] is not None else 0
            info['FeH'] = meta['FeH'] if meta['FeH'] is not None else 0
            info['Mstar'] = meta['M'] if meta['M'] is not None else 0
        info['KID'] = self.df.iloc[idx]['KID']
        if 'predicted period' in meta.keys():
            info['predicted period'] = meta['predicted period']
            info['mean_period_confidence'] = meta['mean_period_confidence']
        if 'Pmax' in meta.keys():
            info['Pmax'] = meta['Pmax']
            info['Pmin'] = meta['Pmin']
        toc = time.time()
        info['time'] = toc - tic
        if mask is None:
            mask = torch.zeros_like(x)
        if mask_y is None:
            mask_y = torch.zeros_like(target)

        # Ensure info and info_y are always dictionaries
        info = info if isinstance(info, dict) else {}
        info_y = info_y if isinstance(info_y, dict) else {}
        result = (x.float(), target.float(), target.float(), mask, info, info_y)
        if any(item is None for item in result):
            print(f"Warning: None value in result for index {idx}")
            print(f"x: {x.shape if x is not None else None}")
            print(f"target: {target.shape if target is not None else None}")
            print(f"mask: {mask.shape if mask is not None else None}")
            print(f"mask_y: {mask_y.shape if mask_y is not None else None}")
            print(f"info: {info}")
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
                spec_seq_len:int=3909,
                meta_columns = ['Teff', 'Mstar', 'logg'],
                ):
        super().__init__(df, prot_df, npy_path, light_transforms,
                light_seq_len, target_transforms=light_transforms, use_acf=use_acf)
        self.spec_path = spec_path
        self.spec_transforms = spec_transforms
        self.spec_seq_len = spec_seq_len
        self.meta_columns = meta_columns
        self.masked_transform = RandomMasking()

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
        light, light_target, _, _, info, _ = super().__getitem__(idx)
        light_time = time.time() - start
        kid = int(info['KID'])
        obsid = int(self.df.iloc[idx]['ObsID'])
        info['obsid'] = obsid
        spectra_filename = os.path.join(self.spec_path, f'{obsid}.fits')
        try:
            spectra, meta = self.read_spectra(spectra_filename)
            spec_time = time.time() - start
        except OSError as e:
            print("Error reading file ", obsid, e)
            spectra = np.zeros((self.spec_seq_len))
            meta = {'RV': 0, 'wavelength': np.zeros(self.spec_seq_len)}
        if self.spec_transforms:
            spectra, _, spec_info = self.spec_transforms(spectra, None, meta)
            spec_transform_time = time.time() - start
        if spectra.shape[-1] < self.spec_seq_len:
            spectra = F.pad(spectra, ((0, self.spec_seq_len - spectra.shape[-1],0,0)), "constant", value=0)
        spectra = torch.nan_to_num(spectra, nan=0)
        masked_spectra, _, spec_info = self.masked_transform(spectra, None, spec_info)
        info.update(spec_info)
        if self.meta_columns is not None:
            w = torch.tensor([info[c] for c in self.meta_columns], dtype=torch.float32)
            info['w'] = w
        # print(f"Light time: {light_time}, Spec time: {spec_time}, Spec transform time: {spec_transform_time}")
        return (light.float().squeeze(0), spectra.float().squeeze(0), light_target.float().squeeze(0),
         masked_spectra.float().squeeze(0), info, info)
    

class FineTuneDataset(LightSpecDataset):
    """
    A dataset for fine-tuning lightcurve and spectra models
    """
    def __init__(self, df:pd.DataFrame=None,
                npy_path:str=None,
                spec_path:str=None,
                prot_df:pd.DataFrame=None,
                light_transforms:object=None,
                spec_transforms:object=None,
                light_seq_len:int=34560,
                use_acf:bool=False,
                spec_seq_len:int=3909,
                labels = ['inc'],
                ):
        super().__init__(df, prot_df, npy_path, spec_path, light_transforms,
                spec_transforms, light_seq_len, use_acf, spec_seq_len)
        self.labels = labels
    
    def __getitem__(self, idx):
        light, spectra, light_target, spectra_target, info, _ = super().__getitem__(idx)
        row = self.df.iloc[idx]
        y = torch.tensor([row[label] for label in self.labels], dtype=torch.float32)
        # print(f"Light time: {light_time}, Spec time: {spec_time}, Spec transform time: {spec_transform_time}")
        return (light, spectra, light_target, spectra_target, y, info)

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
