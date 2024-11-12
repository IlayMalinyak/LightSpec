import os
import torch
from torch.utils.data import Dataset
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from typing import List
import copy


import sys
from os import path
ROOT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print("running from ", ROOT_DIR) 

from transforms.transforms import *

mpl.rcParams['axes.linewidth'] = 4
plt.rcParams.update({'font.size': 30, 'figure.figsize': (14,10), 'lines.linewidth': 4})
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["gray", "r", "c", 'm', 'brown'])
plt.rcParams.update({'xtick.labelsize': 22, 'ytick.labelsize': 22})
plt.rcParams.update({'legend.fontsize': 22})


class DualDataset(Dataset):
    def __init__(self, data_dir,
                    labels_path,
                    range_dict,
                    labels_names=['Period', 'Inclination'],
                    lc_transforms=None,
                    spectra_transforms=None,
                ):
        self.data_dir = data_dir
        self.labels = pd.read_csv(labels_path)
        self.range_dict = range_dict
        self.labels_names = labels_names
        self.lc_dir = os.path.join(data_dir, 'lc')
        self.spectra_dir = os.path.join(data_dir, 'spectra')
        self.lc_transforms = lc_transforms
        self.spectra_transforms = spectra_transforms
        self.path_list = os.listdir(self.lc_dir)
        
        
    def __len__(self):
        return len(self.path_list)

    def _normalize(self, x, key):
        min_val = float(self.range_dict[key][0])
        max_val = float(self.range_dict[key][1])
        return (x - min_val) / (max_val - min_val)

    def __getitem__(self, idx):
        try:
            spec = pd.read_parquet(os.path.join(self.spectra_dir, f'{idx}.pqt')).values
            lc = pd.read_parquet(os.path.join(self.lc_dir, f'{idx}.pqt')).values
        except FileNotFoundError as e:
            lc = np.zeros((48000, 2))
            spec = np.zeros((3909, 1))
        spectra = spec[:,-1]
        flux = lc[:,-1]
        info_s = dict()
        info_lc = {'data_dir': self.data_dir}
        if self.spectra_transforms:
            spectra, _,info_s = self.spectra_transforms(spectra, info=info_s)
        if self.lc_transforms:
            flux,_,info_lc = self.lc_transforms(flux, info=info_lc)
        label = self.labels.iloc[idx].to_dict()
        info = {'spectra': info_s, 'lc': info_lc}
        y = torch.tensor([self._normalize(label[name], name) for name in self.labels_names], dtype=torch.float32)
        return flux.squeeze().unsqueeze(0), spectra.squeeze().unsqueeze(0), y , info

class SpectraDataset(Dataset):
    def __init__(self, data_dir, transforms=None, max_len=3909):
        self.data_dir = data_dir
        self.transforms = transforms
        files_names = os.listdir(data_dir)
        self.path_list = [os.path.join(data_dir, k) for k in files_names]
        self.max_len = max_len
        self.mask_transform = RandomMasking()
        
    def __len__(self):
        return len(self.path_list)

    def read_spectra(self, filename):
        with fits.open(filename) as hdulist:
          binaryext = hdulist[1].data
          header = hdulist[0].header
        x = binaryext['FLUX'].astype(np.float32)
        wv = binaryext['WAVELENGTH'].astype(np.float32)
        meta = dict()
        return x,wv, meta


    def __getitem__(self, idx):
        try:
            spectra, wv, meta = self.read_spectra(self.path_list[idx])
        except OSError as e:
            print("Error reading file ", self.path_list[idx])
            return torch.zeros(self.max_len), torch.zeros(self.max_len),
            torch.zeros(self.max_len, dtype=torch.bool)
        # spectra = torch.tensor(spectra, dtype=torch.float32)
        # wv = torch.tensor(wv, dtype=torch.float32).squeeze()
        # mask = torch.zeros_like(spectra)
        if self.transforms:
            spectra, _, _ = self.transforms(spectra, None, meta)
        spectra_masked, mask, _ = self.mask_transform(spectra, None, meta)
        if spectra_masked.shape[-1] < self.max_len:
            pad = torch.zeros(1, self.max_len - spectra_masked.shape[-1])
            spectra_masked = torch.cat([spectra_masked, pad], dim=-1)
            pad_mask = torch.zeros(1, self.max_len  - mask.shape[-1], dtype=torch.bool)
            mask = torch.cat([mask, pad_mask], dim=-1)
            pad_spectra = torch.zeros(1, self.max_len - spectra.shape[-1])
            spectra = torch.cat([spectra, pad_spectra], dim=-1)

        return spectra_masked.float().squeeze(0), spectra.float().squeeze(0), mask.squeeze(0)

if __name__ == '__main__':
    s_transforms = Compose([MovingAvg(7), Normalize("minmax", axis=0), ])
    lc_transforms = Compose([MovingAvg(13), Normalize("minmax", axis=0)])
    data_root = '/data/simulations/dataset'
    ds = DualDataset(data_root, os.path.join(data_root, 'simulation_properties.csv'),
     lc_transforms=lc_transforms, spectra_transforms=s_transforms)
    print(len(ds))
    for i, data in enumerate(ds): 
        flux, spectra, label = data
        time = np.linspace(0, flux.shape[0]/48, flux.shape[0])
        wv = np.load('/data/lamost/example_wv.npy')
        fig, ax = plt.subplots(2,1)
        ax[0].plot(time, flux)
        ax[1].plot(wv, spectra)
        fig.suptitle(f"Period: {label['Period']}, Inclination: {label['Inclination']}")
        plt.savefig(f'/data/lightSpec/images/dual_{i}.png')
        if i > 10:
            break
    # plt.plot(wv.squeeze(), spectra.squeeze())
    # plt.savefig('/data/lightSpec/images/transformed.png')


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
                ):
    """
    dataset for Kepler data
    Args:
        root_dir (str): root directory of the data
        path_list (List): list of paths to the data
        df (pd.DataFrame, optional): dataframe with the data. Defaults to None.
        mask_prob (float, optional): masking probability
        mask_val (float, optional): masking value. Defaults to -1.
        np_array (bool, optional): flag to load data as numpy array. Defaults to False.
        prot_df (pd.DataFrame, optional): refernce Dataframe (like McQ14). Defaults to None.
        keep_ratio (float, optional): ratio of masked values to keep. Defaults to 0.8.
        random_ratio (float, optional): ratio of masked values to convert into random numbers. Defaults to 0.2.
        uniform_bound (int, optional): bound for random numbers range. Defaults to 2.
        target_transforms (object, optional): transformations to target. Defaults to None.
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
      else:
        self.df['predicted period'] = np.nan
      

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
    row = self.df.iloc[idx]
    if 'prot' in row.keys():
        y_val = row['prot']
    elif 'Prot' in row.keys():
        y_val = row['Prot']
    else:
        y_val = row['predicted period']

    if self.npy_path is not None:
        try:
            file_path = os.path.join(self.npy_path, f"{int(row['KID'])}.npy")
            x = np.load(file_path)            
            # Check if x is a numpy array and has a length
            if not isinstance(x, np.ndarray) or x.size == 0:
                print(f"Warning: Empty or invalid numpy array for {row['KID']}")
                x = np.zeros((self.seq_len, 1))
        except FileNotFoundError:
            print(f"Error: File not found for {row['KID']}")
            x = np.zeros((self.seq_len, 1))
        except Exception as e:
            print(f"Error loading file for {row['KID']}: {str(e)}")
            x = np.zeros((self.seq_len, 1))
        
        meta = {'TEFF': None, 'RADIUS': None, 'LOGG': None, 'KMAG': None}
        return x, meta, None, y_val

    try:
        paths = row['data_file_path']
        for i in range(len(paths)):
            x, time, meta = self.read_lc(paths[i])
            x /= x.max()
            if i == 0:
                x_tot = x.copy()
            else:
                border_val = np.nanmean(x) - np.nanmean(x_tot)
                x -= border_val
                x_tot = np.concatenate((x_tot, np.array(x)))
            self.cur_len = len(x)
    except (TypeError, ValueError, FileNotFoundError, OSError) as e:
        print("Error: ", e)
        x_tot, meta = np.zeros((self.seq_len), 1), {'TEFF': None, 'RADIUS': None, 'LOGG': None, 'KMAG': None}
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
        except Exception as e:
            print(f"Error in transforms for index {idx}: {str(e)}")
            x = torch.zeros(1, self.seq_len)
            mask = torch.zeros_like(x)
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
        except Exception as e:
            print(f"Error in target transforms for index {idx}: {str(e)}")
            target = torch.zeros(1, self.seq_len)
            mask_y = torch.zeros_like(target)
    else:
        target = x.clone()
        mask_y = torch.zeros_like(target)
    if len(meta):
      info['Teff'] = meta['TEFF'] if meta['TEFF'] is not None else 0
      info['R'] = meta['RADIUS'] if meta['RADIUS'] is not None else 0
      info['logg'] = meta['LOGG'] if meta['LOGG'] is not None else 0
      info['kmag'] = meta['KMAG'] if meta['KMAG'] is not None else 0
    # info['path'] = self.df.iloc[idx]['data_file_path'] 
    info['KID'] = self.df.iloc[idx]['KID'] 
    toc = time.time()
    info['time'] = toc - tic
    if mask is None:
        mask = torch.zeros_like(x)
    if mask_y is None:
        mask_y = torch.zeros_like(target)

    # Ensure info and info_y are always dictionaries
    info = info if isinstance(info, dict) else {}
    info_y = info_y if isinstance(info_y, dict) else {}
    result = (x.float(), target.float(), mask, mask_y, info, info_y)
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
    def __init__(self, df:pd.DataFrame=None,
                prot_df:pd.DataFrame=None,
                npy_path:str=None,
                spec_path:str=None,
                light_transforms:object=None,
                spec_transforms:object=None,
                light_seq_len:int=34560,
                spec_seq_len:int=3909,
                ):
        super().__init__(df, prot_df, npy_path, light_transforms, light_seq_len)
        self.spec_path = spec_path
        self.spec_transforms = spec_transforms
        self.spec_seq_len = spec_seq_len

    def read_spectra(self, filename):
        with fits.open(filename) as hdulist:
            binaryext = hdulist[1].data
            header = hdulist[0].header
        x = binaryext['FLUX'].astype(np.float32)
        wv = binaryext['WAVELENGTH'].astype(np.float32)
        meta = dict()
        return x,wv, meta

    def __getitem__(self, idx):
        light, _, _, _, info, _ = super().__getitem__(idx)
        kid = int(info['KID'])
        obsid = int(self.df.iloc[idx]['ObsID'])
        info['obsid'] = obsid
        spectra_filename = os.path.join(self.spec_path, f'{obsid}.fits')
        try:
            spectra, wv, meta = self.read_spectra(spectra_filename)
        except OSError as e:
            print("Error reading file ", obsid, e)
            spectra = np.zeros((1, self.spec_seq_len))
            meta = dict()
        # spectra = torch.tensor(spectra, dtype=torch.float32)
        # wv = torch.tensor(wv, dtype=torch.float32).squeeze()
        # mask = torch.zeros_like(spectra)
        if self.spec_transforms:
            spectra, _, _ = self.spec_transforms(spectra, None, meta)
        if spectra.shape[-1] < self.spec_seq_len:
            spectra = F.pad(spectra, ((0, self.spec_seq_len - spectra.shape[-1],0,0)), "constant", value=0)
        return light.float(), spectra.float(), torch.zeros_like(light), torch.zeros_like(spectra), info, info