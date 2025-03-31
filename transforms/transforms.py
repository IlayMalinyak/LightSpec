import numpy as np
import torch
import time
from scipy.signal import savgol_filter as savgol
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
from scipy.signal import medfilt
from scipy.optimize import curve_fit
import os
from statsmodels.tsa.stattools import acf as A




class Compose:
    """Composes several transforms together. 
    Adapted from https://pytorch.org/vision/master/_modules/torchvision/transforms/transforms.html#Compose

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, mask=None, info=dict(), step=None):
        new_info = info.copy() if info else {}
        if len(x.shape) == 1:
                x = x[:, np.newaxis]
        out = x
        t0 = time.time()
        # print(f"Initial type: {out.dtype}")
        for t in self.transforms:
            out, mask, info = t(out, mask=mask, info=info)
            # print(f"{t}  shape: {out.shape}")
            # if mask is not None:
            #     print("mask shape: ", mask.shape)
            # else:
            #     print("mask is None")
        return out, mask, info

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string

class Crop:
    """
    Crop the input to a specified size.
    """
    def __init__(self, crop_size, start=0):
        self.crop_size = crop_size
        self.start = start

    def __call__(self, x, mask=None, info=dict()):
        if isinstance(x, np.ndarray):
            return self._crop_numpy(x, mask=mask, info=info)
        elif isinstance(x, torch.Tensor):
            return self._crop_torch(x, mask=mask, info=info)
        else:
            raise TypeError("Input must be either a numpy array or a PyTorch tensor")

    def _crop_numpy(self, x, mask=None, info=dict()):
        if x.shape[-1] <= self.crop_size:
            return x, mask, info
        x = x[self.start:self.start + self.crop_size,...]
        if mask is not None:
            mask = mask[self.start:self.start + self.crop_size, ...]
        return x, mask, info

    def _crop_torch(self, x, mask=None, info=dict()):
        if x.size(-1) <= self.crop_size:
            return x, mask, info
        start = x.size(-1) // 2 - self.crop_size // 2
        x = x[start:start + self.crop_size, ...]
        if mask is not None:
            mask = mask[start:start + self.crop_size, ...]
        return x, mask, info

    def __repr__(self):
        return f"Crop(crop_size={self.crop_size})"

class RandomCrop:
    """
    Randomly crop the input to a specified size.
    """
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, x, mask=None, info=dict()):
        if isinstance(x, np.ndarray):
            return self._crop_numpy(x, mask=mask, info=info)
        elif isinstance(x, torch.Tensor):
            return self._crop_torch(x, mask=mask, info=info)
        else:
            raise TypeError("Input must be either a numpy array or a PyTorch tensor")

    def _crop_numpy(self, x, mask=None, info=dict()):
        if x.shape[0] <= self.crop_size:
            return x, mask, info
        start = np.random.randint(0, x.shape[0] - self.crop_size)
        info['crop_start'] = start
        x = x[start:start + self.crop_size,...]
        if mask is not None:
            mask = mask[start:start + self.crop_size,...]
        return x, mask, info

    def _crop_torch(self, x, mask=None, info=dict()):
        if x.size(0) <= self.crop_size:
            return x
        start = torch.randint(0, x.size(0) - self.crop_size, (1,))
        info['crop_start'] = start.item()
        x = x[start:start + self.crop_size,...]
        if mask is not None:
            mask = mask[start:start + self.crop_size,...]
        return x, mask, info

    def __repr__(self):
        return f"RandomCrop(crop_size={self.crop_size})"

class TrendRemover:
    """
    Remove long-term trends using moving average transformation.
    """
    def __init__(self, window_size=100*48):
        self.window_size = window_size

    def __call__(self, x, mask=None, info=dict()):
        if isinstance(x, np.ndarray):
            return self._remove_trend_numpy(x, mask=mask, info=info)
        elif isinstance(x, torch.Tensor):
            return self._remove_trend_torch(x, mask=mask, info=info)
        else:
            raise TypeError("Input must be either a numpy array or a PyTorch tensor")

    def _remove_trend_numpy(self, x, mask=None, info=dict()):
        # Compute moving average
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        moving_avg = (cumsum[self.window_size:] - cumsum[:-self.window_size]) / float(self.window_size)
        
        # Pad the moving average to match original length
        pad_left = np.mean(moving_avg[:self.window_size//2])
        pad_right = np.mean(moving_avg[-self.window_size//2:])
        moving_avg = np.pad(moving_avg, 
                             (self.window_size//2, self.window_size - self.window_size//2 - 1), 
                             mode='constant', 
                             constant_values=(pad_left, pad_right))
        
        # Remove trend
        detrended = x - moving_avg + 1
        print(np.isnan(detrended).sum(), moving_avg)
        return detrended, mask, info

    def _remove_trend_torch(self, x, mask=None, info=dict()):
        # Convert to numpy, process, then convert back to torch
        x_np = x.numpy() if torch.is_tensor(x) else x
        detrended_np, mask, info = self._remove_trend_numpy(x_np, mask, info)
        return torch.tensor(detrended_np), mask, info


class MovingAvg():
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride=1):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding_left = kernel_size // 2
        self.padding_right = (kernel_size - 1) // 2

    def __call__(self, x, mask=None, info=dict()):
        if isinstance(x, np.ndarray):
            x = savgol(x, self.kernel_size, 1, mode='mirror', axis=0)
            return x,mask, info
            
        elif isinstance(x, torch.Tensor):
            # Check if x is 1D, if so add batch and channel dimensions
            if x.dim() == 1:
                x = x.unsqueeze(0).unsqueeze(0)
            elif x.dim() == 2:
                x = x.unsqueeze(1)
            # Apply moving average
            x = F.pad(x, (self.padding_left, self.padding_right))
            x = F.avg_pool1d(x, kernel_size=self.kernel_size, stride=self.stride)

            
            # Remove added dimensions if they were added
            if x.size(0) == 1 and x.size(1) == 1:
                x = x.squeeze(0).squeeze(0)
            elif x.size(1) == 1:
                x = x.squeeze(1)
            
            return x, mask, info
        
        else:
            raise TypeError("Input must be either a numpy array or a PyTorch tensor")

    def __repr__(self):
        return f"moving_avg(kernel_size={self.kernel_size}, stride={self.stride})"
    

class RandomMasking:
    """
    Randomly mask elements in the input for self-supervised learning tasks.
    Some masked elements are replaced with a predefined value, others with random numbers.
    """
    def __init__(self, mask_prob=0.15, replace_prob=0.8, mask_value=0, 
                 random_low=0, random_high=None):
        """
        Initialize the RandomMasking transformation.

        :param mask_prob: Probability of masking an element
        :param replace_prob: Probability of replacing a masked element with mask_value
        :param random_prob: Probability of replacing a masked element with a random value
        :param mask_value: The value to use for masking
        :param random_low: Lower bound for random replacement (inclusive)
        :param random_high: Upper bound for random replacement (exclusive)
        """
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob
        self.mask_value = mask_value
        self.random_low = random_low
        self.random_high = random_high

        assert 0 <= mask_prob <= 1, "mask_prob must be between 0 and 1"
        assert 0 <= replace_prob <= 1, "replace_prob must be between 0 and 1"

    def __call__(self, x, mask=None, info=dict()):
        if isinstance(x, np.ndarray):
            return self._mask_numpy(x, mask=mask, info=info)
        elif isinstance(x, torch.Tensor):
            return self._mask_torch(x, mask=mask, info=info)
        else:
            raise TypeError("Input must be either a numpy array or a PyTorch tensor")

    def _mask_numpy(self, x, mask=None, info=dict()):
        if len(x.shape) == 1:
            x = x[np.newaxis, :]
        if self.random_high is None:
            self.random_high = x.max()
        mask = np.random.rand(*x[0].shape) < self.mask_prob
        
        # Create a copy of x to modify
        masked_x = x.copy()
        
        # Replace with mask_value
        replace_mask = mask & (np.random.rand(*x[0].shape) < self.replace_prob)
        masked_x[:, replace_mask] = self.mask_value
        
        # Replace with random values
        random_mask = mask & ~replace_mask
        masked_x[:, random_mask] = np.random.uniform(self.random_low, self.random_high, size=random_mask.sum())
        
        return masked_x, mask, info

    def _mask_torch(self, x, mask=None, info=dict()):
        if self.random_high is None:
            self.random_high = x.max()
        if mask is None:
            mask = torch.rand_like(x) < self.mask_prob
        
        # Create a copy of x to modify
        masked_x = x.clone()
        
        # Replace with mask_value
        replace_mask = mask & (torch.rand_like(x) < self.replace_prob)
        masked_x[replace_mask] = self.mask_value
        
        # Replace with random values
        random_mask = mask & ~replace_mask
        masked_x[random_mask] = torch.rand_like(x[random_mask]) * (self.random_high - self.random_low) + self.random_low
        
        return masked_x, mask, info

    def __repr__(self):
        return (f"RandomMasking(mask_prob={self.mask_prob}, replace_prob={self.replace_prob}, "
                f"mask_value={self.mask_value}, "
                f"random_low={self.random_low}, random_high={self.random_high})")

class Normalize:
    """
    Normalize the input data according to a specified scheme.
    Supported schemes: 'std' (standardization), 'minmax', 'median', 'dist'
    """
    def __init__(self, scheme=['std'], axis=0, max_std=203429):
        """
        Initialize the Normalize transformation.

        :param scheme: Normalization scheme ('std', 'minmax', 'dist', 'mag' or 'median')
        :param axis: Axis or axes along which to normalize. None for global normalization.
        """
        self.axis = axis
        self.scheme = scheme
        self.max_std = max_std
        
    def __call__(self, x, mask=None, info=dict()):
        x_total = []
        for i, s in enumerate(self.scheme):
            info[f'normalize {i}'] = s
            if isinstance(x, np.ndarray):
                x_norm, mask, info = self._normalize_numpy(x.squeeze(), s,  mask, info)
                x_total.append(np.expand_dims(x_norm, axis=self.axis))
            elif isinstance(x, torch.Tensor):
                x_norm, mask, info = self._normalize_torch(x.squeeze(), s, mask, info)
                x_total.append(x_norm.unsqueeze(self.axis))
        if isinstance(x, np.ndarray):
            x_total = np.concatenate(x_total, axis=self.axis)
        else:
            x_total = torch.cat(x_total, dim=self.axis)
        return x_total, mask, info

    def _normalize_numpy(self, x, scheme, mask=None, info=dict()):
        if mask is None:
            mask = np.zeros_like(x, dtype=bool)
        else:
            mask = mask.squeeze()
        x_masked = x[~mask]
        if scheme == 'std':
            mean = np.mean(x_masked, axis=self.axis, keepdims=True)
            std = np.std(x_masked, axis=self.axis, keepdims=True)
            x =  (x - mean) / (std + 1e-8)
        elif scheme == 'max_std':
            x = x / (x.std() + 1e-8) * (1 / np.log(self.max_std / np.std(x)) + 1) 
        elif scheme == 'minmax':
            min_val = np.min(x_masked, axis=self.axis, keepdims=True)
            max_val = np.max(x_masked, axis=self.axis, keepdims=True)
            x =  (x - min_val) / (max_val - min_val + 1e-8)
        elif scheme == 'median':
            median = np.nanmedian(x_masked + 1e-3, axis=self.axis, keepdims=True)
            x =  x / median
        elif scheme == 'dist':
            d = info['Dist']
            assert d is not None, "Distance must be provided for 'dist' normalization"
            x = x * d ** 2
            x = x - x.mean()    # Subtract mean
        elif scheme == 'dist_median':
            d = info['Dist']
            assert d is not None, "Distance must be provided for 'dist' normalization"
            x = x * d ** 2
            x = x / np.nanmedian(x + 1e-3)
        elif scheme == 'dist_std':
            d = info['Dist']
            assert d is not None, "Distance must be provided for 'dist' normalization"
            x = x * d ** 2
            x = (x - x.mean()) / (x.std() + 1e-8)
        elif scheme == 'mag':
            k = info['KMAG']
            assert k is not None, "KMAG must be provided for 'mag' normalization"
            k_inv = 2**(-k)
            x = x / k_inv
        elif scheme == 'kmag_median':
            k = info['KMAG']
            assert k is not None, "KMAG must be provided for 'mag' normalization"
            k_inv = 2**(-k)
            x = x / k_inv
            x = x / np.nanmedian(x + 1e-3)
        return x, mask, info

    def _normalize_torch(self, x, scheme, mask=None, info=dict()):
        if mask is None:
            mask = torch.zeros_like(x, dtype=torch.bool)
        x_masked = x[~mask]
        if scheme == 'std':
            mean = torch.mean(x_masked, dim=self.axis, keepdim=True)
            std = torch.std(x_masked, dim=self.axis, keepdim=True)
            x =  (x - mean) / (std + 1e-8)
        elif scheme == 'minmax':
            min_val = torch.min(x_masked, dim=self.axis, keepdim=True)[0]
            max_val = torch.max(x_masked, dim=self.axis, keepdim=True)[0]
            x =  (x - min_val) / (max_val - min_val + 1e-8)
        elif scheme == 'median':
            median = torch.median(x, dim=self.axis, keepdim=True)[0]
            x  = x / median 
        return x, mask, info

    def __repr__(self):
        return f"Normalize(scheme='{self.scheme}', axis={self.axis})"


class AvgDetrend:
    """
    Detrend the input data using a moving average filter.
    """
    def __init__(self, kernel_size, polyorder=1):
        """
        Initialize the AvgDetrend transformation.

        :param kernel_size: Size of the moving average filter
        :param polyorder: Order of the polynomial to fit
        """
        self.kernel_size = kernel_size
        self.polyorder = polyorder

    def __call__(self, x, mask=None, info=dict()):
        if isinstance(x, np.ndarray):
            return self._detrend_numpy(x, mask, info)
        elif isinstance(x, torch.Tensor):
            return self._detrend_torch(x, mask, info)
        else:
            raise TypeError("Input must be either a numpy array or a PyTorch tensor")

    def _detrend_numpy(self, x, mask=None, info=dict()):
        if len(x.shape) > 1:
            x = x.squeeze()
        x = x.astype(np.float64)
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
    
        # Calculate the moving average
        weights = np.ones(self.kernel_size) / self.kernel_size
        ma = np.convolve(x, weights, mode='same')
        # Handle edge effects
        half_window = self.kernel_size // 2
        ma[:half_window] = ma[half_window]
        ma[-half_window:] = ma[-half_window-1]
        res = (x - ma) + 1
        # Subtract moving average from the original series
        return res, mask, info

    def _detrend_torch(self, x, mask=None, info=dict()):
        if mask is None:
            mask = torch.zeros_like(x, dtype=torch.bool)
        ma = torch.nn.functional.avg_pool1d(x.unsqueeze(0).unsqueeze(0), self.kernel_size,
         stride=1, padding=self.kernel_size//2)
        return x - ma.squeeze(), mask, info

    def __repr__(self):
        return f"AvgDetrend(kernel_size={self.kernel_size}, polyorder={self.polyorder})"

class ToTensor():
    def __init__(self):
        pass
    def __call__(self, x, mask=None, info=None, step=None):
        x = torch.tensor(x)
        if mask is not None:
           mask = torch.tensor(mask)
        return x, mask, info
    def __repr__(self):
        return "ToTensor"

class Shuffle():
    def __init__(self, segment_len=48*90, seed=1234):
        self.segment_len = segment_len
        self.seed = seed
    def __call__(self, x, mask=None, info=None, step=None):
        if isinstance(x, np.ndarray):
            np.random.seed(self.seed)
            x = x / np.nanmedian(x)
            num_segments = int(np.ceil(len(x) / self.segment_len))
            x_segments = np.array_split(x, num_segments)
            np.random.shuffle(x_segments)
            x = np.concatenate(x_segments)
            if mask is not None:
                mask_segments = np.array_split(mask, num_segments)
                np.random.shuffle(mask_segments)
                mask = np.concatenate(mask_segments)
            t = 1000 * time.time()
            np.random.seed(int(t) % 2**32)
        else:
            raise NotImplementedError
        return x, mask, info
    def __repr__(self):
        return f"Shuffle(seg_len={self.segment_len})"

class Identity():
    def __init__(self):
        pass
    def __call__(self, x, mask=None, info=None, step=None):
        return x, mask, info
    def __repr__(self):
        return "Identity"

class AddGaussianNoise(object):
    def __init__(self, sigma=1.0, exclude_mask=False, mask_only=False):
        self.sigma = sigma
        self.exclude_mask = exclude_mask
        self.mask_only = mask_only
        assert not (exclude_mask and mask_only)

    def __call__(self, x, mask=None, info=None, step=None):
        exclude_mask = None
        if mask is not None:
            if self.exclude_mask:
                exclude_mask = mask
            elif self.mask_only:
                exclude_mask = ~mask
        if isinstance(x, np.ndarray):
            out = self.add_gaussian_noise_np(
                x, self.sigma, mask=exclude_mask)
        else:
            out = self.add_gaussian_noise_torch(
                x, self.sigma, mask=exclude_mask)
        return out, mask, info
    
    def add_gaussian_noise_np(self, x: np.ndarray, sigma: float, mask: np.ndarray = None):
        out = x.copy()
        if mask is None:
            out = np.random.normal(x, sigma).astype(out.dtype)
        else:
            out[~mask] = np.random.normal(
                out[~mask], sigma).astype(dtype=out.dtype)
        return out
    
    def add_gaussian_noise_torch(self, x: torch.Tensor, sigma: float, mask: torch.Tensor = None):
        out = x.clone()
        if mask is None:
            out += torch.randn_like(out) * sigma
        else:
            out[~mask] += torch.randn_like(out[~mask]) * sigma
        return out
    
class RandomTransform():
    def __init__(self, transforms, p=None):
        self.transforms = transforms
        self.p = p
    def __call__(self, x, mask=None, info=None):
        t = np.random.choice(self.transforms, p=self.p)
        if 'random_transform' in info:
            info['random_transform'].append(str(t))
        else:
            info['random_transform'] = [str(t)]
        x, mask, info = t(x, mask=mask, info=info)
        return x, mask, info
    def __repr__(self):
        return f"RandomTransform(p={self.p}"
    

class FillNans():
    """
    Fill NaN values in the input data.
    """
    def __init__(self, interpolate=False):
        self.interpolate = interpolate
    def __call__(self, x, mask=None, info=None, step=None):
        if isinstance(x, np.ndarray):
            x = self.fill_nan_np(x, interpolate=self.interpolate)
        else:
            raise NotImplementedError
        
        print("nans: ", np.sum(np.isnan(x)))
        return x, mask, info
    
    def fill_nan_np(self, x:np.ndarray, interpolate:bool=True):
        print("shape: ", x.shape)
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
    def __repr__(self):
        return f"FillNans(interpolate={self.interpolate})"
    

class SmoothSpectraGaussian():
    def __init__(self, sigma=35):
        self.sigma = sigma

    def __call__(self, x, mask=None, info=None):
        if isinstance(x, np.ndarray):
            gaussian_weight_matrix = np.exp(-0.5*(x[:,None]-x[None,:])**2/(self.sigma**2))
            smoothed_flux = np.sum(gaussian_weight_matrix*x, axis=1)/np.sum(gaussian_weight_matrix, axis=1)
            x = x / smoothed_flux
        else:
            raise NotImplementedError
        return x, mask, info


class LAMOSTSpectrumPreprocessor:
    """
    Preprocessing class for LAMOST spectra implementing wavelength correction, 
    resampling, denoising, continuum normalization, and secondary normalization.
    
    Follows preprocessing steps from
    "Estimating stellar parameters from LAMOST low-resolution spectra", X. Li, B. Lin.
    """
    def __init__(self, 
                 blue_wavelength_range=(3841, 5800),
                 red_wavelength_range=(5800, 8798),
                 resample_step=0.0001,
                 median_filter_size=3,
                 polynomial_order=5,
                 rv_norm=True,
                 continuum_norm=True,
                 plot_steps=False):
        """
        Initialize preprocessing parameters.
        
        Args:
            blue_wavelength_range (tuple): Wavelength range for blue end
            red_wavelength_range (tuple): Wavelength range for red end
            resample_step (float): Logarithmic resampling step size
            median_filter_size (int): Size of median filter window
            polynomial_order (int): Order of polynomial for continuum estimation
        """
        self.blue_range = blue_wavelength_range
        self.red_range = red_wavelength_range
        self.resample_step = resample_step
        self.median_filter_size = median_filter_size
        self.polynomial_order = polynomial_order
        self.rv_norm = rv_norm
        self.continuum_norm = continuum_norm
        self.plot_steps = plot_steps



    def __call__(self, spectrum, mask=None, info=dict()):
        """
        Apply full preprocessing pipeline to input spectrum.
        
        Args:
            spectrum (np.ndarray or torch.Tensor): Input spectrum flux values
            wavelength (np.ndarray): Original wavelength array
            radial_velocity (float): Radial velocity for wavelength correction
        
        Returns:
            Preprocessed spectrum
        """
        # Convert to numpy if torch tensor
        if torch.is_tensor(spectrum):
            spectrum = spectrum.numpy()
        if self.rv_norm:
            radial_velocity = info['RV']
        wavelength = info['wavelength']

        if self.plot_steps:
            fig, ax = plt.subplots(6, 2, figsize=(60, 36), gridspec_kw={'height_ratios': [1, 1, 1, 1, 1, 1]})

            merged_ax = fig.add_subplot(611)  # Spans across both columns
            merged_ax.plot(wavelength.squeeze(), spectrum.squeeze())
            merged_ax.set_title("Original Spectrum")
        
        # 1. Wavelength Correction
        if self.rv_norm:
            corrected_wavelength = self._wavelength_correction(wavelength, radial_velocity)
        else:
            corrected_wavelength = wavelength
        info['corrected_wavelength'] = corrected_wavelength

        
        # Separate blue and red ends
        blue_mask = (corrected_wavelength >= self.blue_range[0]) & (corrected_wavelength <= self.blue_range[1])
        red_mask = (corrected_wavelength >= self.red_range[0]) & (corrected_wavelength <= self.red_range[1])
        
        blue_wavelength = corrected_wavelength[blue_mask]
        red_wavelength = corrected_wavelength[red_mask]
        info['blue_corrected_wavelength'] = blue_wavelength
        info['red_corrected_wavelength'] = red_wavelength

        
        blue_spectrum = spectrum[blue_mask].squeeze()
        red_spectrum = spectrum[red_mask].squeeze()

        if self.plot_steps:
            ax[1,0].plot(blue_wavelength, blue_spectrum, label='Blue Spectrum')
            ax[1,1].plot(red_wavelength, red_spectrum, label='Red Spectrum')
            ax[1, 0].set_title("WV Correction blue", loc='center')
            ax[1, 1].set_title("WV Correction red", loc='center')

            
        # 2. Linear Interpolation Resampling (Separately)
        blue_resampled = self._linear_interpolation_resample(blue_spectrum, blue_wavelength, is_blue=True)
        red_resampled = self._linear_interpolation_resample(red_spectrum, red_wavelength, is_blue=False)

        if self.plot_steps:
            ax[2,0].plot(np.arange(len(blue_resampled)), blue_resampled, label='Blue Resampled')
            ax[2,1].plot(np.arange(len(red_resampled)), red_resampled, label='Red Resampled')
            ax[2, 0].set_title("Linear Interpolation Resampling blue")
            ax[2, 1].set_title("Linear Interpolation Resampling red")

        
        # 3. Denoising (Median Filtering)
        blue_denoised = self._median_filter_denoise(blue_resampled)
        red_denoised = self._median_filter_denoise(red_resampled)

        if self.plot_steps:
            ax[3,0].plot(np.arange(len(blue_denoised)), blue_denoised, label='Blue Denoised')
            ax[3,1].plot(np.arange(len(red_denoised)), red_denoised, label='Red Denoised')
            ax[3, 0].set_title("Median Filtering Denoising blue")
            ax[3, 1].set_title("Median Filtering Denoising red")
        
        if self.continuum_norm:
            # 4. Continuum Normalization (Separately)
            blue_normalized = self._continuum_normalization(blue_denoised, is_blue=True)
            red_normalized = self._continuum_normalization(red_denoised, is_blue=False)
            # blue_final_norm = self._secondary_normalization(blue_normalized)
            # red_final_norm = self._secondary_normalization(red_normalized)
            # final_norm = np.concatenate([blue_final, red_final])[None,:]
            # info['x_norm'] = final_norm
            if self.plot_steps:
                ax[4,0].plot(np.arange(len(blue_normalized)), blue_normalized, label='Blue Normalized')
                ax[4,1].plot(np.arange(len(red_normalized)), red_normalized, label='Red Normalized')
                ax[4, 0].set_title("Continuum Normalization blue")
                ax[4, 1].set_title("Continuum Normalization red")
        else:
            blue_normalized = blue_denoised
            red_normalized = red_denoised

        
        # 5. Secondary Denoising and Normalization
        blue_final = self._secondary_normalization(blue_normalized)
        red_final = self._secondary_normalization(red_normalized)

        if self.plot_steps:
            ax[5,0].plot(np.arange(len(blue_final)), blue_final, label='Blue Final')
            ax[5,1].plot(np.arange(len(red_final)), red_final, label='Red Final')
            ax[5, 0].set_title("Secondary Normalization blue")
            ax[5, 1].set_title("Secondary Normalization red")
            fig.suptitle(f"LAMOST Spectrum Preprocessing - {info['obsid']}")
            plt.tight_layout()
            plt.savefig(f'/data/lightSpec/images/lamost_{info["obsid"]}_preprocessing.png')
        
        return np.concatenate([blue_final, red_final])[None,:], mask, info

    def _wavelength_correction(self, wavelength, radial_velocity):
        """
        Correct wavelength based on radial velocity.
        
        λ′ = λ * (1 + RV/c)
        """
        c = 299792.458  # Speed of light in km/s
        return wavelength * (1 + radial_velocity / c)

    def _linear_interpolation_resample(self, spectrum, wavelength, is_blue=True):
        """
        Resample spectrum using linear interpolation in logarithmic space.
        """
        # Determine wavelength range based on whether it's blue or red end
        wave_range = self.blue_range if is_blue else self.red_range
        
        # Create logarithmic wavelength grid
        log_wave_start = np.log10(wave_range[0])
        log_wave_end = np.log10(wave_range[1])
        new_log_wavelengths = np.arange(
            log_wave_start, 
            log_wave_end, 
            self.resample_step
        )
        new_wavelengths = 10 ** new_log_wavelengths
        
        # Interpolate spectrum
        interpolator = interpolate.interp1d(
            wavelength, 
            spectrum, 
            kind='linear', 
            fill_value='extrapolate'
        )
        resampled_spectrum = interpolator(new_wavelengths)
        
        return resampled_spectrum

    def _median_filter_denoise(self, spectrum):
        """
        Apply median filtering for noise reduction.
        """
        return medfilt(spectrum, kernel_size=self.median_filter_size)

    def _continuum_normalization(self, spectrum, is_blue=True):
        """
        Estimate and normalize continuum using polynomial fitting.
        """
        x = np.arange(len(spectrum))
        poly_coeffs = np.polyfit(x, spectrum, deg=self.polynomial_order)
        continuum = np.polyval(poly_coeffs, x)
        
        # Normalize by dividing spectrum by its estimated continuum
        normalized_spectrum = spectrum / continuum
        
        return normalized_spectrum

    def _secondary_normalization(self, spectrum):
        """
        Secondary normalization with outlier replacement and z-score transformation.
        """
        mu = np.mean(spectrum)
        sigma = np.std(spectrum)
        
        # Replace outliers
        spectrum = np.where(
            (spectrum < mu - 3*sigma) | (spectrum > mu + 3*sigma), 
            mu, 
            spectrum
        )
        
        # Z-score transformation
        normalized_spectrum = (spectrum - mu) / sigma
        
        return normalized_spectrum

    def __repr__(self):
        return (f"LAMOSTSpectrumPreprocessor("
                f"blue_range={self.blue_range}, "
                f"red_range={self.red_range}, "
                f"resample_step={self.resample_step})")

class ACF():
    def __init__(self, max_lag_day=None, day_cadence=1/48,
                   max_len=None):
        self.max_lag_day = max_lag_day
        self.day_cadence = day_cadence
        self.max_len = max_len

    def __call__(self, x, mask=None, info=None, step=None):
        if isinstance(x, np.ndarray):
            if len(x.shape) > 1:
                x = x.squeeze()
            if self.max_lag_day is None:
                acf = A(x, nlags=len(x))
            else:
                acf = A(x, nlags=self.max_lag_day/self.day_cadence - 1)
            # if mask is not None:
                # acf[mask] = np.nan
            if self.max_len is not None and (len(acf) < self.max_len):
                acf = np.pad(acf, ((0, self.max_len - len(acf))))
            acf = (acf - acf.mean()) / acf.std()
            info['acf'] = acf[None]

        else:
            raise NotImplementedError
        return x, mask, info
    def __repr__(self):
        return f"ACF(max_lag={self.max_lag_day})"

class FFT():
    def __init__(self, use_magnitude=True, log_scale=True, seq_len=None):
        self.use_magnitude = use_magnitude
        self.log_scale = log_scale
        self.seq_len = seq_len
        
    def __call__(self, x, mask=None, info=None):
        if isinstance(x, np.ndarray):
            # Compute FFT
            fft_result = np.fft.rfft(x.squeeze())
            
            # Use magnitude or power spectrum
            if self.use_magnitude:
                freq_features = np.abs(fft_result)
            else:
                freq_features = np.abs(fft_result)**2
                
            # Apply log scaling if enabled
            if self.log_scale:
                freq_features = np.log1p(freq_features)
            
            # Ensure output length matches input by using interpolation
            target_length = self.seq_len or len(x)
            current_length = len(freq_features)
            
            if current_length != target_length:
                # Interpolate to match original array length
                positions = np.linspace(0, current_length - 1, target_length)
                freq_features = np.interp(positions, np.arange(current_length), freq_features)
            
            # Store frequency information
            if info is not None:
                info['fft_freqs'] = np.fft.rfftfreq(len(x.squeeze()))
                info['fft'] = freq_features[None]
            
            # # Combine with original features
            # x_with_freq = np.hstack((freq_features[:, None], x))
            #
            return x, mask, info
        else:
            raise NotImplementedError
