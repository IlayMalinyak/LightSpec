import numpy as np
import torch
import time
from scipy.signal import savgol_filter as savgol
import torch.nn.functional as F


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
        for t in self.transforms:
            out, mask, info = t(out, mask=mask, info=info)
            # print(f"{t} took {time.time() - t0}")
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
        if self.random_high is None:
            self.random_high = x.max()
        if mask is None:
            mask = np.random.rand(*x.shape) < self.mask_prob
        
        # Create a copy of x to modify
        masked_x = x.copy()
        
        # Replace with mask_value
        replace_mask = mask & (np.random.rand(*x.shape) < self.replace_prob)
        masked_x[replace_mask] = self.mask_value
        
        # Replace with random values
        random_mask = mask & ~replace_mask
        masked_x[random_mask] = np.random.uniform(self.random_low, self.random_high, size=random_mask.sum())
        
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
    Supported schemes: 'std' (standardization), 'minmax', 'median'
    """
    def __init__(self, scheme='std', axis=None):
        """
        Initialize the Normalize transformation.

        :param scheme: Normalization scheme ('std', 'minmax', or 'median')
        :param axis: Axis or axes along which to normalize. None for global normalization.
        """
        self.scheme = scheme.lower()
        self.axis = axis
        assert self.scheme in ['std', 'minmax', 'median'], "Unsupported normalization scheme"

    def __call__(self, x, mask=None, info=dict()):
        info['normalize'] = self.scheme
        if isinstance(x, np.ndarray):
            return self._normalize_numpy(x, mask, info)
        elif isinstance(x, torch.Tensor):
            return self._normalize_torch(x, mask, info)
        else:
            raise TypeError("Input must be either a numpy array or a PyTorch tensor")

    def _normalize_numpy(self, x, mask=None, info=dict()):
        if mask is None:
            mask = np.zeros_like(x, dtype=bool)
        x_masked = x[~mask]
        if self.scheme == 'std':
            mean = np.mean(x_masked, axis=self.axis, keepdims=True)
            std = np.std(x_masked, axis=self.axis, keepdims=True)
            return (x - mean) / (std + 1e-8), mask, info
        elif self.scheme == 'minmax':
            min_val = np.min(x_masked, axis=self.axis, keepdims=True)
            max_val = np.max(x_masked, axis=self.axis, keepdims=True)
            return (x - min_val) / (max_val - min_val + 1e-8), mask, info
        elif self.scheme == 'median':
            median = np.median(x_masked, axis=self.axis, keepdims=True)
            return x / median, mask, info

    def _normalize_torch(self, x, mask=None, info=dict()):
        if mask is None:
            mask = torch.zeros_like(x, dtype=torch.bool)
        x_masked = x[~mask]
        if self.scheme == 'std':
            mean = torch.mean(x_masked, dim=self.axis, keepdim=True)
            std = torch.std(x_masked, dim=self.axis, keepdim=True)
            return (x - mean) / (std + 1e-8), mask, info
        elif self.scheme == 'minmax':
            min_val = torch.min(x_masked, dim=self.axis, keepdim=True)[0]
            max_val = torch.max(x_masked, dim=self.axis, keepdim=True)[0]
            return (x - min_val) / (max_val - min_val + 1e-8), mask, info
        elif self.scheme == 'median':
            median = torch.median(x, dim=self.axis, keepdim=True)[0]
            return x / median, mask, info

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
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
    
        # Calculate the moving average
        weights = np.ones(self.kernel_size) / self.kernel_size
        ma = np.convolve(x, weights, mode='same')
        
        # Handle edge effects
        half_window = self.kernel_size // 2
        ma[:half_window] = ma[half_window]
        ma[-half_window:] = ma[-half_window-1]
        
        # Subtract moving average from the original series
        return (x - ma)[:, np.newaxis], mask, info

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
