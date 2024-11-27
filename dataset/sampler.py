import torch
from tqdm import tqdm
import torch
import random
from tqdm import tqdm
from torch.utils.data import Sampler, Subset
import matplotlib.pyplot as plt


class DistributedUniqueLightCurveSampler(Sampler):
    def __init__(self, dataset, batch_size, num_replicas=None, rank=None):
        if num_replicas is None:
            num_replicas = torch.distributed.get_world_size() if torch.distributed.is_available() else 1
        if rank is None:
            rank = torch.distributed.get_rank() if torch.distributed.is_available() else 0
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        
        # Handle Subset specifically
        if isinstance(dataset, Subset):
            self.original_indices = dataset.indices
            self.base_dataset = dataset.dataset
        else:
            self.original_indices = list(range(len(dataset)))
            self.base_dataset = dataset
        
        # Lazy initialization of light curve tracking
        self._light_curve_indices = None
        
    def _build_light_curve_index(self):
        """
        Efficiently build light curve index without loading entire dataset
        Uses a generator approach to minimize memory usage
        """
        # Handle both full dataset and subset
        if hasattr(self.base_dataset, 'df'):
            dataframe = self.base_dataset.df
        else:
            raise AttributeError("Could not find a dataframe for indexing")
        
        # Use a dictionary with list comprehension for memory efficiency
        light_curve_mapping = {}
        
        # Iterate through subset indices
        for local_idx, original_idx in enumerate(self.original_indices):
            row = dataframe.iloc[original_idx]
            light_curve_id = int(row['KID'])
            
            if light_curve_id not in light_curve_mapping:
                light_curve_mapping[light_curve_id] = []
            
            light_curve_mapping[light_curve_id].append(local_idx)
        
        return light_curve_mapping
    
    @property
    def light_curve_indices(self):
        """
        Lazy-loaded light curve indices
        """
        if self._light_curve_indices is None:
            self._light_curve_indices = self._build_light_curve_index()
        return self._light_curve_indices
    
    def __iter__(self):
        """
        Generate batches with unique light curves and spectra
        """
        # Get light curve indices (lazy-loaded)
        light_curve_ids = list(self.light_curve_indices.keys())
        print("light_curve_ids: ", len(light_curve_ids), light_curve_ids[0])
        
        # Shuffle light curve IDs
        random.shuffle(light_curve_ids)
        
        # Track used spectra
        used_spectra = set()
        batch_indices = []
        
        for light_curve_id in light_curve_ids:
            # Get all spectra for this light curve not yet used
            available_spectra = [
                idx for idx in self.light_curve_indices[light_curve_id] 
                if idx not in used_spectra
            ]
            
            if not available_spectra:
                continue
            
            # Select a random spectrum
            selected_spectrum = random.choice(available_spectra)
            
            batch_indices.append(selected_spectrum)
            used_spectra.add(selected_spectrum)
            
            # Stop when we have a full batch
            if len(batch_indices) >= self.batch_size:
                break
        
        # If batch is not full, cycle through again with remaining light curves
        while len(batch_indices) < self.batch_size:
            random.shuffle(light_curve_ids)
            for light_curve_id in light_curve_ids:
                available_spectra = [
                    idx for idx in self.light_curve_indices[light_curve_id] 
                    if idx not in used_spectra
                ]
                
                if not available_spectra:
                    continue
                
                selected_spectrum = random.choice(available_spectra)
                batch_indices.append(selected_spectrum)
                used_spectra.add(selected_spectrum)
                
                if len(batch_indices) >= self.batch_size:
                    break
            
            # Prevent infinite loop
            if len(batch_indices) < self.batch_size:
                break
        
        # Trim or pad to exact batch size
        batch_indices = batch_indices[:self.batch_size]
        
        return iter(batch_indices)
    
    def __len__(self):
        return len(self.light_curve_indices)

class DistinctParameterSampler(Sampler):
    def __init__(self, dataset, batch_size, thresholds, num_replicas=1, rank=0):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.thresholds = thresholds
        self.num_replicas = num_replicas
        self.rank = rank
        
        # Precompute metadata for faster sampling
        self.metadata = []
        for idx in range(len(dataset)):
            _, _, _, _, info, _ = dataset[idx]
            self.metadata.append({
                'index': idx, 
                'Teff': info['Teff'], 
                'M': info['M'], 
                'logg': info['logg']
            })
    
    def is_distinct(self, sample1, sample2):
        return all(
            abs(sample1[param] - sample2[param]) <= self.thresholds[param] 
            for param in ['Teff', 'M', 'logg']
        )
    
    def __iter__(self):
        # Shuffle metadata
        random.shuffle(self.metadata)
        
        # Distribute samples across replicas
        per_replica = len(self.metadata) // self.num_replicas
        start = self.rank * per_replica
        end = start + per_replica
        local_metadata = self.metadata[start:end]
        
        indices = []
        while len(local_metadata) >= self.batch_size:
            batch = []
            for sample in local_metadata[:]:  # Create a copy to iterate safely
                if len(batch) == 0 or all(
                    self.is_distinct(sample, existing) 
                    for existing in batch
                ):
                    batch.append(sample)
                    local_metadata.remove(sample)
                
                if len(batch) == self.batch_size:
                    indices.append([item['index'] for item in batch])
                    break
        
        return iter(indices[0])  # Return only the first batch of indices
    
    def __len__(self):
        return len(self.metadata) // (self.batch_size * self.num_replicas)