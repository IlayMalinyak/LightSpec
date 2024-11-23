import torch
from tqdm import tqdm
import torch
import random

class DistinctParameterSampler(torch.utils.data.Sampler):
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