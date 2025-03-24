import torch
from tqdm import tqdm
import torch
import random
from tqdm import tqdm
from torch.utils.data import Sampler, Subset, Dataset
import torch.distributed as dist
import matplotlib.pyplot as plt
from random import shuffle
from typing import List, Optional, TypeVar, Iterator
import math
import numpy as np

T_co = TypeVar('T_co', covariant=True)

import torch
from torch.utils.data import Sampler
import numpy as np
import math
from typing import Optional, List, TypeVar, Iterator
from collections import defaultdict

T_co = TypeVar('T_co', covariant=True)

class BalancedDistributedSampler(Sampler[T_co]):
    """
    Sampler that ensures class balance in each batch while being compatible with distributed training.
    
    Args:
        dataset: Dataset used for sampling.
        labels: List of labels for each dataset item (must be integers).
        num_replicas: Number of processes participating in distributed training.
        rank: Rank of the current process within num_replicas.
        shuffle: Whether to shuffle the indices.
        balanced_ratio: Target ratio between classes (e.g., 1.0 means equal distribution).
    """
    
    def __init__(
        self, 
        dataset: torch.utils.data.Dataset, 
        labels: List[int], 
        num_replicas: int = 1, 
        rank: int = 0, 
        shuffle: bool = True,
        balanced_ratio: float = 1.0,
        verbose: bool = False
    ) -> None:
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.balanced_ratio = balanced_ratio
        self.verbose = verbose
        
        # Cast labels to integers
        self.labels = np.array([int(label) for label in labels])
        
        # Group indices by label
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.label_to_indices[label].append(idx)
            
        # Get all possible class labels
        self.unique_labels = sorted(self.label_to_indices.keys())
        if self.verbose:
            print(f"Unique labels: {self.unique_labels}")
            for label in self.unique_labels:
                print(f"Class {label} count: {len(self.label_to_indices[label])}")
        
        # Determine number of samples
        n_minority = min([len(self.label_to_indices[label]) for label in self.unique_labels])
        n_majority = max([len(self.label_to_indices[label]) for label in self.unique_labels])
        
        # Calculate how many samples we need per class based on balanced_ratio
        # 0.0 = original distribution, 1.0 = completely balanced
        # Formula: adjusted = minority + (majority - minority) * (1 - balanced_ratio)
        samples_per_class = {}
        for label in self.unique_labels:
            class_size = len(self.label_to_indices[label])
            if class_size == n_majority:  # Majority class
                samples_per_class[label] = int(n_minority + (class_size - n_minority) * (1 - self.balanced_ratio))
            else:  # Minority class(es)
                samples_per_class[label] = int(class_size)
        
        if self.verbose:
            print(f"Samples per class after balancing: {samples_per_class}")
            
        # Ensure we have the right number of samples per process
        total_samples = sum(samples_per_class.values())
        self.num_samples = total_samples // self.num_replicas
        if total_samples % self.num_replicas != 0:
            self.num_samples += 1
        self.total_size = self.num_samples * self.num_replicas
        
        # Calculate how many samples from each class per process
        self.samples_per_class_per_process = {}
        for label in self.unique_labels:
            self.samples_per_class_per_process[label] = samples_per_class[label] // self.num_replicas
            if self.samples_per_class_per_process[label] == 0:
                self.samples_per_class_per_process[label] = 1  # Ensure at least 1 sample
                
    def __iter__(self) -> Iterator[T_co]:
        indices = []
        
        if self.shuffle:
            # Deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.epoch)
            
            # Shuffle indices within each class
            for label in self.unique_labels:
                label_indices = self.label_to_indices[label]
                perm = torch.randperm(len(label_indices), generator=g).tolist()
                indices_for_label = [label_indices[idx] for idx in perm]
                
                # Take samples_per_class_per_process for this process (with replacement if needed)
                samples_needed = self.samples_per_class_per_process[label]
                if samples_needed > len(indices_for_label):
                    # Need to oversample with replacement
                    additional = np.random.choice(
                        indices_for_label, 
                        size=samples_needed - len(indices_for_label),
                        replace=True
                    ).tolist()
                    indices_for_label.extend(additional)
                
                # Add to overall indices
                indices.extend(indices_for_label[:samples_needed])
        else:
            # Same as above but without shuffling
            for label in self.unique_labels:
                label_indices = self.label_to_indices[label]
                samples_needed = self.samples_per_class_per_process[label]
                if samples_needed > len(label_indices):
                    # Need to oversample with replacement
                    indices_for_label = label_indices.copy()
                    additional = np.random.choice(
                        label_indices, 
                        size=samples_needed - len(label_indices),
                        replace=True
                    ).tolist()
                    indices_for_label.extend(additional)
                else:
                    indices_for_label = label_indices[:samples_needed]
                
                # Add to overall indices
                indices.extend(indices_for_label)
        
        # Shuffle all indices
        if self.shuffle:
            np.random.shuffle(indices)
        
        # Slice to get indices for this rank
        indices = indices[self.rank::self.num_replicas]
        
        if self.verbose:
            # Verify balance in this process
            rank_labels = [self.labels[idx] for idx in indices]
            class_counts = {}
            for label in self.unique_labels:
                class_counts[label] = rank_labels.count(label)
            print(f"Rank {self.rank} - Class counts: {class_counts}")
        
        return iter(indices)
        
    def __len__(self) -> int:
        return self.num_samples
        
    def set_epoch(self, epoch: int) -> None:
        """
        Sets the epoch for this sampler. Required for proper shuffling.
        
        Args:
            epoch (int): Epoch number
        """
        self.epoch = epoch
class DistributedUniqueLightCurveSampler(Sampler):
    def __init__(self, dataset, batch_size, num_replicas=None, rank=None):
        # Ensure distributed setup
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
        
        # Distributed sampling calculations
        total_size = len(self.original_indices)
        per_replica_size = total_size // self.num_replicas
        
        # Ensure each replica gets an equal subset
        self.indices_per_replica = self.original_indices[
            self.rank * per_replica_size : (self.rank + 1) * per_replica_size
        ]
        
    def _build_light_curve_index(self):
        """
        Efficiently build light curve index for this replica's subset
        """
        if hasattr(self.base_dataset, 'df'):
            dataframe = self.base_dataset.df
        else:
            raise AttributeError("Could not find a dataframe for indexing")
        
        light_curve_mapping = {}
        
        print(f"Building light curve index for replica {self.rank}...")
        for local_idx, original_idx in enumerate(self.indices_per_replica):
            row = dataframe.iloc[original_idx]
            light_curve_id = int(row['KID'])
            
            if light_curve_id not in light_curve_mapping:
                light_curve_mapping[light_curve_id] = []
            
            light_curve_mapping[light_curve_id].append(local_idx)
        
        return light_curve_mapping
    
    @property
    def light_curve_indices(self):
        """
        Lazy-loaded light curve indices specific to this replica
        """
        if self._light_curve_indices is None:
            self._light_curve_indices = self._build_light_curve_index()
        return self._light_curve_indices
    
    def __iter__(self):
        # Reset the generator for reproducibility
        generator = torch.Generator()
        generator.manual_seed(self.rank)  # Use rank for consistent shuffling

        # Shuffle light curve IDs for randomness
        light_curve_ids = list(self.light_curve_indices.keys())
        torch.manual_seed(self.rank)
        shuffle(light_curve_ids)

        batches = []
        used_light_curves = set()

        # Generate batches
        for _ in range(len(self)):
            batch_indices = []

            while len(batch_indices) < self.batch_size:
                for light_curve_id in light_curve_ids:
                    if light_curve_id in used_light_curves:
                        continue

                    available_indices = self.light_curve_indices[light_curve_id]

                    if available_indices:
                        # Select a random index
                        selected_idx = available_indices[
                            torch.randint(len(available_indices), (1,), generator=generator).item()
                        ]

                        batch_indices.append(selected_idx)
                        used_light_curves.add(light_curve_id)

                    if len(batch_indices) >= self.batch_size:
                        break

                # If we can't fill the batch, break to prevent infinite loop
                if len(batch_indices) >= self.batch_size:
                    break

            # Pad the batch if necessary
            while len(batch_indices) < self.batch_size:
                batch_indices.append(batch_indices[0])  # Repeat an existing index

            batches.append(batch_indices)

        # Yield each batch
        for batch in batches:
            yield batch

    def __len__(self):
        total_samples = len(self.indices_per_replica)
        num_batches = total_samples // self.batch_size
        print(f"Calculating __len__: Total samples {total_samples}, Batch size {self.batch_size}, Num batches {num_batches}")
        return num_batches
    
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

class DistributedBalancedSampler(Sampler):
    """
    A distributed sampler that supports:
    - Balanced sampling across different worker nodes
    - Optional shuffling
    - Handling datasets of different sizes
    - Consistent sampling across epochs
    """
    def __init__(
        self, 
        dataset: Dataset, 
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False
    ):
        """
        Args:
            dataset (Dataset): Input dataset
            num_replicas (int, optional): Number of processes participating in distributed training
            rank (int, optional): Rank of the current process
            shuffle (bool): Whether to shuffle the data
            seed (int): Random seed for shuffling
            drop_last (bool): If True, drop the last incomplete batch
        """
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Torch distributed is not available. Cannot determine number of replicas.")
            num_replicas = dist.get_world_size()
        
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Torch distributed is not available. Cannot determine rank.")
            rank = dist.get_rank()
        
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Determine samples per replica
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil(
                (len(self.dataset) - (self.num_replicas - 1)) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        
        self.total_size = self.num_samples * self.num_replicas
        
        # Seed management
        self.seed = seed
    
    def __iter__(self):
        """
        Generate indices for the current process
        """
        # Set the seed for reproducibility
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        
        # Create indices
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        else:
            indices = list(range(len(self.dataset)))
        
        # Pad indices to ensure even distribution
        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size > 0:
                indices += indices[:padding_size]
        
        # Ensure each process gets the same number of samples
        assert len(indices) >= self.num_samples
        
        # Select indices for the current process
        start_idx = self.rank * self.num_samples
        end_idx = start_idx + self.num_samples
        process_indices = indices[start_idx:end_idx]
        
        return iter(process_indices)
    
    def __len__(self):
        """
        Return the number of samples for this process
        """
        return self.num_samples
    
    def set_epoch(self, epoch: int):
        """
        Sets the epoch for shuffling
        
        Args:
            epoch (int): Current epoch number
        """
        self.epoch = epoch

class UniqueIDDistributedSampler(Sampler):
    """
    A distributed sampler that ensures:
    - Unique samples per batch based on KID (Light Curve ID)
    - Balanced sampling across distributed workers
    - Consistent and reproducible sampling
    """
    def __init__(
        self, 
        dataset: Dataset, 
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False
    ):
        """
        Args:
            dataset (Dataset): Input dataset with a dataframe containing 'KID' column
            num_replicas (int, optional): Number of processes participating in distributed training
            rank (int, optional): Rank of the current process
            shuffle (bool): Whether to shuffle the data
            seed (int): Random seed for shuffling
            drop_last (bool): If True, drop the last incomplete batch
        """
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Torch distributed is not available. Cannot determine number of replicas.")
            num_replicas = dist.get_world_size()
        
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Torch distributed is not available. Cannot determine rank.")
            rank = dist.get_rank()
        
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        if isinstance(dataset, Subset):
            self.dataset = dataset.dataset
        
        print(self.dataset)

        # Get the underlying dataframe
        if hasattr(self.dataset, 'df'):
            self.dataframe = self.dataset.df
        else:
            raise AttributeError("Could not find a dataframe with 'KID' column")
        
        # Build initial index mapping
        self.indices = list(range(len(self.dataset)))
        self._build_light_curve_index()
        
        # Determine samples per replica
        if self.drop_last and len(self.indices) % self.num_replicas != 0:
            self.num_samples = math.ceil(
                (len(self.indices) - (self.num_replicas - 1)) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(self.indices) / self.num_replicas)
        
        self.total_size = self.num_samples * self.num_replicas
    
    def _build_light_curve_index(self):
        """
        Build light curve index mapping KID to indices
        """
        # Reset mapping
        self.light_curve_mapping = {}
        
        # Generator for reproducible randomness
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        
        # Shuffle indices if required
        if self.shuffle:
            shuffled_indices = torch.randperm(len(self.indices), generator=generator).tolist()
        else:
            shuffled_indices = self.indices.copy()
        
        # Build mapping
        for idx in shuffled_indices:
            row = self.dataframe.iloc[idx]
            light_curve_id = int(row['KID'])
            
            if light_curve_id not in self.light_curve_mapping:
                self.light_curve_mapping[light_curve_id] = []
            
            self.light_curve_mapping[light_curve_id].append(idx)
    
    def __iter__(self):
        """
        Generate batch indices ensuring unique KID representation
        """
        # Rebuild index mapping for current epoch
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        
        # Select one random index for each unique KID
        batch_indices = []
        for kid_indices in self.light_curve_mapping.values():
            # Randomly select one index for this KID
            selected_idx = kid_indices[torch.randint(len(kid_indices), size=(1,), generator=generator).item()]
            batch_indices.append(selected_idx)
        
        # Pad if necessary to ensure even distribution
        if not self.drop_last:
            padding_size = self.total_size - len(batch_indices)
            if padding_size > 0:
                # Repeat random selections to pad
                extra_indices = [
                    list(self.light_curve_mapping.values())[
                        torch.randint(len(self.light_curve_mapping), size=(1,), generator=generator).item()
                    ][torch.randint(len(kid_indices), size=(1,), generator=generator).item()]
                    for _ in range(padding_size)
                ]
                batch_indices.extend(extra_indices)
        
        # Distribute indices across replicas
        start_idx = self.rank * self.num_samples
        end_idx = start_idx + self.num_samples
        
        process_indices = batch_indices[start_idx:end_idx]
        
        return iter(process_indices)
    
    def __len__(self):
        """
        Return the number of samples for this process
        """
        return self.num_samples
    
    def set_epoch(self, epoch: int):
        """
        Sets the epoch for shuffling
        
        Args:
            epoch (int): Current epoch number
        """
        self.epoch = epoch
        # Rebuild index mapping when epoch changes
        self._build_light_curve_index()
