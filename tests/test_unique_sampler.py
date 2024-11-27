import torch
import numpy as np
import pytest
from dataset.dataset import create_moco_loader
from util.utils import kepler_collate_fn

def test_light_curve_uniqueness(dataset, batch_size=32, num_batches=10):
    """
    Test the uniqueness of light curves in each batch
    
    Args:
    - dataset: The LightSpecDataset instance
    - batch_size: Number of samples per batch
    - num_batches: Number of batches to check
    """
    # Create the data loader with our custom sampler
    train_loader = create_moco_loader(
        dataset, 
        batch_size=batch_size,
        num_workers=0,  # Important for deterministic testing
        collate_fn=kepler_collate_fn
    )
    
    # Iterate through a limited number of batches
    for batch_idx, indices in enumerate(train_loader):
        print(f"Batch {batch_idx}: {indices}")
        if batch_idx >= num_batches:
            break
        
        # Verify batch size
        assert len(indices) == batch_size, f"Batch {batch_idx} does not have expected size. size = {len(indices)}." \
          f"batch_size = {batch_size}"
        
        # Extract light curve IDs for this batch
        batch_light_curve_ids = []
        used_spectra = set()
        
        for idx in indices:
            # Retrieve info without loading entire data
            _, _, _, _, info, _ = dataset[idx]
            light_curve_id = info['KID']
            
            # Check for uniqueness of light curve in this batch
            assert light_curve_id not in batch_light_curve_ids, \
                f"Duplicate light curve {light_curve_id} in batch {batch_idx}"
            batch_light_curve_ids.append(light_curve_id)
            
            # Check for uniqueness of spectrum
            assert idx not in used_spectra, \
                f"Spectrum index {idx} reused in batch {batch_idx}"
            used_spectra.add(idx)
        
        # Optional: Print batch details for verification
        print(f"Batch {batch_idx}: Light Curve IDs = {batch_light_curve_ids}")
    
    print(f"Passed uniqueness checks for {num_batches} batches!")

def test_sampler_coverage(dataset, batch_size=32, num_batches=10):
    """
    Test that the sampler covers a good portion of the dataset
    
    Args:
    - dataset: The LightSpecDataset instance
    - batch_size: Number of samples per batch
    - num_batches: Number of batches to check
    """
    # Create the data loader with our custom sampler
    train_loader = create_moco_loader(
        dataset, 
        batch_size=batch_size,
        num_workers=0  # Important for deterministic testing
    )
    
    # Track all unique indices and light curves seen
    all_indices = set()
    all_light_curves = set()
    
    # Iterate through a limited number of batches
    for batch_idx, indices in enumerate(train_loader):
        if batch_idx >= num_batches:
            break
        
        for idx in indices:
            # Retrieve info without loading entire data
            _, _, _, _, info, _ = dataset[idx]
            light_curve_id = info['KID']
            
            all_indices.add(idx)
            all_light_curves.add(light_curve_id)
    
    # Calculate coverage
    total_dataset_size = len(dataset)
    total_unique_light_curves = len(set(
        info['KID'] for _, _, _, _, info, _ in [dataset[i] for i in range(total_dataset_size)]
    ))
    
    print(f"Unique indices covered: {len(all_indices)} / {total_dataset_size}")
    print(f"Unique light curves covered: {len(all_light_curves)} / {total_unique_light_curves}")
    
    # Optional: Add some basic assertions to verify coverage
    assert len(all_indices) > 0, "No indices were sampled"
    assert len(all_light_curves) > 0, "No light curves were sampled"

def run_sampler_tests(dataset, batch_size=32, num_batches=10):
    """
    Run all sampler tests
    
    Args:
    - dataset: The LightSpecDataset instance
    """
    print("Running Light Curve Uniqueness Test...")
    test_light_curve_uniqueness(dataset, batch_size=batch_size, num_batches=num_batches)
    
    print("\nRunning Sampler Coverage Test...")
    test_sampler_coverage(dataset, batch_size=batch_size, num_batches=num_batches)