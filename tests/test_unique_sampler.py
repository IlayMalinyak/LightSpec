import torch
import numpy as np
import pytest
from dataset.dataset import create_unique_loader
from util.utils import kepler_collate_fn
from tqdm import tqdm
import time

def test_light_curve_uniqueness(dataset, batch_size=32, num_batches=10):
    """
    Test the uniqueness of light curves in each batch
    
    Args:
    - dataset: The LightSpecDataset instance
    - batch_size: Number of samples per batch
    - num_batches: Number of batches to check
    """
    # Create the data loader with our custom sampler
    train_loader = create_unique_loader(
        dataset, 
        batch_size=batch_size,
        num_workers=0,  # Important for deterministic testing
        collate_fn=kepler_collate_fn
    )
    print(f"Created data loader with {len(train_loader)} batches")
    print("dataset length: ", len(dataset))

    pbar = tqdm(train_loader)
    print(f"Checking uniqueness of light curves in {num_batches} batches...")
    # Iterate through a limited number of batches
    start_time = time.time()
    for i, batch in enumerate(pbar):
        if i >= num_batches:
            break
        _, _, _, _, info, _ = batch
        kids = [inf['KID'] for inf in info]
        # # check for uniqueness of light curve in this batch
        assert len(kids) == len(set(kids)), f"Duplicate light curve in batch {i}"
        
        # Optional: Print batch details for verification
    
    print(f"Passed uniqueness checks for {num_batches} batches!")
    print(f"Time taken: {time.time() - start_time:.2f} seconds. time per batch: {(time.time() - start_time)/num_batches:.2f} seconds")

def run_sampler_tests(dataset, batch_size=32, num_batches=10):
    """
    Run all sampler tests
    
    Args:
    - dataset: The LightSpecDataset instance
    """
    print("Running Light Curve Uniqueness Test...")
    test_light_curve_uniqueness(dataset, batch_size=batch_size, num_batches=num_batches)