from dataset.dataset import KeplerDataset
from util.utils import get_all_samples_df
import numpy as np
from matplotlib import pyplot as plt
import os
import warnings
import signal
import time
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
import tqdm
warnings.filterwarnings("ignore")
import traceback

class GracefulTimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise GracefulTimeoutException("Processing time limit reached")

def get_existing_kids(root_data_folder, folder_name):
    """Get a set of existing KIDs from the output directory."""
    output_dir = f'{root_data_folder}/{folder_name}'
    if not os.path.exists(output_dir):
        return set()
    
    existing_files = os.listdir(output_dir)
    return {os.path.splitext(f)[0] for f in existing_files if f.endswith('.npy')}

def filter_items_to_process(dataset, existing_kids):
    """Filter out items that already have corresponding NPY files."""
    items_to_process = []
    for idx in range(len(dataset)):
        _, _, _, _, info, _ = dataset[idx]
        kid = info['KID']
        if kid not in existing_kids:
            items_to_process.append((idx, dataset))
    return items_to_process

def process_single_item(idx_dataset_tuple, root_data_folder, folder_name, results_dict):
    idx, dataset = idx_dataset_tuple
    try:
        x, y, _, _, info, _ = dataset[idx]
        kid = info['KID']
        output_path = f'{root_data_folder}/{folder_name}/{kid}.npy'
        
        # Ensure directory exists (thread-safe)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the file and print success
        np.save(output_path, x)
        print(f'Successfully saved {kid}.npy', flush=True)
        
        # Verify file was saved
        if not os.path.exists(output_path):
            print(f'ERROR: Failed to save {kid}.npy', flush=True)
            results_dict['errors'] += 1
            return f'Error saving {kid}'
        
        results_dict['successful'] += 1
        return f'Successfully processed {kid}'
    except Exception as e:
        print(f'Error processing {idx}: {str(e)}', flush=True)
        traceback.print_exc()
        results_dict['errors'] += 1
        return f'Error processing index {idx}: {str(e)}'

def print_summary(results_dict, start_time, existing_count=0, timed_out=False):
    elapsed_time = time.time() - start_time
    
    print("\nProcessing Summary:")
    print(f"Successfully processed: {results_dict['successful']}")
    print(f"Already existing (skipped): {existing_count}")
    print(f"Errors encountered: {results_dict['errors']}")
    print(f"Total time elapsed: {elapsed_time/3600:.2f} hours")
    if timed_out:
        print("Process stopped due to time limit (23 hours 57 minutes)")

def kepler_fits_to_npy(raw=False, num_processes=None):
    # Set timeout for 23 hours and 57 minutes (in seconds)
    timeout_seconds = (23 * 60 * 60) + (57 * 60)
    
    # Set up the timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    start_time = time.time()
    
    # Create a manager for sharing results between processes
    with Manager() as manager:
        results_dict = manager.dict({
            'successful': 0,
            'errors': 0
        })
        
        try:
            # Set up paths
            folder_name = 'npy' if not raw else 'raw_npy'
            root_data_folder = "/data/lightPred/data"
            
            # Ensure output directory exists
            os.makedirs(f'{root_data_folder}/{folder_name}', exist_ok=True)
            
            # Load dataset
            print('loading dataframe...')
            kepler_df = get_all_samples_df(num_qs=None)
            print("filtering existings files....")
            existing_files = os.listdir(f"{root_data_folder}/{folder_name}")
            if len(existing_files):
                kepler_df['exists'] = kepler_df.apply(lambda x: f"{x['KID']}.npy" in existing_files, axis=1)
            else:
                kepler_df['exists'] = False
            kepler_df = kepler_df[~kepler_df['exists']]
            print(f"finished filter with {len(kepler_df)} samples")
            
            train_dataset = KeplerDataset(
                df=kepler_df,
                transforms=None,
                target_transforms=None,
                npy_path=None,
                seq_len=None,
                scale_flux=not raw
            )
            print(f"Total dataset size: {len(train_dataset)}")
           # Use SLURM CPU count if available, otherwise fall back to CPU count - 1
            if num_processes is None:
                try:
                    num_processes = int(os.environ["SLURM_CPUS_PER_TASK"])
                except (KeyError, ValueError):
                    num_processes = max(1, cpu_count() - 1)
            
            # Create a list of (index, dataset) tuples
            items_to_process = [(i, train_dataset) for i in range(len(train_dataset))]
            
            # Prepare the partial function with fixed arguments
            process_func = partial(
                process_single_item,
                root_data_folder=root_data_folder,
                folder_name=folder_name,
                results_dict=results_dict
            )
            
            # Process items in parallel with progress bar
            print(f"Processing with {num_processes} processes...")
            with Pool(processes=num_processes) as pool:
                for _ in tqdm.tqdm(
                    pool.imap_unordered(process_func, items_to_process),
                    total=len(items_to_process),
                    desc="Converting FITS to NPY"
                ):
                    pass
                    
            print_summary(results_dict, start_time)
            
        except GracefulTimeoutException:
            print_summary(results_dict, start_time, timed_out=True)
        finally:
            # Disable the alarm
            signal.alarm(0)

if __name__ == '__main__':
    kepler_fits_to_npy(raw=True)
