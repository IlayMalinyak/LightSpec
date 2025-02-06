from dataset.dataset import KeplerDataset
from util.utils import get_all_samples_df
import numpy as np
from matplotlib import pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")


def kepler_fits_to_npy(raw=False):
    folder_name = 'npy' if not raw else 'raw_npy'
    root_data_folder =  "/data/lightPred/data"
    kepler_df = get_all_samples_df(num_qs=None)
    
    train_dataset = KeplerDataset(df=kepler_df, transforms=None,
                                target_transforms=None,
                                npy_path = None,
                                seq_len=None,
                                )
    print("len full dataset", len(train_dataset))
    for i in range(len(train_dataset)):
        # try:
            x, y, _, _, info, _ = train_dataset[i]
            kid = info['KID']
            if f'{kid}.npy' in os.listdir(f'{root_data_folder}/{folder_name}'):
                # print(f'{kid} already exists')
                continue
            np.save(f'{root_data_folder}/{folder_name}/{kid}.npy', x)
            # print(f'{kid} saved')
            # if i % 10 == 0:
            #     plt.plot(x)
            #     plt.savefig(f'/data/tests/{kid}.png')
            #     plt.close()
            # if i > 100:
            #     break
            if i % 1000 == 0:
                print(f'{i} done')
        # except Exception as e:
        #     print(e)
        #     continue

if __name__ == '__main__':
    kepler_fits_to_npy(raw=True)