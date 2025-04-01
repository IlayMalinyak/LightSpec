import torch
from torch.cuda.amp import autocast
import numpy as np
import time
import os
import yaml
import json
from matplotlib import pyplot as plt
import glob
from collections import OrderedDict
from tqdm import tqdm
import torch.distributed as dist
import umap
from torch.profiler import profile, record_function, ProfilerActivity
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader
from util.utils import kepler_collate_fn, setup_dist_logger 
from pathlib import Path
import io
import zipfile
# import wandb

def count_occurence(x,y):
  coord_counts = {}
  for i in range(len(x)):
      coord = (x[i], y[i])
      if coord in coord_counts:
          coord_counts[coord] += 1
      else:
          coord_counts[coord] = 1

def save_compressed_checkpoint(model, save_path, results, use_zip=True):
        """
        Save model checkpoint with compression
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        if use_zip:
            # Save model state dict to buffer first
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer, 
                    _use_new_zipfile_serialization=True,
                    pickle_protocol=4)
            
            # Save buffer to compressed zip
            model_path = str(Path(save_path).with_suffix('.zip'))
            with zipfile.ZipFile(model_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr('model.pt', buffer.getvalue())
                
            # Save results separately
            results_path = str(Path(save_path).with_suffix('.json'))
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
                
        else:
            # Save with built-in compression
            model_path = str(Path(save_path).with_suffix('.pt'))
            torch.save(model.state_dict(), model_path,
                    _use_new_zipfile_serialization=True,
                    pickle_protocol=4)
                
            results_path = str(Path(save_path).with_suffix('.json'))
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
                
        return model_path, results_path

def load_compressed_checkpoint(model, save_path):
    """
    Load model checkpoint from compressed format
    """
    if save_path.endswith('.zip'):
        with zipfile.ZipFile(save_path) as zf:
            with zf.open('model.pt') as f:
                buffer = io.BytesIO(f.read())
                state_dict = torch.load(buffer)
    else:
        state_dict = torch.load(save_path)
    
    model.load_state_dict(state_dict)
    return model



class Trainer(object):
    """
    A class that encapsulates the training loop for a PyTorch model.
    """
    def __init__(self, model, optimizer, criterion, train_dataloader, device, world_size=1, output_dim=2,
                 scheduler=None, val_dataloader=None,   max_iter=np.inf, scaler=None, logger=None,
                  grad_clip=False, exp_num=None, log_path=None, exp_name=None, plot_every=None,
                   cos_inc=False, range_update=None, accumulation_step=1, wandb_log=False, num_quantiles=1,
                   update_func=lambda x: x):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scaler = scaler
        self.grad_clip = grad_clip
        self.cos_inc = cos_inc
        self.output_dim = output_dim
        self.scheduler = scheduler
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader
        self.train_sampler = self.get_sampler_from_dataloader(train_dataloader)
        self.val_sampler = self.get_sampler_from_dataloader(val_dataloader)
        self.max_iter = max_iter
        self.device = device
        self.world_size = world_size
        self.exp_num = exp_num
        self.exp_name = exp_name
        self.log_path = log_path
        self.best_state_dict = None
        self.plot_every = plot_every
        self.logger = logger
        self.range_update = range_update
        self.accumulation_step = accumulation_step
        self.wandb = wandb_log
        self.num_quantiles = num_quantiles
        self.update_func = update_func
        # if log_path is not None:
        #     self.logger =SummaryWriter(f'{self.log_path}/exp{self.exp_num}')
        #     # print(f"logger path: {self.log_path}/exp{self.exp_num}")

        # print("logger is: ", self.logger)
    
    def get_sampler_from_dataloader(self, dataloader):
        if hasattr(dataloader, 'sampler'):
            if isinstance(dataloader.sampler, torch.utils.data.DistributedSampler):
                return dataloader.sampler
            elif hasattr(dataloader.sampler, 'sampler'):
                return dataloader.sampler.sampler
        
        if hasattr(dataloader, 'batch_sampler') and hasattr(dataloader.batch_sampler, 'sampler'):
            return dataloader.batch_sampler.sampler
        
        return None
    
    def fit(self, num_epochs, device,  early_stopping=None, start_epoch=0, best='loss', conf=False):
        """
        Fits the model for the given number of epochs.
        """
        min_loss = np.inf
        best_acc = 0
        train_loss, val_loss,  = [], []
        train_acc, val_acc = [], []
        lrs = []
        epochs = []
        self.train_aux_loss_1 = []
        self.train_aux_loss_2 = []
        self.val_aux_loss_1 = []
        self.val_aux_loss_2 = []
        self.train_logits_mean = []
        self.train_logits_std = []
        self.val_logits_mean = []
        self.val_logits_std = []
        # self.optim_params['lr_history'] = []
        epochs_without_improvement = 0
        if self.logger is None:
            self.logger = setup_dist_logger(device)
        main_proccess = (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0) or self.device == 'cpu'

        self.logger.info(f"Starting training for {num_epochs} epochs")
        global_time = time.time()
        self.epoch = 0
        for epoch in range(start_epoch, start_epoch + num_epochs):
            epochs.append(epoch)
            self.epoch = epoch
            start_time = time.time()
            plot = (self.plot_every is not None) and (epoch % self.plot_every == 0)
            t_loss, t_acc = self.train_epoch(device, epoch=epoch)
            t_loss_mean = np.nanmean(t_loss)
            train_loss.extend(t_loss)
            global_train_accuracy, global_train_loss = self.process_loss(t_acc, t_loss_mean)
            if main_proccess:  # Only perform this on the master GPU
                train_acc.append(global_train_accuracy.mean().item())
                
            v_loss, v_acc = self.eval_epoch(device, epoch=epoch)
            v_loss_mean = np.nanmean(v_loss)
            val_loss.extend(v_loss)
            global_val_accuracy, global_val_loss = self.process_loss(v_acc, v_loss_mean)
            if main_proccess:  # Only perform this on the master GPU                
                val_acc.append(global_val_accuracy.mean().item())
                
                current_objective = global_val_loss if best == 'loss' else global_val_accuracy.mean()
                improved = False
                
                if best == 'loss':
                    if current_objective < min_loss:
                        min_loss = current_objective
                        improved = True
                else:
                    if current_objective > best_acc:
                        best_acc = current_objective
                        improved = True
                
                if improved:
                    model_name = f'{self.log_path}/{self.exp_num}/{self.exp_name}.pth'
                    print(f"saving model at {model_name}...")
                    torch.save(self.model.state_dict(), model_name)
                    self.best_state_dict = self.model.state_dict()
                    # model_path, output_filename = save_compressed_checkpoint(
                    #                            self.model, model_name, res, use_zip=True )
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                res = {"epochs": epochs, "train_loss": train_loss, "val_loss": val_loss,
                        "train_acc": train_acc, "val_acc": val_acc, "train_aux_loss_1": self.train_aux_loss_1,
                        "train_aux_loss_2":self.train_aux_loss_2, "val_aux_loss_1":self.val_aux_loss_1,
                        "val_aux_loss_2": self.val_aux_loss_2, "train_logits_mean": self.train_logits_mean,
                         "train_logits_std": self.train_logits_std, "val_logits_mean": self.val_logits_mean,
                          "val_logits_std": self.val_logits_std, "lrs": lrs}

                current_lr = self.optimizer.param_groups[0]['lr'] if self.scheduler is None \
                            else self.scheduler.get_last_lr()[0]
                
                lrs.append(current_lr)
                
                output_filename = f'{self.log_path}/{self.exp_num}/{self.exp_name}.json'
                with open(output_filename, "w") as f:
                    json.dump(res, f, indent=2)
                logger.info(f"saved results at {output_filename}")
                
                logger.info(f'Epoch {epoch}, lr {current_lr}, Train Loss: {global_train_loss:.6f}, Val Loss:'\
                
                        f'{global_val_loss:.6f}, Train Acc: {global_train_accuracy.round(decimals=4).tolist()}, '\
                f'Val Acc: {global_val_accuracy.round(decimals=4).tolist()},'\
                  f'Time: {time.time() - start_time:.2f}s, Total Time: {(time.time() - global_time)/3600} hr', flush=True)
                if epoch % 10 == 0:
                    logger.info(os.system('nvidia-smi'))

                if epochs_without_improvement == early_stopping:
                    logger.info('early stopping!', flush=True)
                    break
                if time.time() - global_time > (23.83 * 3600):
                    logger.info("time limit reached")
                    break 

        return {"epochs":epochs, "train_loss": train_loss,
                 "val_loss": val_loss, "train_acc": train_acc,
                "val_acc": val_acc, "train_aux_loss_1": self.train_aux_loss_1,
                "train_aux_loss_2": self.train_aux_loss_2, "val_aux_loss_1":self.val_aux_loss_1,
                "val_aux_loss_2": self.val_aux_loss_2, "train_logits_mean": self.train_logits_mean,
                 "train_logits_std": self.train_logits_std, "val_logits_mean": self.val_logits_mean,
                  "val_logits_std": self.val_logits_std, "lrs": lrs}

    def process_loss(self, acc, loss_mean):
        if  torch.cuda.is_available() and torch.distributed.is_initialized():
            global_accuracy = torch.tensor(acc).cuda()  # Convert accuracy to a tensor on the GPU
            torch.distributed.reduce(global_accuracy, dst=0, op=torch.distributed.ReduceOp.SUM)
            global_loss = torch.tensor(loss_mean).cuda()  # Convert loss to a tensor on the GPU
            torch.distributed.reduce(global_loss, dst=0, op=torch.distributed.ReduceOp.SUM)
            
            # Divide both loss and accuracy by world size
            world_size = torch.distributed.get_world_size()
            global_loss /= world_size
            global_accuracy /= world_size
        else:
            global_loss = torch.tensor(loss_mean)
            global_accuracy = torch.tensor(acc)
        return global_accuracy, global_loss

    def load_best_model(self, to_ddp=True, from_ddp=True):
        data_dir = f'{self.log_path}/exp{self.exp_num}'
        # data_dir = f'{self.log_path}/exp29' # for debugging

        state_dict_files = glob.glob(data_dir + '/*.pth')
        print("loading model from ", state_dict_files[-1])
        
        state_dict = torch.load(state_dict_files[-1]) if to_ddp else torch.load(state_dict_files[0],map_location=self.device)
    
        if from_ddp:
            print("loading distributed model")
            # Remove "module." from keys
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    while key.startswith('module.'):
                        key = key[7:]
                new_state_dict[key] = value
            state_dict = new_state_dict
        # print("state_dict: ", state_dict.keys())
        # print("model: ", self.model.state_dict().keys())

        self.model.load_state_dict(state_dict, strict=False)

    def check_gradients(self):
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm > 100:
                   print(f"Large gradient in {name}: {grad_norm}")

    def train_epoch(self, device, epoch):
        """
        Trains the model for one epoch.
        """
        if self.train_sampler is not None:
            try:
                self.train_sampler.set_epoch(epoch)
            except AttributeError:
                pass
        self.model.train()
        train_loss = []
        train_acc = 0
        total = 0
        all_accs = torch.zeros(self.output_dim, device=device)
        pbar = tqdm(self.train_dl)
        for i, batch in enumerate(pbar):
            if self.optimizer is not None:
                self.optimizer.zero_grad()
            loss, acc , y = self.train_batch(batch, i + epoch * len(self.train_dl), device)
            train_loss.append(loss.item())
            all_accs = all_accs + acc
            total += len(y)
            pbar.set_description(f"train_acc: {acc}, train_loss:  {loss.item():.4f}")      
            if i > self.max_iter:
                break
        print("number of train_accs: ", all_accs, "total: ", total)
        return train_loss, all_accs/total
    
    def train_batch(self, batch, batch_idx, device):
        lc,spec,_,_, y,info = batch
        b, _, _ = lc.shape
        spec = spec.to(device)
        lc = lc.to(device)
        if isinstance(y, tuple):
            y = torch.stack(y)
        y = y.to(device)
        y_pred = self.model(lc.float(), spec.float())
        y_pred = y_pred.reshape(b, -1, self.output_dim)
        if len(y.shape) == 1:
            y = y.unsqueeze(1)
        loss = 0
        for i in range(self.output_dim):
            y_pred_i = y_pred[:, :, i]
            y_i = y[:, i]
            loss += self.criterion(y_pred_i, y_i)
        loss /= self.output_dim
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        y_pred_mean = y_pred[:, y_pred.shape[1]//2, :]
        diff = torch.abs(y_pred_mean - y)
        acc = (diff < (y/10)).sum(0)
        # if self.wandb:
            # wandb.log({"train_loss": loss.item(), "train_acc": acc})
        return loss, acc, y

    def eval_epoch(self, device, epoch):
        """
        Evaluates the model for one epoch.
        """
        self.model.eval()
        val_loss = []
        val_acc = 0
        total = 0
        all_accs = torch.zeros(self.output_dim, device=device)
        pbar = tqdm(self.val_dl)
        for i,batch in enumerate(pbar):
            loss, acc, y = self.eval_batch(batch, i + epoch * len(self.val_dl), device)
            val_loss.append(loss.item())
            all_accs = all_accs + acc
            total += len(y)
            pbar.set_description(f"val_acc: {acc}, val_loss:  {loss.item():.4f}")
            if i > self.max_iter:
                break
        return val_loss, all_accs/total

    def eval_batch(self, batch, batch_idx, device):
        lc,spec,_,_,y,info = batch
        spec = spec.to(device)
        lc = lc.to(device)
        b, _, _ = lc.shape
        if isinstance(y, tuple):
            y = torch.stack(y)
        y = y.to(device)
        with torch.no_grad():
            y_pred= self.model(lc.float(), spec.float())
            y_pred = y_pred.reshape(b, -1, self.output_dim)
        if len(y.shape) == 1:
            y = y.unsqueeze(1)
        loss = 0
        for i in range(self.output_dim):
            y_pred_i = y_pred[:, :, i]
            y_i = y[:, i]
            loss += self.criterion(y_pred_i, y_i)
        loss /= self.output_dim
        y_pred_mean = y_pred[:, y_pred.shape[1]//2, :]
        diff = torch.abs(y_pred_mean - y)
        acc = (diff < (y/10)).sum(0)
        # if self.wandb:
        #     wandb.log({"val_loss": loss.item(), "val_acc": acc})
        return loss, acc, y

    def predict(self, test_dataloader, device, load_best=True):
        """
        Returns the predictions of the model on the given dataset.
        """
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
        if load_best:
            self.load_best_model(from_ddp=False)
        self.model.eval()
        preds = np.zeros((0, self.output_dim))
        targets = np.zeros((0, self.output_dim))
        confs = np.zeros((0, self.output_dim))
        tot_kic = []
        tot_teff = []
        for i,(lc_spec,_,_, y,info) in enumerate(test_dataloader):
            spec = spec.to(device)
            lc = lc.to(device)
            if isinstance(y, tuple):
                y = torch.stack(y)
            y = y.to(device)
            with torch.no_grad():
                y_pred = self.model(spec.float(), lc.float())
            y_pred = y_pred.reshape(b, -1, self.output_dim)
            preds = np.concatenate((preds, y_pred.cpu().numpy()))
            if y.shape[1] == self.output_dim:
                targets = np.concatenate((targets, y.cpu().numpy()))
        print("target len: ", len(targets), "dataset: ", len(test_dataloader.dataset))
        return preds, targets

class KFoldTrainer(Trainer):
    """
    A trainer class that implements k-fold cross validation by extending the base Trainer class.
    """
    def __init__(self, model, optimizer, criterion, dataset, device, n_splits=5, batch_size=32, 
                 shuffle=True, **kwargs):
        self.n_splits = n_splits
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Create a dummy dataloader just to initialize parent class
        dummy_dataloader = DataLoader(dataset, batch_size=batch_size)
        super().__init__(model=model, optimizer=optimizer, criterion=criterion, 
                        train_dataloader=dummy_dataloader, device=device, **kwargs)
        
        # Initialize k-fold splitter
        self.kfold = KFold(n_splits=n_splits, shuffle=shuffle)

    def train_fold(self, train_idx, val_idx, num_epochs, early_stopping=None):
        """
        Train the model on a specific fold.
        
        Args:
            train_idx (array-like): Indices for training data
            val_idx (array-like): Indices for validation data
            num_epochs (int): Number of epochs to train
            early_stopping (int, optional): Number of epochs to wait before early stopping
        """
        # Create train and validation datasets for this fold
        train_subset = Subset(self.dataset, train_idx)
        val_subset = Subset(self.dataset, val_idx)
        
        # Create data loaders for this fold
        self.train_dl = DataLoader(train_subset, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=kepler_collate_fn)
        self.val_dl = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False, collate_fn=kepler_collate_fn)
        
        # Update samplers
        self.train_sampler = self.get_sampler_from_dataloader(self.train_dl)
        self.val_sampler = self.get_sampler_from_dataloader(self.val_dl)
        
        # Train the model for this fold
        return self.fit(num_epochs, self.device, early_stopping=early_stopping)

    def run_kfold(self, num_epochs, early_stopping=None):
        """
        Run k-fold cross validation.
        
        Args:
            num_epochs (int): Number of epochs to train each fold
            early_stopping (int, optional): Number of epochs to wait before early stopping
        
        Returns:
            dict: Dictionary containing results from all folds
        """
        fold_results = []
        
        # Get indices for all samples
        indices = np.arange(len(self.dataset))
        
        # Run training for each fold
        for fold, (train_idx, val_idx) in enumerate(self.kfold.split(indices)):
            print(f"\nTraining Fold {fold+1}/{self.n_splits}")
            
            # Reset model weights
            self.model.apply(self._weight_reset)
            
            # Reset optimizer
            # if hasattr(self, 'optimizer'):
            #     self.optimizer.state = {}
            #     for group in self.optimizer.param_groups:
            #         group['lr'] = group['initial_lr']
            #
            # Train the fold
            fold_result = self.train_fold(train_idx, val_idx, num_epochs, early_stopping)
            fold_results.append(fold_result)
            
            # Save fold results
            if self.log_path is not None:
                fold_path = f'{self.log_path}/{self.exp_num}/fold_{fold+1}'
                os.makedirs(fold_path, exist_ok=True)
                torch.save(self.model.state_dict(), f'{fold_path}/model.pth')
                
                with open(f'{fold_path}/results.json', 'w') as f:
                    json.dump(fold_result, f, indent=2)

        # Calculate average metrics across folds
        avg_metrics = self._calculate_average_metrics(fold_results)
        
        return {
            'fold_results': fold_results,
            'average_metrics': avg_metrics
        }

    @staticmethod
    def _weight_reset(m):
        """
        Reset model weights.
        """
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()

    def _calculate_average_metrics(self, fold_results):
        """
        Calculate average metrics across all folds.
        """
        avg_metrics = {
            'train_loss': np.mean([np.mean(fold['train_loss']) for fold in fold_results]),
            'val_loss': np.mean([np.mean(fold['val_loss']) for fold in fold_results]),
            'train_acc': np.mean([np.mean(fold['train_acc']) for fold in fold_results]),
            'val_acc': np.mean([np.mean(fold['val_acc']) for fold in fold_results])
        }
        return avg_metrics

    def train_final_model_and_test(self, test_dataloader, num_epochs, early_stopping=None):
        """
        Train a final model on the entire dataset and evaluate on the test set.
        
        Args:
            test_dataloader (DataLoader): DataLoader for the test set
            num_epochs (int): Number of epochs to train the final model
            early_stopping (int, optional): Number of epochs to wait before early stopping
            
        Returns:
            dict: Dictionary containing training results and test metrics
        """
        print("\nTraining final model on all training data...")
        
        # Reset model weights
        self.model.apply(self._weight_reset)
        
        # Reset optimizer
        # if hasattr(self, 'optimizer'):
        #     self.optimizer.state = {}
        #     for group in self.optimizer.param_groups:
        #         group['lr'] = group['initial_lr']
        #
        # Create dataloader for all training data
        self.train_dl = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=kepler_collate_fn)
        self.val_dl = self.train_dl  # Use same data for validation during training
        
        # Update samplers
        self.train_sampler = self.get_sampler_from_dataloader(self.train_dl)
        self.val_sampler = self.get_sampler_from_dataloader(self.val_dl)
        
        # Train the final model
        training_results = self.fit(num_epochs, self.device, early_stopping=early_stopping)
        
        # Save the final model
        if self.log_path is not None:
            final_model_path = f'{self.log_path}/{self.exp_num}/final_model'
            os.makedirs(final_model_path, exist_ok=True)
            torch.save(self.model.state_dict(), f'{final_model_path}/model.pth')
            
            with open(f'{final_model_path}/training_results.json', 'w') as f:
                json.dump(training_results, f, indent=2)
        
        # Evaluate on test set
        print("\nEvaluating final model on test set...")
        test_metrics = self.evaluate_on_test_set(test_dataloader)
        
        return {
            'train_results': training_results,
            'test_results': test_metrics
        }

    def evaluate_on_test_set(self, test_dataloader):
        """
        Evaluate the trained model on a test set.
        
        Args:
            test_dataloader (DataLoader): DataLoader for the test set
            
        Returns:
            dict: Dictionary containing test metrics
        """
        self.model.eval()
        test_loss = []
        all_accs = torch.zeros(self.output_dim, device=self.device)
        total = 0
        
        print("Running evaluation on test set...")
        
        pbar = tqdm(test_dataloader)
        for i, batch in enumerate(pbar):
            lc, spec, _, _, y, info = batch
            spec = spec.to(self.device)
            lc = lc.to(self.device)
            b, _, _ = lc.shape
            if isinstance(y, tuple):
                y = torch.stack(y)     
            y = y.to(self.device)
            with torch.no_grad():
                y_pred = self.model(lc.float(), spec.float())
                y_pred = y_pred.reshape(b, -1, self.output_dim)
                
            if len(y.shape) == 1:
                y = y.unsqueeze(1)
                
            loss = 0
            for i in range(self.output_dim):
                y_pred_i = y_pred[:, :, i]
                y_i = y[:, i]
                loss += self.criterion(y_pred_i, y_i)
            loss /= self.output_dim
            
            test_loss.append(loss.item())
            
            y_pred_mean = y_pred[:, y_pred.shape[1]//2, :]
            diff = torch.abs(y_pred_mean - y)
            acc = (diff < (y/10)).sum(0)
            all_accs = all_accs + acc
            total += len(y)
        
        # Calculate average metrics
        avg_test_loss = np.mean(test_loss)
        avg_test_acc = all_accs.cpu().numpy() / total
        
        print(f"Test Loss: {avg_test_loss:.6f}")
        print(f"Test Accuracy: {avg_test_acc}")
        
        return {
                'y': y,
                'y_pred': y_pred,
            'test_loss': avg_test_loss,
            'test_acc': avg_test_acc.tolist(),
        }
     
    

class MaskedSSLTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def train_batch(self,batch, batch_idx, device):
        """
        Trains the model for one batch.
        """
        # with autocast():
        x, y, _, mask, info,_ = batch
        y, mask, x = y.to(device), mask.to(device), x.to(device)
        out = self.model(x, y)
        if x.isnan().sum() > 0:
            print("nan in x, idx: ", batch_idx)
        if y.isnan().sum() > 0:
            print("nan in y, idx: ", batch_idx)
        if out.isnan().sum() > 0:
            print("nan in out, idx: ", batch_idx)
        self.logger(f'max vals {out.max()} , {y.max()}')
        loss = self.criterion(out, y)
        
        if loss.isnan():
            print("nan in loss, idx: ", batch_idx)
            print("out range", out.min(), out.max())
            print("y range", y.min(), y.max())
            
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            if (batch_idx + 1) % self.accumulation_step == 0:
                # Add gradient clipping before optimizer step
                if self.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    self.check_gradients()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                if self.scheduler is not None:
                    self.scheduler.step()
                self.scaler.update()
        else:
            loss.backward()
            if (batch_idx + 1) % self.accumulation_step == 0:
                # Add gradient clipping before optimizer step
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
        acc = self.mask_accuracy(out, y.squeeze(), mask.squeeze())
        # if self.wandb:
        #     wandb.log({"train_loss": loss.item(), "train_acc": acc})
        return loss, acc, y
    
    def eval_batch(self, batch, batch_idx, device):
        """
        Evaluates the model for one batch.
        """
        x, y, _, mask,info,_ = batch
        y, mask, x = y.to(device), mask.to(device), x.to(device)
        with torch.no_grad():
            out = self.model(x, y)
        loss = self.criterion(out, y)
        acc = self.mask_accuracy(out, y.squeeze(), mask.squeeze())
        # if self.wandb:
        #     wandb.log({"val_loss": loss.item(), "val_acc": acc})
        return loss, acc, y
    
    def mask_accuracy(self, result, target, inverse_token_mask, epsilon=1e-5):
        # print(inverse_token_mask.shape, result.shape, target.shape)
        r = result.masked_select(inverse_token_mask)
        t = target.masked_select(inverse_token_mask)
        s = (torch.abs(r - t) < epsilon).sum()
        return s / inverse_token_mask.sum()


class JEPATrainer(Trainer):
    def __init__(self, use_w=True, lc_reg_idx=-1, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.use_w = use_w
        self.lc_reg_idx = lc_reg_idx
        self.alpha = alpha
        self.lc_losses = []
        self.spec_losses = []
        self.jepa_losses = []
        
    def train_batch(self,batch, batch_idx, device):
        lc, spectra, y, lc_target, spectra_target, info = batch 
        lc, spectra, y, lc_target, spectra_target = lc.to(device), spectra.to(device), y.to(device), lc_target.to(device), spectra_target.to(device)
        w = None
        if self.use_w:
            w = torch.stack([i['w'] for i in info]).to(device)
        out = self.model(lc ,spectra, w=w)
        y_lc = y[:, self.lc_reg_idx:]
        y_spec = y[:, :self.lc_reg_idx]
        pred_lc = out['lc_pred'].view(lc.shape[0], -1, self.num_quantiles)
        pred_spec = out['spectra_pred'].view(lc.shape[0], -1, self.num_quantiles)
        lc_loss = self.criterion(pred_lc, y_lc)
        lc_loss = lc_loss.nan_to_num(0).mean()
        spec_loss = self.criterion(pred_spec, y_spec)
        spec_loss = spec_loss.nan_to_num(0).mean()
        reg_loss = lc_loss + spec_loss
        self.lc_losses.append(lc_loss)
        self.spec_losses.append(spec_loss)
        
        jepa_loss = out['loss']
        self.jepa_losses.append(jepa_loss)
        loss = reg_loss * self.alpha + jepa_loss * (1 - self.alpha)
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        
        if batch_idx % 200 == 0:
            print('separate losses - ', lc_loss.item(), spec_loss.item(), jepa_loss.item())
        self.train_aux_loss_1.append(loss.item())
        self.train_aux_loss_2.append(reg_loss.item())

        lc_pred_med = pred_lc[: ,:, self.num_quantiles // 2]
        spec_pred_med = pred_spec[: ,:, self.num_quantiles // 2]
        pred_med = torch.cat([spec_pred_med, lc_pred_med], dim=-1)
        acc = (torch.abs(pred_med - y) < y * 0.1).sum(0)

        return loss, acc, y
    
    def eval_batch(self,batch, batch_idx, device):
        lc, spectra, y, lc_target, spectra_target, info = batch 
        lc, spectra, y, lc_target, spectra_target = lc.to(device), spectra.to(device), y.to(device), lc_target.to(device), spectra_target.to(device)
        w = None
        if self.use_w:
            w = torch.stack([i['w'] for i in info]).to(device)
        with torch.no_grad():
            out = self.model(lc ,spectra, w=w)
        y_lc = y[:, self.lc_reg_idx:]
        y_spec = y[:, :self.lc_reg_idx]
        pred_lc = out['lc_pred'].view(lc.shape[0], -1, self.num_quantiles)
        pred_spec = out['spectra_pred'].view(lc.shape[0], -1, self.num_quantiles)
        lc_loss = self.criterion(pred_lc, y_lc)
        lc_loss = lc_loss.nan_to_num(0).mean()
        spec_loss = self.criterion(pred_spec, y_spec)
        spec_loss = spec_loss.nan_to_num(0).mean()
        reg_loss = lc_loss + spec_loss
        jepa_loss = out['loss']
        loss = reg_loss * self.alpha + jepa_loss * (1 - self.alpha)

        self.val_aux_loss_1.append(loss.item())
        self.val_aux_loss_2.append(reg_loss.item())

        lc_pred_med = pred_lc[: ,:, self.num_quantiles // 2]
        spec_pred_med = pred_spec[: ,:, self.num_quantiles // 2]
        pred_med = torch.cat([spec_pred_med, lc_pred_med], dim=-1)
        acc = (torch.abs(pred_med - y) < y * 0.1).sum(0)

        return loss, acc, y

class ContrastiveTrainer(Trainer):
    def __init__(self, temperature=1, full_input=False, stack_pairs=True, only_lc=True,
                 use_w=False, use_pred_coeff=False, pred_coeff_val=None, ssl_weight=0.5,
                 weight_decay=False,   **kwargs):
        super().__init__(**kwargs)
        self.stack_pairs = stack_pairs
        self.temperature = temperature
        self.use_w = use_w
        self.use_pred_coeff = use_pred_coeff
        self.pred_coeff_val = pred_coeff_val
        self.ssl_weight = ssl_weight
        self.weight_decay = weight_decay
        self.full_input = full_input
        self.only_lc = only_lc
        
        
    def train_batch(self, batch, batch_idx, device):
        self.optimizer.zero_grad()  # Add this if not done elsewhere
        
        lc, spectra, y, lc_target, spectra_target, info = batch 
        lc, spectra, y, lc_target, spectra_target = lc.to(device), spectra.to(device), y.to(device), lc_target.to(device), spectra_target.to(device)
        # Forward pass
        if self.stack_pairs:
            x = torch.cat((lc, lc_target), dim=0)
            out = self.model(x, temperature=self.temperature)
        elif self.only_lc:
            out = self.model(lc, lc_target) 
        elif self.use_w:
            w = torch.stack([i['w'] for i in info]).to(device)
            pred_coeff = float(self.pred_coeff_val) if self.pred_coeff_val != 'None' else max(1 - batch_idx / (3 * len(self.train_dl)), 0.5)
            if self.full_input:
                out = self.model(lightcurves=lc, spectra=spectra,
                                 lightcurves2=lc_target, spectra2=spectra_target,
                                  w=w, pred_coeff=pred_coeff)
            else:
                out = self.model(lc, spectra, w=w, pred_coeff=pred_coeff)
        else:
            out = self.model(lc, spectra)

        loss = out['loss']
       
        acc = 0
        if self.criterion is not None:
            preds = out['preds']
            preds = preds.view(lc.shape[0], -1, self.num_quantiles)
            reg_loss = self.criterion(preds, y)

            # print("shapes ", preds.shape, y.shape,
            #  reg_loss.shape, "nans in y: ",
            #  "nans in preds: ", preds.isnan().sum(),
            #   y.isnan().sum(), "nans in loss: ",
            #    reg_loss.isnan().sum())

            reg_loss = reg_loss.nan_to_num(0).mean()
            self.train_aux_loss_1.append(loss.item())
            self.train_aux_loss_2.append(reg_loss.item())
            # if self.weight_decay:
            #     cur_weight = self.ssl_weight - 0.025 * self.epoch
            loss = loss  * self.ssl_weight + reg_loss * (1 - self.ssl_weight)
            preds_med = preds[: ,:, self.num_quantiles // 2]
            acc = (torch.abs(preds_med - y) < y * 0.1).sum(0)
        else:
            if 'loss_pred' in out.keys():
                self.train_aux_loss_1.append(out['loss_pred'].item())
            if 'loss_contrastive' in out.keys():
                self.train_aux_loss_2.append(out['loss_contrastive'].item())
        if ('z' in out.keys()) or ('q' in out.keys()):
            logits_key = 'z' if 'z' in out.keys() else 'q'
            logits = out[logits_key]
            norm = torch.norm(logits, dim=1, keepdim=True)
            norm_logits = logits / norm
            logits_mean = norm_logits.mean(-1)
            logits_std = norm_logits.std(-1)
            self.train_logits_mean.extend(logits_mean.cpu().tolist())
            self.train_logits_std.extend(logits_std.cpu().tolist())
        

        # Backward pass with gradient scaling
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            if (batch_idx + 1) % self.accumulation_step == 0:
                self.scaler.unscale_(self.optimizer)
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step()
                # self.check_gradients()  # Monitor gradients
        else:
            loss.backward()
            if (batch_idx + 1) % self.accumulation_step == 0:
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                # self.check_gradients()  # Monitor gradients

        return loss, acc, y

    def eval_batch(self,batch, batch_idx, device):
        lc, spectra, y, lc_target, spectra_target, info = batch 
        lc, spectra, y, lc_target, spectra_target = lc.to(device), spectra.to(device), y.to(device), lc_target.to(device), spectra_target.to(device)
        with torch.no_grad():
            if self.stack_pairs:
                x = torch.cat((lc, lc_target), dim=0)
                out = self.model(x, temperature=self.temperature)
            elif self.only_lc:
                out = self.model(lc, lc_target) 
            elif self.use_w:
                w = torch.stack([i['w'] for i in info]).to(device)
                pred_coeff = float(self.pred_coeff_val) if self.pred_coeff_val != 'None' else max(1 - batch_idx / (3 * len(self.train_dl)), 0.5)
                if self.full_input:
                    out = self.model(lightcurves=lc, spectra=spectra,
                                 lightcurves2=lc_target, spectra2=spectra_target,
                                  w=w, pred_coeff=pred_coeff)
                else:
                    out = self.model(lc, spectra, w=w, pred_coeff=pred_coeff)
            else:
                out = self.model(lc, spectra)   
        loss = out['loss']
        acc = 0
        if self.criterion is not None:
            preds = out['preds']
            preds = preds.view(lc.shape[0], -1, self.num_quantiles)
            reg_loss = self.criterion(preds, y)
            reg_loss = reg_loss.nan_to_num(0).mean()
            self.val_aux_loss_1.append(loss.item())
            self.val_aux_loss_2.append(reg_loss.item())
            loss = loss  * self.ssl_weight + reg_loss * (1 - self.ssl_weight)
            preds_med = preds[: ,:, self.num_quantiles // 2]
            acc = (torch.abs(preds_med - y) < y * 0.1).sum(0)
        else:
            if 'loss_pred' in out.keys():
                self.val_aux_loss_1.append(out['loss_pred'].item())
            if 'loss_contrastive' in out.keys():
                self.val_aux_loss_2.append(out['loss_contrastive'].item())
        if ('z' in out.keys()) or ('q' in out.keys()):
            logits_key = 'z' if 'z' in out.keys() else 'q'
            logits = out[logits_key]
            logits_mean = logits.mean(-1)
            logits_std = logits.std(-1)
            self.val_logits_mean.extend(logits_mean.cpu().tolist())
            self.val_logits_std.extend(logits_std.cpu().tolist())
        return loss, acc, y
    
    def predict(self, test_dataloader, device, load_best=False):
        """
        Returns the predictions of the model on the given dataset.
        """
        self.model.eval()
        preds = np.zeros((0, self.output_dim, self.num_quantiles))
        targets = np.zeros((0, self.output_dim))
        tot_kic = []
        tot_teff = []
        aggregated_info = {}
        pbar = tqdm(test_dataloader)

        for i,(lc, spectra, y, lc_target, spectra_target, info) in enumerate(pbar):
            lc, spectra, y, lc_target, spectra_target = lc.to(device), spectra.to(device), y.to(device), lc_target.to(device), spectra_target.to(device)
            b = lc.shape[0]
            for item in info:
                for key, value in item.items():
                    # Check if value is a scalar (not an array/tensor)
                    if np.isscalar(value):
                        if key not in aggregated_info:
                            aggregated_info[key] = []
                        aggregated_info[key].append(value)
            with torch.no_grad():
                if self.stack_pairs:
                    x = torch.cat((lc, lc_target), dim=0)
                    out = self.model(x, temperature=self.temperature)
                elif self.only_lc:
                    out = self.model(lc, lc_target) 
                elif self.use_w:
                    w = torch.stack([i['w'] for i in info]).to(device)
                    pred_coeff = float(self.pred_coeff_val) if self.pred_coeff_val != 'None' else max(1 - i / (3 * len(self.train_dl)), 0.5)
                    if self.full_input:
                        out = self.model(lightcurves=lc, spectra=spectra,
                                    lightcurves2=lc_target, spectra2=spectra_target,
                                    w=w, pred_coeff=pred_coeff)
                    else:
                        out = self.model(lc, spectra, w=w, pred_coeff=pred_coeff)
                else:
                    out = self.model(lc, spectra)
            acc = 0
            loss = out['loss'] if 'loss' in out.keys() else 0
            if self.criterion is not None:
                y_pred = out['preds']
                y_pred = y_pred.view(lc.shape[0], -1, self.num_quantiles)
                preds = np.concatenate((preds, y_pred.cpu().numpy()))
                reg_loss = self.criterion(y_pred, y)
                reg_loss = reg_loss.nan_to_num(0).mean()
                # self.val_aux_loss_1.append(loss.item())
                # self.val_aux_loss_2.append(reg_loss.item())
                loss = loss  * self.ssl_weight + reg_loss * (1 - self.ssl_weight)
                preds_med = y_pred[: ,:, self.num_quantiles // 2]
                acc = (torch.abs(preds_med - y) < y * 0.1).sum(0)
            targets = np.concatenate((targets, y.cpu().numpy()))
            if i > self.max_iter:
                break
            pbar.set_description(f"test_loss: {loss.item():.4f}, test_acc: {acc}")
        print("target len: ", len(targets), "dataset: ", len(test_dataloader.dataset))
        return preds, targets, aggregated_info

class ClassificationTrainer(Trainer):
    def __init__(self, num_cls=1, use_w=False, **kwargs):
        super().__init__(**kwargs)
        self.num_cls = num_cls
        self.use_w = use_w

    def train_batch(self, batch, batch_idx, device):
        lc, spectra,y , _, _,info = batch
        # print("numberof 1s: ", (y == 1).sum(), "number of 0s: ", (y == 0).sum())
        lc, spectra, y = lc.to(device), spectra.to(device), y.to(device)
        b = lc.shape[0]
        W = None
        if self.use_w:
            w = torch.stack([i['w'] for i in info]).to(device)
        out = self.model(lc, spectra, w_tune=w)
        if isinstance(out, tuple):
            out = out[0]
        elif isinstance(out, dict):
            out = out['preds']
        # print("nans in y: ", torch.isnan(y).sum(), "nans in out: ", torch.isnan(out).sum())
        loss = self.criterion(out, y)
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        probs = torch.sigmoid(out)
        acc = ((probs > 0.5) == y).sum(0)
        # acc = (torch.abs((out - y)) < 0.1).sum(0)
        return loss, acc, y 

    def eval_batch(self, batch, batch_idx, device):
        lc, spectra,y , _, _,info = batch
        lc, spectra, y = lc.to(device), spectra.to(device), y.to(device)
        w = None
        if self.use_w:
            w = torch.stack([i['w'] for i in info]).to(device)
        out = self.model(lc, spectra, w_tune=w)
        if isinstance(out, tuple):
            out = out[0]
        elif isinstance(out, dict):
            out = out['preds']
        loss = self.criterion(out, y)
        acc = (torch.abs((out - y)) < 0.1).sum(0)
        return loss, acc, y
    
    def predict(self, test_dataloader, device, load_best=False):
        """
        Returns the predictions of the model on the given dataset.
        """
        self.model.eval()
        preds = np.zeros((0, self.num_cls))
        targets = np.zeros((0, self.num_cls))
        tot_kic = []
        tot_teff = []
        aggregated_info = {}
        pbar = tqdm(test_dataloader)

        for i,(lc, spectra, y, _, _, info, _) in enumerate(pbar):
            lc, spectra, y = lc.to(device), spectra.to(device), y.to(device)
            b = lc.shape[0]
            for item in info:
                for key, value in item.items():
                    # Check if value is a scalar (not an array/tensor)
                    if np.isscalar(value):
                        if key not in aggregated_info:
                            aggregated_info[key] = []
                        aggregated_info[key].append(value)
            with torch.no_grad():
                w = None
                if self.use_w:
                    w = torch.stack([i['w'] for i in info]).to(device)
                out = self.model(lc, spectra, w_tune=w)
                probs = torch.softmax(out, dim=1)
            preds = np.concatenate((preds, probs.cpu().numpy()))
            targets = np.concatenate((targets, y.cpu().numpy()))
            if i > self.max_iter:
                break
        print("target len: ", len(targets), "dataset: ", len(test_dataloader.dataset))
        return preds, targets, aggregated_info

class RegressorTrainer(Trainer):
    def __init__(self, use_w, only_lc=False, **kwargs):
        super().__init__(**kwargs)
        self.use_w = use_w
        self.only_lc = only_lc
    
    def train_batch(self, batch, batch_idx, device):
        lc, spectra, y , lc2, spectra2,info = batch
        lc, spectra, y = lc.to(device), spectra.to(device), y.to(device)
        b = lc.shape[0]
        w = None
        print("lcs: ", lc.shape, lc2.shape, lc.isnan().sum(), lc2.isnan().sum())
        if self.use_w:
            w = torch.stack([i['w'] for i in info]).to(device)
            out = self.model(lc, spectra, w_tune=w) 
        elif self.only_lc:
            out = self.model(lc, lc2)
        else:
            out = self.model(lc, spectra)
        if isinstance(out, tuple):
            out = out[0]
        elif isinstance(out, dict):
            out = out['preds']
        if self.num_quantiles > 1:
            out = out.view(b, -1, self.num_quantiles)
        else:
            out = out.squeeze(1)
            if len(y.shape) == 2 and y.shape[1] == 1:
                y = y.squeeze(1)
        # print(out.shape, y.shape, out.isnan().sum(), y.isnan().sum())
        loss = self.criterion(out, y)
        # print('shapes: ', x.shape, y.shape, out.shape, 'w: ', w.shape, 'loss: ', loss.shape)
        # loss = (loss * w.unsqueeze(-1)).mean(0).sum()
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
                self.scheduler.step()
        if self.num_quantiles > 1:
            out_median = out[..., out.shape[-1]//2]
            if (len(out_median.shape) == 2) and (len(y.shape) == 1):
                out_median = out_median.squeeze(1)
        else:
            out_median = out
        acc = (torch.abs(out_median - y) < y * 0.1).sum(0)
        return loss, acc, y

    def eval_batch(self, batch, batch_idx, device):
        lc, spectra,y , lc2, spectra2,info = batch
        lc, spectra, y = lc.to(device), spectra.to(device), y.to(device)
        b = lc.shape[0]
        with torch.no_grad():
            w = None
            if self.use_w:
                w = torch.stack([i['w'] for i in info]).to(device)
                out = self.model(lc, spectra, w_tune=w) 
            elif self.only_lc:
                out = self.model(lc, lc2)
            else:
                out = self.model(lc, spectra)
        if isinstance(out, tuple):
            out = out[0]
        elif isinstance(out, dict):
            out = out['preds']
        if self.num_quantiles > 1:
            out = out.view(b, -1, self.num_quantiles)
        else:
            out = out.squeeze(1)
            if len(y.shape) == 2 and y.shape[1] == 1:
                y = y.squeeze(1)
        if isinstance(out, tuple):
            out = out[0]
        elif isinstance(out, dict):
            out = out['preds']
        if self.num_quantiles > 1:
            out = out.view(b, -1, self.num_quantiles)
        else:
            out = out.squeeze(1)
        loss = self.criterion(out, y)
        # loss = (loss * w.unsqueeze(-1)).mean(0).sum()
        # if self.wandb:
        #     wandb.log({"val_loss": loss.item()})

        if self.num_quantiles > 1:
            out_median = out[..., out.shape[-1]//2]
            if (len(out_median.shape) == 2) and (len(y.shape) == 1):
                out_median = out_median.squeeze(1)
        else:
            out_median = out
        acc = (torch.abs(out_median - y) < y * 0.1).sum(0)
        return loss, acc, y

    def predict(self, test_dataloader, device, load_best=False):
        """
        Returns the predictions of the model on the given dataset.
        """
        self.model.eval()
        preds = np.zeros((0, self.output_dim, self.num_quantiles))
        targets = np.zeros((0, self.output_dim))
        tot_kic = []
        tot_teff = []
        aggregated_info = {}
        pbar = tqdm(test_dataloader)

        for i,(lc, spectra, y, _, _, info, _) in enumerate(pbar):
            lc, spectra, y = lc.to(device), spectra.to(device), y.to(device)
            b = lc.shape[0]
            for item in info:
                for key, value in item.items():
                    # Check if value is a scalar (not an array/tensor)
                    if np.isscalar(value):
                        if key not in aggregated_info:
                            aggregated_info[key] = []
                        aggregated_info[key].append(value)
            with torch.no_grad():
                w = None
                if self.use_w:
                    w = torch.stack([i['w'] for i in info]).to(device)
                out = self.model(lc, spectra, w_tune=w)
                if isinstance(out, tuple):
                    out = out[0]
                elif isinstance(out, dict):
                    out = out['preds']
                y_pred = out.view(b, -1, self.num_quantiles)
            out_diff = int(y.shape[1] - y_pred.shape[1])
            y = y[:, out_diff:]
            preds = np.concatenate((preds, y_pred.cpu().numpy()))
            targets = np.concatenate((targets, y.cpu().numpy()))
            if i > self.max_iter:
                break
        print("target len: ", len(targets), "dataset: ", len(test_dataloader.dataset))
        return preds, targets, aggregated_info

class MaskedRegressorTrainer(Trainer):
    def __init__(self, w_name, w_init_val, ssl_criterion, ssl_weight=0.5, **kwargs):
        super().__init__(**kwargs)
        self.w_name = w_name
        self.ssl_criterion = ssl_criterion
        self.w_init_val = w_init_val
        self.ssl_weight = ssl_weight  # Weight to balance between SSL and regression
        self.drop_first_y = False
        
    def train_batch(self, batch, batch_idx, device):
        const = self.w_init_val/(1+self.epoch)
        x_masked, x, y, mask, _, info= batch
        x_masked, x, y, mask = x_masked.to(device), x.to(device), y.to(device), mask.to(device)
        b = x_masked.shape[0]
        
        # Get proximity weights for regression
        if self.w_name is None:
            w = torch.ones(x_masked.size(0)).to(device)
        else:
            w = torch.tensor([i[self.w_name] for i in info]).to(device)
        # w = const * torch.exp(w*const)
        
        # Forward pass for both tasks
        reg_out,ssl_out,_  = self.model(x_masked, x)

        # print('shapes: ', x_masked.shape, x.shape, y.shape, mask.shape, reg_out.shape, ssl_out.shape)          
        # Calculate SSL loss (masked filling)
        if (len(x.shape) == 3) and (x.shape[1] > 1):
            x = x[:, 0, :]
        ssl_loss = self.ssl_criterion(ssl_out, x)
        ssl_acc = self.mask_accuracy(ssl_out, x, mask)
        
        # Calculate regression loss
        reg_out = reg_out.view(b, -1, self.num_quantiles)
        out_diff = int(y.shape[1] - reg_out.shape[1])
        y = y[:, out_diff:]
        if self.drop_first_y:
            reg_out = reg_out[:, 1:]
            y = y[:, 1:]
        reg_loss = self.criterion(reg_out, y)
        reg_loss = (reg_loss * w.unsqueeze(-1)).mean()
        out_median = reg_out[..., reg_out.shape[-1]//2]
        reg_acc = (torch.abs(out_median - y) < y * 0.1).sum(0)
        self.train_aux_loss_1.append(ssl_loss.item())
        self.train_aux_loss_2.append(reg_loss.item())
        # Combine losses
        loss = (self.ssl_weight * ssl_loss) + ((1 - self.ssl_weight) * reg_loss)
        
         # Backward pass with gradient scaling
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            if (batch_idx + 1) % self.accumulation_step == 0:
                self.scaler.unscale_(self.optimizer)
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.check_gradients()  # Monitor gradients
        else:
            loss.backward()
            if (batch_idx + 1) % self.accumulation_step == 0:
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.check_gradients()  # Monitor gradients
                
        return loss, reg_acc, x

    def eval_batch(self, batch, batch_idx, device):
        const = self.w_init_val/(1+self.epoch)
        x_masked, x, y, mask, _, info = batch
        x_masked, x, y, mask = x_masked.to(device), x.to(device), y.to(device), mask.to(device)
        b = x_masked.shape[0]
        
        if self.w_name is None:
            w = torch.ones(x_masked.size(0)).to(device)
        else:
            w = torch.tensor([i[self.w_name] for i in info]).to(device)
        # w = const * torch.exp(w*const)
        
        with torch.no_grad():
            reg_out,ssl_out,_  = self.model(x_masked, x)  # Masked filling task
            
            ssl_loss = self.ssl_criterion(ssl_out, x)
            ssl_acc = self.mask_accuracy(ssl_out, x, mask)

            reg_out = reg_out.view(b, -1, self.num_quantiles)
            reg_out = reg_out.view(b, -1, self.num_quantiles)
            out_diff = int(y.shape[1] - reg_out.shape[1])
            y = y[:, out_diff:]
            if self.drop_first_y:
                reg_out = reg_out[:, 1:]
                y = y[:, 1:]
            reg_loss = self.criterion(reg_out, y)
            reg_loss = (reg_loss * w.unsqueeze(-1)).mean()
            out_median = reg_out[..., reg_out.shape[-1]//2]
            reg_acc = (torch.abs(out_median - y) < y * 0.1).sum(0)
            self.val_aux_loss_1.append(ssl_loss.item())
            self.val_aux_loss_2.append(reg_loss.item())

            total_loss = (self.ssl_weight * ssl_loss) + ((1 - self.ssl_weight) * reg_loss)
            
        return total_loss, reg_acc, x
        
    def mask_accuracy(self, result, target, inverse_token_mask, epsilon=1e-5):
        r = result.masked_select(inverse_token_mask)
        t = target.masked_select(inverse_token_mask)
        s = (torch.abs(r - t) < epsilon).sum()
        return s / inverse_token_mask.sum()

    def predict(self, test_dataloader, device):
        """
        Returns the predictions of the model on the given dataset.
        """
        self.model.eval()
        preds = np.zeros((0, self.output_dim, self.num_quantiles))
        targets = np.zeros((0, self.output_dim))
        tot_kic = []
        tot_teff = []
        aggregated_info = {}
        pbar = tqdm(test_dataloader)

        for i,(x_masked, x, y, mask, _ , info) in enumerate(pbar):
            x_masked, x, y, mask = x_masked.to(device), x.to(device), y.to(device), mask.to(device)
            b = x_masked.shape[0]
            for item in info:
                for key, value in item.items():
                    # Check if value is a scalar (not an array/tensor)
                    if np.isscalar(value):
                        if key not in aggregated_info:
                            aggregated_info[key] = []
                        aggregated_info[key].append(value)
            with torch.no_grad():
                y_pred, _, _ = self.model(x_masked, x)
                y_pred = y_pred.view(b, -1, self.num_quantiles)
            out_diff = int(y.shape[1] - y_pred.shape[1])
            y = y[:, out_diff:]
            if self.drop_first_y:
                reg_out = reg_out[:, 1:]
                y = y[:, 1:]
            preds = np.concatenate((preds, y_pred.cpu().numpy()))
            targets = np.concatenate((targets, y.cpu().numpy()))
            if i > self.max_iter:
                break
        print("target len: ", len(targets), "dataset: ", len(test_dataloader.dataset))
        return preds, targets, aggregated_info

class ContrastiveRegressorTrainer(Trainer):
    def __init__(self, temperature=1, stack_pairs=False,
                 use_w=False, use_pred_coeff=False, pred_coeff_val=None, ssl_weight=0.5,
                   **kwargs):
        super().__init__(**kwargs)
        self.stack_pairs = stack_pairs
        self.temperature = temperature
        self.use_w = use_w
        self.use_pred_coeff = use_pred_coeff
        self.pred_coeff_val = pred_coeff_val
        self.ssl_weight = ssl_weight
        print("ssl_weight: ", self.ssl_weight)
        
    def train_batch(self, batch, batch_idx, device):
        self.optimizer.zero_grad()  # Add this if not done elsewhere
        
        x1, x2, y, _, _, info = batch 
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        
        # Forward pass
        if self.stack_pairs:
            x = torch.cat((x1, x2), dim=0)
            out = self.model(x, temperature=self.temperature)
        elif self.use_w:
            w = torch.stack([i['w'] for i in info]).to(device)
            pred_coeff = float(self.pred_coeff_val) if self.pred_coeff_val != 'None' else max(1 - batch_idx / (3 * len(self.train_dl)), 0.5)
            out = self.model(x1, x2, w=w, pred_coeff=pred_coeff) if self.use_pred_coeff else self.model(x1, x2, w=w)
        else:
            out = self.model(x1, x2)    

        loss = out['loss']
        acc = 0
        preds = out['output_reg']
        preds = preds.view(x1.shape[0], -1, self.num_quantiles)
        reg_loss = self.criterion(preds, y)
        self.train_aux_loss_1.append(loss.item())
        self.train_aux_loss_2.append(reg_loss.item())
        loss = loss  * self.ssl_weight + reg_loss * (1 - self.ssl_weight)
        preds_med = preds[: ,:, self.num_quantiles // 2]
        acc = (torch.abs(preds_med - y) < y * 0.1).sum(0)
        return loss, acc, y

    def eval_batch(self,batch, batch_idx, device):
        x1, x2, y, _, _, info = batch 
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        with torch.no_grad():
            if self.stack_pairs:
                x = torch.cat((x1, x2), dim=0)
                out = self.model(x, temperature=self.temperature)
            if self.use_w:
                w = torch.stack([i['w'] for i in info]).to(device)
                if self.use_pred_coeff:                    
                    if self.pred_coeff_val != 'None':
                        pred_coeff = float(self.pred_coeff_val)
                    else:
                        pred_coeff = max(1 - batch_idx / (3 * len(self.train_dl)), 0.5)
                    out = self.model(x1, x2, w=w, pred_coeff=pred_coeff)
                else:
                    out = self.model(x1, x2, w=w)
            else:
                out = self.model(x1, x2)
        loss = out['loss']
        acc = 0
        preds = out['output_reg']
        preds = preds.view(x1.shape[0], -1, self.num_quantiles)
        reg_loss = self.criterion(preds, y)
        self.val_aux_loss_1.append(loss.item())
        self.val_aux_loss_2.append(reg_loss.item())
        loss = loss  * self.ssl_weight + reg_loss * (1 - self.ssl_weight)
        preds_med = preds[: ,:, self.num_quantiles // 2]
        acc = (torch.abs(preds_med - y) < y * 0.1).sum(0)
        return loss, acc, y

class MultiResolutionTrainer(MaskedRegressorTrainer):
    def __init__(self, high_res_train, high_res_val, lambda_high_res=1, **kwargs):
        super().__init__(**kwargs)
        self.high_res_train = high_res_train
        self.high_res_val = high_res_val
        self.lambda_high_res = lambda_high_res
    
    def train_epoch(self, device, epoch):
        self.epoch = epoch  # Make sure epoch is set
        if self.train_sampler is not None:
            try:
                self.train_sampler.set_epoch(epoch)
            except AttributeError:
                pass
        
        self.model.train()
        train_losses = []
        train_losses_high = []
        all_accs = 0
        all_accs_high = []
        
        high_res_iterator = iter(self.high_res_train)
        total_samples = 0

        pbar = tqdm(self.train_dl)
        
        for i, batch in enumerate(pbar):
            # Get high-res batch
            try:
                high_res_batch = next(high_res_iterator)
            except StopIteration:
                high_res_iterator = iter(self.high_res_train)
                high_res_batch = next(high_res_iterator)
            if self.optimizer is not None:
                self.optimizer.zero_grad()
            self.drop_first_y = True
            loss, acc , y = self.train_batch(batch, i, device)
            self.drop_first_y = False
            loss_high, acc_high, y_high = self.train_batch(high_res_batch, i, device)
            train_losses.append((loss.item() + loss_high.item()) / 2)
             #  add virtual vsini accuracy for low res
            acc = torch.cat((torch.tensor([acc_high[0]], device=acc_high.device) * len(y) / len(y_high), acc))  
            all_accs = all_accs + ( acc / len(y) + acc_high / len(y_high)) / 2
            pbar.set_description(f"train_loss:  {loss.item():.3f}, train_loss_high: {loss_high.item():.3f} "
                                f"train_acc: {acc}, train_acc_high: {acc_high},")      
            if i > self.max_iter:
                break
        return train_losses, all_accs / i
    
    def eval_epoch(self, device, epoch):
        """
        Evaluates the model for one epoch.
        """
        self.model.eval()
        val_loss = []
        val_acc = 0
        total = 0
        all_accs = torch.zeros(self.output_dim, device=device)
        high_res_iterator = iter(self.high_res_train)
        total_samples = 0

        pbar = tqdm(self.train_dl)
        for i, batch in enumerate(pbar):
            # Get high-res batch
            try:
                high_res_batch = next(high_res_iterator)
            except StopIteration:
                high_res_iterator = iter(self.high_res_train)
                high_res_batch = next(high_res_iterator)
            self.drop_first_y = True
            loss, acc, y = self.eval_batch(batch, i, device)
            self.drop_first_y = False
            loss_high, acc_high, y_high = self.eval_batch(high_res_batch, i, device)
            val_loss.append((loss.item() + loss_high.item()) / 2)
            #  add virtual vsini accuracy for low res
            acc = torch.cat((torch.tensor([acc_high[0]], device=acc_high.device) * len(y) / len(y_high), acc))    
            all_accs = all_accs + ( acc / len(y) + acc_high / len(y_high)) / 2
            pbar.set_description(f"val_loss:  {loss.item():.3f}, val_loss_high: {loss_high.item():.3f} "
                                f"val_acc: {acc}, val_acc_high: {acc_high},")
            if i > self.max_iter:
                break
        return val_loss, all_accs / i

class DualTrainer(Trainer):
    def __init__(self, trainer_lc, trainer_spec, lambda_dual,
                    high_res_lc=False, high_res_spec=True, **kwargs):
        super().__init__(**kwargs)
        self.trainer_lc = trainer_lc
        self.trainer_spec = trainer_spec
        self.lambda_dual = lambda_dual
        self.high_res_lc = high_res_lc
        self.high_res_spec = high_res_spec

    def high_res_batch(self, trainer, high_res_iterator, batch, i, device):
        try:
            hr_batch = next(high_res_iterator)
        except StopIteration:
            high_res_iterator = iter(trainer.high_res_train)
            hr_batch = next(high_res_iterator)
        trainer.drop_first_y = True
        loss, acc , y = trainer.train_batch(batch, i, device)
        trainer.drop_first_y = False
        loss_high, acc_high, y_high = trainer.train_batch(hr_batch, i, device)
        loss = (loss.item() + loss_high.item()) / 2
            #  add virtual vsini accuracy for low res
        acc = torch.cat((torch.tensor([acc_high[0]], device=acc_high.device) * len(y) / len(y_high), acc))
        return loss, acc, y 
        
    def train_epoch(self, device, epoch):
        self.epoch = epoch  # Make sure epoch is set
        if self.train_sampler is not None:
            try:
                self.train_sampler.set_epoch(epoch)
            except AttributeError:
                pass
        
        if self.high_res_lc:
            high_res_lc_iterator = iter(self.trainer_lc.high_res_train)
        if self.high_res_spec:
            high_res_spec_iterator = iter(self.trainer_spec.high_res_train)
        self.model.train()
        train_losses = []
        all_accs = 0
        total_samples = 0
        pbar = tqdm(self.train_dl)
        for i, batch in enumerate(pbar):
            if self.trainer_lc.optimizer is not None:
                self.trainer_lc.optimizer.zero_grad()
            if self.high_res_lc:
                loss_lc, acc_lc, y_lc = self.high_res_batch(self.trainer_lc, high_res_lc_iterator, batch, i, device)
            else:
                loss_lc, acc_lc, y_lc = self.trainer_lc.train_batch(batch, i, device)
            if self.trainer_spec.optimizer is not None:
                self.trainer_spec.optimizer.zero_grad()
            if self.high_res_spec:
                loss_spec, acc_spec, y_spec = self.high_res_batch(self.trainer_spec, high_res_spec_iterator, batch, i, device)
            else:
                loss_spec, acc_spec, y_spec = self.trainer_spec.train_batch(batch, i, device)
            loss = loss_lc + loss_spec
            loss_dual, acc_dual, y_dual = self.train_batch(batch, i, device)
            total_loss = loss_lc + loss_spec + self.lambda_dual * loss_dual
            train_losses.append(total_loss.item())
            all_accs = all_accs + (acc_lc + acc_spec + acc_dual) / 3
            pbar.set_description(f"train_loss_lc:  {loss_lc.item():.3f},  train_loss_spec:  {loss_spec.item():.3f}"
                                f"train_loss_dual: {loss_dual.item()}, train_acc_lc: {acc_lc}, train_acc_spec: {acc_spec},"
                                f"train_acc_dual: {acc_dual},")      
            if i > self.max_iter:
                break
        return train_losses, all_accs / i
    
    def eval_epoch(self, device, epoch):
        """
        Evaluates the model for one epoch.
        """
        self.model.eval()
        val_loss = []
        val_acc = 0
        total = 0
        all_accs = torch.zeros(self.output_dim, device=device)
        pbar = tqdm(self.val_dl)
        for i, batch in enumerate(pbar):
            loss_lc, acc_lc, y_lc = self.trainer_lc.eval_batch(batch, i, device)
            loss_spec, acc_spec, y_spec = self.trainer_spec.eval_batch(batch, i, device)
            loss = loss_lc + loss_spec
            loss_dual, acc_dual, y_dual = self.eval_batch(batch, i, device)
            total_loss = loss_lc + loss_spec + self.lambda_dual * loss_dual
            val_loss.append(total_loss.item())
            all_accs = all_accs + (acc_lc + acc_spec + acc_dual) / 3
            pbar.set_description(f"val_loss_lc:  {loss_lc.item():.3f},  val_loss_spec:  {loss_spec.item():.3f}"
                                f"val_loss_dual: {loss_dual.item()}, val_acc_lc: {acc_lc}, val_acc_spec: {acc_spec},"
                                f"val_acc_dual: {acc_dual},")      
            if i > self.max_iter:
                break
        return val_loss, all_accs / i
    

class LightSpecTrainer(Trainer):
    def train_batch(self,batch, batch_idx, device):
        start_time = time.time()
        lc, spec, lc2, spec2, info1, info2 = batch 
        lc, lc2, spec, spec2 = lc.to(device), lc2.to(device), spec.to(device), spec2.to(device)
        # print("before padd: ", lc.shape, spec.shape, lc2.shape, spec2.shape)
        spec = torch.nn.functional.pad(spec, (0, lc.shape[-1] - spec.shape[-1], 0,0))
        spec2 = torch.nn.functional.pad(spec2, (0, lc2.shape[-1] - spec2.shape[-1], 0,0))
        # print('after padd: ',  lc.shape, spec.shape, lc2.shape, spec2.shape)
        x1 = torch.cat((lc, spec.unsqueeze(1)), dim=1)
        x2 = torch.cat((lc2, spec2.unsqueeze(1)), dim=1)
        out = self.model(x1, x2)    
        model_time = time.time() - start_time
        # print("nans: ", x1.isnan().sum(), x2.isnan().sum())
        loss = out['loss']
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            if (batch_idx + 1) % self.accumulation_step == 0:
                # Add gradient clipping before optimizer step
                if self.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    # self.check_gradients()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                if self.scheduler is not None:
                    self.scheduler.step()
                self.scaler.update()
        else:
            loss.backward()
            if (batch_idx + 1) % self.accumulation_step == 0:
                # Add gradient clipping before optimizer step
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
        if torch.isnan(loss).sum() > 0:
            print("nan in loss, idx: ", batch_idx)
            exit()
        backward_time = time.time() - start_time - model_time
        self.optimizer.step()
        optimizer_time = time.time() - start_time - model_time - backward_time
        if self.scheduler is not None:
                    self.scheduler.step()
        # if self.wandb
        #     wandb.log({"train_loss": loss.item()})
        # print(f"model time: {model_time}, backward time: {backward_time}, optimizer time: {optimizer_time}")
        return loss, 0., x1

    def eval_batch(self,batch, batch_idx, device):
        lc, spec, lc2, spec2, info1, info2 = batch 
        lc, lc2, spec, spec2 = lc.to(device), lc2.to(device), spec.to(device), spec2.to(device)
        # print("before padd: ", lc.shape, spec.shape, lc2.shape, spec2.shape)
        spec = torch.nn.functional.pad(spec, (0, lc.shape[-1] - spec.shape[-1], 0,0))
        spec2 = torch.nn.functional.pad(spec2, (0, lc2.shape[-1] - spec2.shape[-1], 0,0))
        # print('after padd: ',  lc.shape, spec.shape, lc2.shape, spec2.shape)
        x1 = torch.cat((lc, spec.unsqueeze(1)), dim=1)
        x2 = torch.cat((lc2, spec2.unsqueeze(1)), dim=1)
        with torch.no_grad():
            out = self.model(x1, x2)
        loss = out['loss']
        # if self.wandb:
        #     wandb.log({"val_loss": loss.item()}) 
        return loss, 0, x1

    
class DoubleInputTrainer(Trainer):
    def __init__(self, num_classes=2, eta=0.5, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.eta = eta
    # def train_epoch(self, device, epoch=None ,plot=False, conf=False):
    #     """
    #     Trains the model for one epoch.
    #     """
    #     self.model.train()
    #     train_loss = []
    #     train_acc = 0
    #     all_accs = torch.zeros(self.num_classes, device=device)
    #     if self.train_sampler is not None:
    #         try:
    #             self.train_sampler.set_epoch(epoch)
    #         except AttributeError:
    #             pass
    #     pbar = tqdm(self.train_dl)
    #     for i, batch in enumerate(pbar):
    #         loss, acc,_ = self.train_batch(batch, device, conf)
    #         train_loss.append(loss.item())   
    #         pbar.set_description(f"train_acc: {acc}, train_loss:  {loss.item()}")
    #         all_accs = all_accs + acc
    #         if i > self.max_iter:
    #             break
    #         if self.range_update is not None and (i % self.range_update == 0):
    #             self.train_dl.dataset.expand_label_range()
    #             print("range: ", y.min(dim=0).values, y.max(dim=0).values)
    #     return train_loss, all_accs/len(self.train_dl.dataset)
    
    def train_batch(self, batch, batch_idx, device):
        x,_,y,_,info,_ = batch
        x1, x2 = x[:,-1,:], x[:,:-1,:]
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device)
        self.optimizer.zero_grad()
        y_pred = self.model(x1.float(), x2.float())
        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]
        # if conf:        
        #     y_pred, conf_pred = y_pred[:, :self.num_classes], y_pred[:, self.num_classes:]
        #     conf_y = torch.abs(y - y_pred)
        if self.num_classes > 1:
            loss_i = self.criterion(y_pred[:, 0], y[:, 0])  # Loss for inclination
            loss_p = self.criterion(y_pred[:, 1], y[:, 1])  # Loss for period
            loss = (self.eta * loss_i) + ((1-self.eta) * loss_p)
            # if conf:
            #     loss_conf_i = self.criterion(conf_pred[:, 0], conf_y[:, 0])
            #     loss_conf_p = self.criterion(conf_pred[:, 1], conf_y[:, 1])
            #     loss += (self.eta * loss_conf_i) + ((1-self.eta) * loss_conf_p)
        else:
            y_pred = y_pred.squeeze()
            # print("nans: ", y_pred.isnan().sum(), y.isnan().sum())
            loss = self.criterion(y_pred, y)
            # if conf:
            #     loss += self.criterion(conf_pred, conf_y)
        loss.backward()
        self.optimizer.step()
        diff = torch.abs(y_pred - y)
        acc = (diff < (y/10)).sum(0)
        return loss, acc, y_pred
    
    # def eval_epoch(self, batch_idx, device):
    #     """
    #     Evaluates the model for one epoch.
    #     """
    #     self.model.eval()
    #     val_loss = []
    #     val_acc = 0
    #     all_accs = torch.zeros(self.num_classes, device=device)
    #     if self.val_sampler is not None:
    #         try:
    #             self.train_sampler.set_epoch(epoch)
    #         except AttributeError:
    #             pass
    #     pbar = tqdm(self.val_dl)
    #     targets = np.zeros((0, self.num_classes))
    #     for i, batch in enumerate(pbar):
    #         loss, acc,_ = self.eval_batch(batch, device, conf)
    #         val_loss.append(loss.item())
    #         all_accs = all_accs + acc  
    #         pbar.set_description(f"val_acc: {acc}, val_loss:  {loss.item()}")
    #         if i > self.max_iter:
    #             break
    #         if self.range_update is not None and (i % self.range_update == 0):
    #             self.train_dl.dataset.expand_label_range()
    #     return val_loss, all_accs/len(self.val_dl.dataset)
    
    def eval_batch(self, batch, batch_idx, device):
        x,_,y,_,info,_ = batch
        x1, x2 = x[:,-1,:], x[:,:-1,:]
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device)
        with torch.no_grad():
            y_pred = self.model(x1.float(), x2.float())
            if isinstance(y_pred, tuple):
                y_pred = y_pred[0]
            # if conf:
            #     y_pred, conf_pred = y_pred[:, :self.num_classes], y_pred[:, self.num_classes:]
            #     conf_y = torch.abs(y - y_pred) 
            # if self.cos_inc:
            #     inc_idx = 0
            #     y_pred[:, inc_idx] = torch.cos(y_pred[:, inc_idx]*np.pi/2)
            #     y[:, inc_idx] = torch.cos(y[:, inc_idx]*np.pi/2)
            if self.num_classes > 1:
                loss_i = self.criterion(y_pred[:, 0], y[:, 0])  # Loss for inclination
                loss_p = self.criterion(y_pred[:, 1], y[:, 1])  # Loss for period
                loss = (self.eta * loss_i) + ((1-self.eta) * loss_p)
                # if conf:
                #     loss_conf_i = self.criterion(conf_pred[:, 0], conf_y[:, 0])
                #     loss_conf_p = self.criterion(conf_pred[:, 1], conf_y[:, 1])
                #     loss += (self.eta * loss_conf_i) + ((1-self.eta) * loss_conf_p)
            else:
                y_pred = y_pred.squeeze()
                loss = self.criterion(y_pred, y)
                # if conf:
                #     loss += self.criterion(conf_pred, conf_y)
            diff = torch.abs(y_pred - y)
            acc = (diff < (y/10)).sum(0)
        return loss, acc, y_pred


class DoubleInputTrainer2(Trainer):
    def __init__(self, num_classes=2, num_quantiles=5, eta=0.5, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.num_quantiles = num_quantiles
        self.eta = eta
    
    def train_batch(self, batch, batch_idx, device):
        lc,spectra,y,_,info,_ = batch
        x1, x2 = lc[:,-1,:], lc[:,:-1,:]
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device)
        self.optimizer.zero_grad()
        y_pred = self.model(x1.float(), x2.float())
        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]
        # if conf:        
        #     y_pred, conf_pred = y_pred[:, :self.num_classes], y_pred[:, self.num_classes:]
        #     conf_y = torch.abs(y - y_pred)
        if self.num_classes > 1:
            loss_i = self.criterion(y_pred[:, 0], y[:, 0])  # Loss for inclination
            loss_p = self.criterion(y_pred[:, 1], y[:, 1])  # Loss for period
            loss = (self.eta * loss_i) + ((1-self.eta) * loss_p)
            # if conf:
            #     loss_conf_i = self.criterion(conf_pred[:, 0], conf_y[:, 0])
            #     loss_conf_p = self.criterion(conf_pred[:, 1], conf_y[:, 1])
            #     loss += (self.eta * loss_conf_i) + ((1-self.eta) * loss_conf_p)
        else:
            y_pred = y_pred.squeeze()
            if isinstance(y_pred, tuple):
                y_pred = y_pred[0]
            y_pred = y_pred.reshape(y_pred.shape[0], -1, self.num_quantiles)
            # print("nans: ", y_pred.isnan().sum(), y.isnan().sum())
            loss = self.criterion(y_pred, y)
            # if conf:
            #     loss += self.criterion(conf_pred, conf_y)
        loss.backward()
        self.optimizer.step()
        diff = torch.abs(y_pred - y)
        acc = (diff < (y/10)).sum(0)
        return loss, acc, y_pred
    
    
    def eval_batch(self, batch, batch_idx, device):
        x,_,y,_,info,_ = batch
        x1, x2 = x[:,-1,:], x[:,:-1,:]
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device)
        with torch.no_grad():
            y_pred = self.model(x1.float(), x2.float())
            if isinstance(y_pred, tuple):
                y_pred = y_pred[0]
            # if conf:
            #     y_pred, conf_pred = y_pred[:, :self.num_classes], y_pred[:, self.num_classes:]
            #     conf_y = torch.abs(y - y_pred) 
            if self.cos_inc:
                inc_idx = 0
                y_pred[:, inc_idx] = torch.cos(y_pred[:, inc_idx]*np.pi/2)
                y[:, inc_idx] = torch.cos(y[:, inc_idx]*np.pi/2)
            if self.num_classes > 1:
                loss_i = self.criterion(y_pred[:, 0], y[:, 0])  # Loss for inclination
                loss_p = self.criterion(y_pred[:, 1], y[:, 1])  # Loss for period
                loss = (self.eta * loss_i) + ((1-self.eta) * loss_p)
                # if conf:
                #     loss_conf_i = self.criterion(conf_pred[:, 0], conf_y[:, 0])
                #     loss_conf_p = self.criterion(conf_pred[:, 1], conf_y[:, 1])
                #     loss += (self.eta * loss_conf_i) + ((1-self.eta) * loss_conf_p)
            else:
                y_pred = y_pred.squeeze()
                loss = self.criterion(y_pred, y)
                # if conf:
                #     loss += self.criterion(conf_pred, conf_y)
            diff = torch.abs(y_pred - y)
            acc = (diff < (y/10)).sum(0)
        return loss, acc, y_pred

