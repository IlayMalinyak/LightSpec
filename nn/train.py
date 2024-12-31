import torch
from torch.cuda.amp import autocast
import numpy as np
import time
import os
import yaml
from matplotlib import pyplot as plt
import glob
from collections import OrderedDict
from tqdm import tqdm
import torch.distributed as dist
import umap
import wandb


def count_occurence(x,y):
  coord_counts = {}
  for i in range(len(x)):
      coord = (x[i], y[i])
      if coord in coord_counts:
          coord_counts[coord] += 1
      else:
          coord_counts[coord] = 1


class Trainer(object):
    """
    A class that encapsulates the training loop for a PyTorch model.
    """
    def __init__(self, model, optimizer, criterion, train_dataloader, device, world_size=1, output_dim=2,
                 scheduler=None, val_dataloader=None,   max_iter=np.inf, scaler=None,
                  grad_clip=False, exp_num=None, log_path=None, exp_name=None, plot_every=None,
                   cos_inc=False, range_update=None, accumulation_step=1, wandb_log=False,
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
        self.logger = None
        self.range_update = range_update
        self.accumulation_step = accumulation_step
        self.wandb = wandb_log
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

    def fit(self, num_epochs, device,  early_stopping=None, only_p=False, best='loss', conf=False):
        """
        Fits the model for the given number of epochs.
        """
        min_loss = np.inf
        best_acc = 0
        train_loss, val_loss,  = [], []
        train_acc, val_acc = [], []
        lrs = []
        # self.optim_params['lr_history'] = []
        epochs_without_improvement = 0
        main_proccess = (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0) or self.device == 'cpu'

        print(f"Starting training for {num_epochs} epochs")
        print("is main process: ", main_proccess, flush=True)
        global_time = time.time()
        self.epoch = 0
        for epoch in range(num_epochs):
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
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                current_lr = self.optimizer.param_groups[0]['lr'] if self.scheduler is None \
                            else self.scheduler.get_last_lr()[0]
                
                lrs.append(current_lr)

                print(f'Epoch {epoch}, lr {current_lr}, Train Loss: {global_train_loss:.6f}, Val Loss:'\
                f'{global_val_loss:.6f}, Train Acc: {global_train_accuracy.round(decimals=4).tolist()}, '\
                f'Val Acc: {global_val_accuracy.round(decimals=4).tolist()},'\
                  f'Time: {time.time() - start_time:.2f}s, Total Time: {(time.time() - global_time)/3600} hr', flush=True)
                if epoch % 10 == 0:
                    print(os.system('nvidia-smi'))

                if epochs_without_improvement == early_stopping:
                    print('early stopping!', flush=True)
                    break
                if time.time() - global_time > (23.83 * 3600):
                    print("time limit reached")
                    break 

        return {"num_epochs":num_epochs, "train_loss": train_loss,
                 "val_loss": val_loss, "train_acc": train_acc, "val_acc": val_acc, "lrs": lrs}

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
                if grad_norm > 10:
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
            loss, acc , y = self.train_batch(batch, i, device)
            train_loss.append(loss.item())
            all_accs = all_accs + acc
            total += len(y)
            pbar.set_description(f"train_acc: {acc}, train_loss:  {loss.item()}")      
            if i > self.max_iter:
                break
        print("number of train_accs: ", train_acc)
        return train_loss, all_accs/total
    
    def train_batch(self, batch, batch_idx, device):
        lc,spec, y,_ = batch
        b, _, _ = lc.shape
        spec = spec.to(device)
        lc = lc.to(device)
        y = y.to(device)
        y_pred = self.model(spec.float(), lc.float())
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
        if self.wandb:
            wandb.log({"train_loss": loss.item(), "train_acc": acc})
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
            loss, acc, y = self.eval_batch(batch, i, device)
            val_loss.append(loss.item())
            all_accs = all_accs + acc
            total += len(y)
            pbar.set_description(f"val_acc: {acc}, val_loss:  {loss.item()}")
            if i > self.max_iter:
                break
        return val_loss, all_accs/total

    def eval_batch(self, batch, batch_idx, device):
        lc,spec,y,_ = batch
        spec = spec.to(device)
        lc = lc.to(device)
        b, _, _ = lc.shape
        y = y.to(device)
        with torch.no_grad():
            y_pred= self.model(spec.float(), lc.float())
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
        if self.wandb:
            wandb.log({"val_loss": loss.item(), "val_acc": acc})
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
        for i,(spec, lc, y,info) in enumerate(test_dataloader):
            spec = spec.to(device)
            lc = lc.to(device)
            y = y.to(device)
            with torch.no_grad():
                y_pred = self.model(spec.float(), lc.float())
            preds = np.concatenate((preds, y_pred.cpu().numpy()))
            if y.shape[1] == self.output_dim:
                targets = np.concatenate((targets, y.cpu().numpy()))
        print("target len: ", len(targets), "dataset: ", len(test_dataloader.dataset))
        return preds, targets

class MaskedSSLTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def train_batch(self,batch, batch_idx, device):
        """
        Trains the model for one batch.
        """
        # with autocast():
        x, y, mask,_,info,_ = batch
        y, mask, x = y.to(device), mask.to(device), x.to(device)
        out = self.model(x, y)
        if x.isnan().sum() > 0:
            print("nan in x, idx: ", batch_idx)
        if y.isnan().sum() > 0:
            print("nan in y, idx: ", batch_idx)
        if out.isnan().sum() > 0:
            print("nan in out, idx: ", batch_idx)
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
        acc = self.mask_accuracy(out, y, mask)
        if self.wandb:
            wandb.log({"train_loss": loss.item(), "train_acc": acc})
        return loss, acc, y
    
    def eval_batch(self, batch, batch_idx, device):
        """
        Evaluates the model for one batch.
        """
        x, y, mask,_,info,_ = batch
        y, mask, x = y.to(device), mask.to(device), x.to(device)
        with torch.no_grad():
            out = self.model(x, y)
        loss = self.criterion(out, y)
        acc = self.mask_accuracy(out, y, mask)
        if self.wandb:
            wandb.log({"val_loss": loss.item(), "val_acc": acc})
        return loss, acc, y
    
    def mask_accuracy(self, result, target, inverse_token_mask, epsilon=1e-5):
        # print(inverse_token_mask.shape, result.shape, target.shape)
        r = result.masked_select(inverse_token_mask)
        t = target.masked_select(inverse_token_mask)
        s = (torch.abs(r - t) < epsilon).sum()
        return s / inverse_token_mask.sum()


class ContrastiveTrainer(Trainer):
    def __init__(self, temperature=1, stack_pairs=False, use_w=False,
                   **kwargs):
        super().__init__(**kwargs)
        self.stack_pairs = stack_pairs
        self.temperature = temperature
        self.use_w = use_w
        
        
    def train_batch(self,batch, batch_idx, device):
        start_time = time.time()
        x1, x2, w, _, info1, info2 = batch 
        x1, x2 = x1.to(device), x2.to(device)
        if self.stack_pairs:
            x = torch.cat((x1, x2), dim=0)
            out = self.model(x, temperature=self.temperature)
            model_time = time.time() - start_time
        else:
            if self.use_w:
                out = self.model(x1, x2, w)
            else:
                out = self.model(x1, x2)
            model_time = time.time() - start_time
        loss = out['loss']
        loss.backward()
        backward_time = time.time() - start_time - model_time
        self.optimizer.step()
        optimizer_time = time.time() - start_time - model_time - backward_time
        if self.scheduler is not None:
                    self.scheduler.step()
        if self.wandb:
            wandb.log({"train_loss": loss.item()})
        # print(f"model time: {model_time}, backward time: {backward_time}, optimizer time: {optimizer_time}")
        return loss, 0., x1

    def eval_batch(self,batch, batch_idx, device):
        x1, x2, w, _, info1, info2 = batch 
        x1, x2 = x1.to(device), x2.to(device)
        with torch.no_grad():
            if self.stack_pairs:
                x = torch.cat((x1, x2), dim=0)
                out = self.model(x, temperature=self.temperature)
            else:
                if self.use_w:
                    out = self.model(x1, x2, w)
                else:
                    out = self.model(x1, x2)
        loss = out['loss']
        if self.wandb:
            wandb.log({"val_loss": loss.item()}) 
        return loss, 0, x1

class RegressorTrainer(Trainer):
    def __init__(self, w_name, w_init_val, **kwargs):
        super().__init__(**kwargs)
        self.w_name = w_name
        self.w_init_val = w_init_val
    
    def train_batch(self, batch, batch_idx, device):
        const = self.w_init_val/(1+self.epoch)
        x, _, y, _, info,_ = batch
        x, y = x.to(device), y.to(device)
        w = torch.tensor([i[self.w_name] for i in info]).to(device)
        # w = 1 + (const * torch.log(w))
        # w = torch.exp(w*const)
        out = self.model(x)
        loss = self.criterion(out, y)
        # print('shapes: ', x.shape, y.shape, 'w: ', w.shape)
        loss = (loss * w.unsqueeze(-1)).mean()
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
                    self.scheduler.step()
        if self.wandb:
            wandb.log({"train_loss": loss.item()})
        acc = (torch.abs(out - y) < y * 0.1).sum(0)
        return loss, acc, x

    def eval_batch(self, batch, batch_idx, device):
        const = self.w_init_val/(1+self.epoch)
        x, _, y, _, info,_ = batch
        x, y = x.to(device), y.to(device)
        w = torch.tensor([i[self.w_name] for i in info]).to(device)
        # w = 1 + (const * torch.log(w))
        w = torch.exp(w*const)
        with torch.no_grad():
            out = self.model(x)
        loss = self.criterion(out, y)
        loss = (loss * w.unsqueeze(-1)).mean()
        if self.wandb:
            wandb.log({"val_loss": loss.item()}) 
        acc = (torch.abs(out - y) < y * 0.1).sum(0)
        return loss, acc, x

class MaskedRegressorTrainer(Trainer):
    def __init__(self, w_name, w_init_val, ssl_criterion, ssl_weight=0.5, **kwargs):
        super().__init__(**kwargs)
        self.w_name = w_name
        self.ssl_criterion = ssl_criterion
        self.w_init_val = w_init_val
        self.ssl_weight = ssl_weight  # Weight to balance between SSL and regression
        
    def train_batch(self, batch, batch_idx, device):
        const = self.w_init_val/(1+self.epoch)
        x_masked, x, y, mask, info, _ = batch
        x_masked, x, y, mask = x_masked.to(device), x.to(device), y.to(device), mask.to(device)
        
        # Get proximity weights for regression
        w = torch.tensor([i[self.w_name] for i in info]).to(device)
        w = const * torch.exp(w*const)
        
        # Forward pass for both tasks
        reg_out,ssl_out  = self.model(x_masked, x)  # Masked filling task
        
        # Calculate SSL loss (masked filling)
        ssl_loss = self.ssl_criterion(ssl_out, x)
        ssl_acc = self.mask_accuracy(ssl_out, x, mask)

        
        # Calculate regression loss
        reg_loss = self.criterion(reg_out, y)
        reg_loss = (reg_loss * w.unsqueeze(-1)).mean()
        reg_acc = (torch.abs(reg_out - y) < y * 0.1).sum(0)
        
        # Combine losses
        total_loss = (self.ssl_weight * ssl_loss) + ((1 - self.ssl_weight) * reg_loss)
        
        # Backward and optimize
        total_loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
            
        # Log if using wandb
        if self.wandb:
            wandb.log({
                "train_loss": total_loss.item(),
                "ssl_loss": ssl_loss.item(),
                "reg_loss": reg_loss.item(),
                "ssl_acc": ssl_acc,
                "reg_acc": reg_acc.mean().item()
            })
            
        return total_loss, reg_acc, x

    def eval_batch(self, batch, batch_idx, device):
        const = self.w_init_val/(1+self.epoch)
        x_masked, x, y, mask, info, _ = batch
        x_masked, x, y, mask = x_masked.to(device), x.to(device), y.to(device), mask.to(device)
        
        w = torch.tensor([i[self.w_name] for i in info]).to(device)
        w = const * torch.exp(w*const)
        
        with torch.no_grad():
            reg_out,ssl_out  = self.model(x_masked, x)  # Masked filling task
            
            ssl_loss = self.criterion(ssl_out, x)
            ssl_acc = self.mask_accuracy(ssl_out, x, mask)
            
            reg_loss = self.criterion(reg_out, y)
            reg_loss = (reg_loss * w.unsqueeze(-1)).mean()
            reg_acc = (torch.abs(reg_out - y) < y * 0.1).sum(0)
            
            total_loss = (self.ssl_weight * ssl_loss) + ((1 - self.ssl_weight) * reg_loss)
        
        if self.wandb:
            wandb.log({
                "val_loss": total_loss.item(),
                "val_ssl_loss": ssl_loss.item(),
                "val_reg_loss": reg_loss.item(),
                "val_ssl_acc": ssl_acc,
                "val_reg_acc": reg_acc.mean().item()
            })
            
        return total_loss, reg_acc, x
        
    def mask_accuracy(self, result, target, inverse_token_mask, epsilon=1e-5):
        r = result.masked_select(inverse_token_mask)
        t = target.masked_select(inverse_token_mask)
        s = (torch.abs(r - t) < epsilon).sum()
        return s / inverse_token_mask.sum()

class AdversarialTrainer(Trainer):
    def __init__(
        self, 
        optimizer_gen,
        optimizer_disc,
        disc_steps=1,  # Number of discriminator steps per generator step
        gen_steps=1,    # Number of generator steps per discriminator step
        use_w=True,
        **kwargs
    ):
        """
        Adversarial trainer mimicking the ContrastiveTrainer interface
        
        Args:
            model: Adversarial multimodal alignment model
            optimizer_gen: Optimizer for generator
            optimizer_disc: Optimizer for discriminator
            scheduler_gen: Learning rate scheduler for generator (optional)
            scheduler_disc: Learning rate scheduler for discriminator (optional)
            wandb: Whether to log metrics to wandb
            stack_pairs: Whether to stack input pairs
            use_w: Whether to use proximity weights
            temperature: Temperature parameter (placeholder, not used in adversarial training)
            disc_steps: Number of discriminator optimization steps
            gen_steps: Number of generator optimization steps
        """
        super().__init__(**kwargs)
        
        self.optimizer_gen = optimizer_gen
        self.optimizer_disc = optimizer_disc
        self.disc_steps = disc_steps
        self.gen_steps = gen_steps
        self.use_w = use_w

    def train_batch(self, batch, batch_idx, device):
        """
        Training method with signature matching ContrastiveTrainer
        
        Args:
            batch: Tuple containing (x1, x2, w, _, info1, info2)
            batch_idx: Batch index
            device: Training device
        
        Returns:
            loss: Total loss
            aux_loss: Auxiliary loss (set to 0)
            x1: First input batch (for potential feature extraction)
        """
        start_time = time.time()
        
        # Unpack batch
        x1, x2, w, _, info1, info2 = batch
        x1, x2 = x1.to(device), x2.to(device)
        
        if w is None:
            w = torch.ones(x1.size(0)) * 0.5
        w = w.to(device)
                
        # Perform multiple discriminator and generator steps
        disc_loss = 0
        gen_loss = 0
            
        # Discriminator steps
        for _ in range(self.disc_steps):
            self.optimizer_disc.zero_grad()
            
            if self.use_w:
                disc_loss = self.model.module.discriminator_loss(x1, x2, w)
            else:
                disc_loss = self.model.module.discriminator_loss(x1, x2, None)
            
            disc_loss.backward()
            self.optimizer_disc.step()
    
        
        # Generator steps
        for _ in range(self.gen_steps):
            self.optimizer_gen.zero_grad()
            
            if self.use_w:
                gen_loss = self.model.module.generator_loss(x1, x2, w)
            else:
                gen_loss = self.model.module.generator_loss(x1, x2, None)
            
            gen_loss.backward()
            self.optimizer_gen.step()
            
        
        # Combine losses (total loss for logging)
        loss = disc_loss + gen_loss
        out = {'loss': loss, 'disc_loss': disc_loss, 'gen_loss': gen_loss}
        
        model_time = time.time() - start_time
        
        # Optional wandb logging
        if self.wandb:
            log_dict = {"train_loss": out['loss'].item()}
            if 'disc_loss' in out:
                log_dict["train_disc_loss"] = out['disc_loss'].item()
            if 'gen_loss' in out:
                log_dict["train_gen_loss"] = out['gen_loss'].item()
            wandb.log(log_dict)
        
        return out['loss'], 0., x1

    def eval_batch(self, batch, batch_idx, device):
        """
        Evaluation method with signature matching ContrastiveTrainer
        Similar to train_batch but without gradient computation
        """
        x1, x2, w, _, info1, info2 = batch
        x1, x2 = x1.to(device), x2.to(device)
        
        if w is None:
            w = torch.ones(x1.size(0)) * 0.5
        w = w.to(device)
        
        with torch.no_grad():
            
            if self.use_w:
                disc_loss = self.model.discriminator_loss(x1, x2, w)
                gen_loss = self.model.generator_loss(x1, x2, w)
            else:
                disc_loss = self.model.discriminator_loss(x1, x2, None)
                gen_loss = self.model.generator_loss(x1, x2, None)
            
            loss = disc_loss + gen_loss
            out = {'loss': loss, 'disc_loss': disc_loss, 'gen_loss': gen_loss}
        
        # Optional wandb logging
        if self.wandb:
            log_dict = {"val_loss": out['loss'].item()}
            if 'disc_loss' in out:
                log_dict["val_disc_loss"] = out['disc_loss'].item()
            if 'gen_loss' in out:
                log_dict["val_gen_loss"] = out['gen_loss'].item()
            wandb.log(log_dict)
        
        return out['loss'], 0, x1
