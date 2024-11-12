class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, base_lr, final_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.final_lr = final_lr
        self.current_step = 0

    def get_lr(self):
        if self.current_step < self.warmup_steps:
            lr = self.base_lr + (self.final_lr - self.base_lr) * (self.current_step / self.warmup_steps)
        else:
            lr = self.final_lr
        return lr

    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

# Example usage:
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# scheduler = WarmupScheduler(optimizer, warmup_steps=1000, total_steps=10000, base_lr=0.001, final_lr=0.01)
# for epoch in range(num_epochs):
#     for batch in data_loader:
#         # Training code here
#         scheduler.step()