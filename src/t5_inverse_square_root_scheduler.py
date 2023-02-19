import argparse
from pathlib import Path
import math

import torch

class NoSheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer):
        super().__init__(optimizer, self.lr_lambda)
    
    def lr_lambda(self, epoch):
        return 1.0
    

class T5InverseSquareRootScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, num_warmup_steps):
        self.num_warmup_steps = num_warmup_steps
        self.start_lr = optimizer.defaults["lr"]
        # 任意のlrをスタートにできるように修正変数αを追加
        self.alpha = self.start_lr / (1 / math.sqrt(self.num_warmup_steps))

        super().__init__(optimizer, self.update_lr)
        

    def update_lr(self, step):
        return (self.alpha / math.sqrt(max(step, self.num_warmup_steps))) / self.start_lr
    
