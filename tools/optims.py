"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import math


class LinearWarmupStepLRScheduler:
    def __init__(
        self,
        optimizer,
        max_epoch,
        min_lr,
        init_lr,
        decay_rate=1,
        warmup_start_lr=-1,
        warmup_steps=0,
        **kwargs
    ):
        self.optimizer = optimizer

        self.max_epoch = max_epoch
        self.min_lr = min_lr

        self.decay_rate = decay_rate

        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr
        
        # 添加用于跟踪当前状态的变量
        self.last_epoch = 0
        self.last_step = 0

    def step(self, cur_epoch, cur_step):
        # 更新当前状态
        self.last_epoch = cur_epoch
        self.last_step = cur_step
        
        if cur_epoch == 0:
            warmup_lr_schedule(
                step=cur_step,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                init_lr=self.warmup_start_lr,
                max_lr=self.init_lr,
            )
        else:
            step_lr_schedule(
                epoch=cur_epoch,
                optimizer=self.optimizer,
                init_lr=self.init_lr,
                min_lr=self.min_lr,
                decay_rate=self.decay_rate,
            )
            
    def state_dict(self):
        """返回当前scheduler的状态，用于保存检查点"""
        return {
            'last_epoch': self.last_epoch,
            'last_step': self.last_step,
            'max_epoch': self.max_epoch,
            'min_lr': self.min_lr,
            'init_lr': self.init_lr,
            'decay_rate': self.decay_rate,
            'warmup_steps': self.warmup_steps,
            'warmup_start_lr': self.warmup_start_lr,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
    
    def load_state_dict(self, state_dict):
        """从保存的状态恢复scheduler"""
        self.last_epoch = state_dict['last_epoch']
        self.last_step = state_dict['last_step']
        self.max_epoch = state_dict['max_epoch']
        self.min_lr = state_dict['min_lr']
        self.init_lr = state_dict['init_lr']
        self.decay_rate = state_dict['decay_rate']
        self.warmup_steps = state_dict['warmup_steps']
        self.warmup_start_lr = state_dict['warmup_start_lr']
        
        # 注意：通常optimizer的state_dict已经在别处加载，所以这里不再加载
        # if 'optimizer_state_dict' in state_dict:
        #     self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])


class LinearWarmupCosineLRScheduler:
    def __init__(
        self,
        optimizer,
        max_epoch,
        min_lr,
        init_lr,
        warmup_steps=0,
        warmup_start_lr=-1,
        **kwargs
    ):
        self.optimizer = optimizer

        self.max_epoch = max_epoch
        self.min_lr = min_lr

        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr
        
        # 添加用于跟踪当前状态的变量
        self.last_epoch = 0
        self.last_step = 0

    def step(self, cur_epoch, cur_step):
        # 更新当前状态
        self.last_epoch = cur_epoch
        self.last_step = cur_step
        
        # assuming the warmup iters less than one epoch
        # if cur_epoch == 0:
        if cur_epoch == 1:
            warmup_lr_schedule(
                step=cur_step,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                init_lr=self.warmup_start_lr,
                max_lr=self.init_lr,
            )
        else:
            cosine_lr_schedule(
                epoch=cur_epoch,
                optimizer=self.optimizer,
                max_epoch=self.max_epoch,
                init_lr=self.init_lr,
                min_lr=self.min_lr,
            )
    
    def state_dict(self):
        """返回当前scheduler的状态，用于保存检查点"""
        return {
            'last_epoch': self.last_epoch,
            'last_step': self.last_step,
            'max_epoch': self.max_epoch,
            'min_lr': self.min_lr,
            'init_lr': self.init_lr,
            'warmup_steps': self.warmup_steps,
            'warmup_start_lr': self.warmup_start_lr,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
    
    def load_state_dict(self, state_dict):
        """从保存的状态恢复scheduler"""
        self.last_epoch = state_dict['last_epoch']
        self.last_step = state_dict['last_step']
        self.max_epoch = state_dict['max_epoch']
        self.min_lr = state_dict['min_lr']
        self.init_lr = state_dict['init_lr']
        self.warmup_steps = state_dict['warmup_steps']
        self.warmup_start_lr = state_dict['warmup_start_lr']
        
        # 注意：通常optimizer的state_dict已经在别处加载，所以这里不再加载
        # 如果需要在这里加载optimizer状态，取消下面的注释
        # if 'optimizer_state_dict' in state_dict:
        #     self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])


def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (
        1.0 + math.cos(math.pi * epoch / max_epoch)
    ) + min_lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
    """Warmup the learning rate"""
    lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max(max_step, 1))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def step_lr_schedule(optimizer, epoch, init_lr, min_lr, decay_rate):
    """Decay the learning rate"""
    lr = max(min_lr, init_lr * (decay_rate ** epoch))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
