import torch
import math
from torch._six import inf
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR

class LinearDecayLRWithWarmup(object):
    """
    adjust lr:

    args:
        warmup_lr: float or None, the learning rate to be touched after warmup
        warmup: int, the number of steps to warmup
    """

    def __init__(self, optimizer, T_max, last_epoch=-1, verbose=False,
                 min_lr=0, warmup_lr=None, warmup=0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.last_epoch = last_epoch
        self.verbose = verbose
        self.warmup_lr = warmup_lr
        self.warmup = warmup

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)
        self.max_lrs = [lr for lr in self.min_lrs]
        
        self._prepare_for_warmup()

    def step(self):
        epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if epoch <= self.warmup:
            self._increase_lr(epoch)
        else:
            self._reduce_lr(epoch)

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            progress = float(epoch - self.warmup) / float(max(1, self.T_max - self.warmup))
            factor = max(0.0, 1 - progress)
            old_lr = float(param_group['lr'])
            new_lr = max(self.max_lrs[i] * factor, self.min_lrs[i])
            param_group['lr'] = new_lr
            if self.verbose:
                print('Epoch {:5d}: reducing learning rate'
                        ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

    def _increase_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = old_lr + self.warmup_lr_steps[i]
            param_group['lr'] = new_lr
            self.max_lrs[i] = max(self.max_lrs[i], new_lr)
            if self.verbose:
                print('Epoch {:5d}: increasing learning rate'
                        ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

    def _prepare_for_warmup(self):
        if self.warmup_lr is not None:
            if isinstance(self.warmup_lr, (list, tuple)):
                if len(self.warmup_lr) != len(self.optimizer.param_groups):
                    raise ValueError("expected {} warmup_lrs, got {}".format(
                        len(self.optimizer.param_groups), len(self.warmup_lr)))
                self.warmup_lrs = list(self.warmup_lr)
            else:
                self.warmup_lrs = [self.warmup_lr] * len(self.optimizer.param_groups)
        else:
            self.warmup_lrs = None
        if self.warmup > self.last_epoch:
            curr_lrs = [group['lr'] for group in self.optimizer.param_groups]
            self.warmup_lr_steps = [max(0, (self.warmup_lrs[i] - curr_lrs[i])/float(self.warmup)) for i in range(len(curr_lrs))]
        else:
            self.warmup_lr_steps = None


    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._prepare_for_warmup()