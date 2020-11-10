from __future__ import division
from math import cos, pi
import warnings
import torch

__all__ = ["LRSequential", "LRScheduler"]

class LRScheduler_base(object):
    """Base class of a learning rate scheduler.

    A scheduler returns a new learning rate based on the number of updates that have
    been performed.

    Parameters
    ----------
    base_lr : float, optional
        The initial learning rate.
    warmup_steps: int
        number of warmup steps used before this scheduler starts decay
    warmup_begin_lr: float
        if using warmup, the learning rate from which it starts warming up
    warmup_mode: string
        warmup can be done in two modes.
        'linear' mode gradually increases lr with each step in equal increments
        'constant' mode keeps lr at warmup_begin_lr for warmup_steps
    """
    def __init__(self, base_lr=0.01,
                 warmup_steps=0, warmup_begin_lr=0, warmup_mode='linear'):
        self.base_lr = base_lr
        assert isinstance(warmup_steps, int)
        self.warmup_steps = warmup_steps

        self.warmup_final_lr = base_lr
        self.warmup_begin_lr = warmup_begin_lr
        if self.warmup_begin_lr > self.warmup_final_lr:
            raise ValueError("Base lr has to be higher than warmup_begin_lr")
        if self.warmup_steps < 0:
            raise ValueError("Warmup steps has to be positive or 0")
        if warmup_mode not in ['linear', 'constant']:
            raise ValueError("Supports only linear and constant modes of warmup")
        self.warmup_mode = warmup_mode

    def get_warmup_lr(self, num_update):
        assert num_update < self.warmup_steps
        if self.warmup_mode == 'linear':
            increase = (self.warmup_final_lr - self.warmup_begin_lr) \
                       * float(num_update) / float(self.warmup_steps)
            return self.warmup_begin_lr + increase
        elif self.warmup_mode == 'constant':
            return self.warmup_begin_lr
        else:
            raise ValueError("Invalid warmup mode %s"%self.warmup_mode)

    def step(self, num_update):
        """Return a new learning rate.

        The ``num_update`` is the upper bound of the number of updates applied to
        every weight.

        Assume the optimizer has updated *i*-th weight by *k_i* times, namely
        ``optimizer.update(i, weight_i)`` is called by *k_i* times. Then::

            num_update = max([k_i for all i])

        Parameters
        ----------
        num_update: int
            the maximal number of updates applied to a weight.
        """
        raise NotImplementedError("must override this")

class LRSequential(LRScheduler_base):
    r"""Compose Learning Rate Schedulers

    Parameters
    ----------

    schedulers: list
        list of LRScheduler objects
    """
    def __init__(self, schedulers, offset=0):
        super(LRSequential, self).__init__()
        assert(len(schedulers) > 0)

        self.update_sep = []
        self.count = 0
        self.num_update = offset
        self.learning_rate = 0
        self.schedulers = []
        for lr in schedulers:
            self.add(lr)

    def add(self, scheduler):
        assert(isinstance(scheduler, LRScheduler))

        scheduler.offset = self.count
        self.count += scheduler.niters
        self.update_sep.append(self.count)
        self.schedulers.append(scheduler)

    def step(self):
        self.update(self.num_update)
        self.num_update += 1
        return self.learning_rate

    def update(self, num_update):
        num_update = min(num_update, self.count - 1)
        ind = len(self.schedulers) - 1
        for i, sep in enumerate(self.update_sep):
            if sep > num_update:
                ind = i
                break
        lr = self.schedulers[ind]
        lr.update(num_update)
        self.learning_rate = lr.learning_rate

class LRScheduler(LRScheduler_base):
    r"""Learning Rate Scheduler

    Parameters
    ----------

    mode : str
        Modes for learning rate scheduler.
        Currently it supports 'constant', 'step', 'linear', 'poly' and 'cosine'.
    base_lr : float
        Base learning rate, i.e. the starting learning rate.
    target_lr : float
        Target learning rate, i.e. the ending learning rate.
        With constant mode target_lr is ignored.
    niters : int
        Number of iterations to be scheduled.
    nepochs : int
        Number of epochs to be scheduled.
    iters_per_epoch : int
        Number of iterations in each epoch.
    offset : int
        Number of iterations before this scheduler.
    power : float
        Power parameter of poly scheduler.
    step_iter : list
        A list of iterations to decay the learning rate.
    step_epoch : list
        A list of epochs to decay the learning rate.
    step_factor : float
        Learning rate decay factor.
    """
    def __init__(self, mode, base_lr=0.1, target_lr=0,
                 niters=0, nepochs=0, iters_per_epoch=0, offset=0,
                 power=2, step_iter=None, step_epoch=None, step_factor=0.1,
                 baselr=None, targetlr=None):
        super(LRScheduler, self).__init__()
        assert(mode in ['constant', 'step', 'linear', 'poly', 'cosine'])

        self.mode = mode
        if mode == 'step':
            assert(step_iter is not None or step_epoch is not None)
        if baselr is not None:
            warnings.warn("baselr is deprecated. Please use base_lr.")
            if base_lr == 0.1:
                base_lr = baselr
        self.base_lr = base_lr
        if targetlr is not None:
            warnings.warn("targetlr is deprecated. Please use target_lr.")
            if target_lr == 0:
                target_lr = targetlr
        self.target_lr = target_lr
        if self.mode == 'constant':
            self.target_lr = self.base_lr

        self.niters = niters
        self.step = step_iter
        epoch_iters = nepochs * iters_per_epoch
        if epoch_iters > 0:
            self.niters = epoch_iters
            if step_epoch is not None:
                self.step = [s*iters_per_epoch for s in step_epoch]

        self.offset = offset
        self.power = power
        self.step_factor = step_factor

    def step(self, num_update):
        self.update(num_update)
        return self.learning_rate

    def update(self, num_update):
        N = self.niters - 1
        T = num_update - self.offset
        T = min(max(0, T), N)

        if self.mode == 'constant':
            factor = 0
        elif self.mode == 'linear':
            factor = 1 - T / N
        elif self.mode == 'poly':
            factor = pow(1 - T / N, self.power)
        elif self.mode == 'cosine':
            factor = (1 + cos(pi * T / N)) / 2
        elif self.mode == 'step':
            if self.step is not None:
                count = sum([1 for s in self.step if s <= T])
                factor = pow(self.step_factor, count)
            else:
                factor = 1
        else:
            raise NotImplementedError

        if self.mode == 'step':
            self.learning_rate = self.base_lr * factor
        else:
            self.learning_rate = self.target_lr + (self.base_lr - self.target_lr) * factor

