from __future__ import division
from collections import namedtuple
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction, Function

"""
Tensor-wise Dynamic Fixed-Point
"""

class Quantization(InplaceFunction):
    '''
    Forward Quantization
    '''
    @staticmethod
    def forward(ctx, input, bits=16, add_noise=True):
        ctx.inplace = False
        if ctx.inplace:
            ctx.mark_dirty(input)
            x = input
        else:
            x = input.clone()

        with torch.no_grad():
            maxpos = pow(2, bits - 1) - 1
            maxneg = - pow(2, bits - 1)

            if x.abs().max() != 0:
                exp = torch.log2(x.abs().max()).ceil().sub(bits - 1)
                scale = torch.pow(2, exp)
                if add_noise:
                    noise = torch.rand_like(x)
                    x.div_(scale).add_(noise).floor_().clamp_(maxneg, maxpos).mul_(scale)
                else :
                    x.div_(scale).round_().clamp_(maxneg, maxpos).mul_(scale)

        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output
        return grad_input, None, None


class Quantization_Gradient(InplaceFunction):
    '''
    Backward Quantization
    '''
    @staticmethod
    def forward(ctx, input, bits=16, add_noise=True):
        ctx.bits = bits
        ctx.add_noise = add_noise
        return input

    @staticmethod
    def backward(ctx, grad_output):
        bits = ctx.bits
        add_nosie = ctx.add_noise
        with torch.no_grad():
            grad_input = quant(grad_output, bits, add_nosie)
        return grad_input, None, None

def quant(x, bits=16, add_noise=True):
    qx = Quantization().apply(x, bits, add_noise)
    return qx

def quant_grad(x, bits=16, add_noise=True):
    qx = Quantization_Gradient().apply(x, bits, add_noise)
    return qx

def quant_fb(x, bits_f = 16, bits_b=16, add_noise=False):
    x = quant(x, bits_f, add_noise)
    x = quant_grad(x, bits_b, add_noise)
    return x

