from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function, InplaceFunction
from torch.nn.modules.utils import _pair


class FixQuantization(Function):
    '''
    Fixed-Point Quantization
    '''
    @staticmethod
    def forward(ctx, input, int_bits=3, dec_bits=4, scheme="weight"):
        ctx.inplace = False
        if ctx.inplace:
            ctx.mark_dirty(input)
            x = input
        else:
            x = input.clone()

        with torch.no_grad():
            #!!!!!!
            max_pos = 2 ** (int_bits + dec_bits) - 1 
            max_neg = - 2 ** (int_bits + dec_bits)
            if scheme == "weight" :
                noise = torch.rand_like(x)
                y = x.mul_(2**dec_bits).add_(noise).floor_().clamp_(max_neg, max_pos).div_(2**dec_bits)
            elif scheme == "act" :
                y = x.mul_(2**dec_bits).floor_().clamp_(max_neg, max_pos).div_(2**dec_bits)
            else :
                raise NotImplementedError
        return y

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output
        return grad_input, None, None, None


def fix_quant(input, int_bits=3, dect_bits=4, scheme="weight"):
    output = FixQuantization().apply(input, int_bits, dect_bits, scheme)
    return output 


class DFX_Quantization(InplaceFunction):
    '''
    Forward Quantization
    Tensor-wise Dynamic Fixed-Point
    '''
    @staticmethod
    def forward(ctx, input, bits=16, scheme="weight"):
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
                if scheme=="weight":
                    noise = torch.rand_like(x)
                    x.div_(scale).add_(noise).floor_().clamp_(maxneg, maxpos).mul_(scale)
                elif scheme == "act"  :
                    x.div_(scale).round_().clamp_(maxneg, maxpos).mul_(scale)

        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output
        return grad_input, None, None

def dfx_quant(x, bits=16, scheme="weight"):
    qx = DFX_Quantization().apply(x, bits, scheme)
    return qx