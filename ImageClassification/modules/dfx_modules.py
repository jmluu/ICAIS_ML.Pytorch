from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dfx import quant, quant_grad, quant_fb

__all__ = ['QLinear', 'QConv2d', 'QBatchNorm2d']

class QLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, weight_copy=True):
        super(QLinear, self).__init__(in_features, out_features, bias=bias)


        # params define
        self.num_bits = 16
        self.add_noise = False 

        self.weight_params = (self.num_bits, self.num_bits,    self.add_noise)
        self.act_params =    (self.num_bits, self.num_bits,   self.add_noise)
        self.weight_copy = weight_copy

    def forward(self, input):


        if self.weight_copy:
            weight = quant_fb(self.weight, *(self.weight_params))
            if self.bias is not None:
                bias = self.bias 
            else:
                bias = None
            output = F.linear(input, weight, bias)
        else:
            self.weight.data = quant_fb(self.weight, *(self.weight_params))
            if self.bias is not None:
                self.bias.data = quant_fb(self.bias, *(self.bias_params))
            else:
                self.bias = None

            output = F.linear(input, self.weight, self.bias)

        output = quant_fb(output, *(self.act_params))

        return output

    def extra_repr(self):
        s = 'DFX_Linear, in_features={in_features}, out_features={out_features}, '
        s += 'dfx = {{ {num_bits}, add_noise = {add_noise} }}'
        if self.bias is not None:
            s += ', bias=True'
        return s.format(**self.__dict__)


class QConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 weight_copy=True):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=bias)
        # params define
        self.fw_bits = 8
        self.bw_bits = 8
        self.add_noise = False

        self.weight_params = (self.fw_bits, self.bw_bits,   self.add_noise)
        self.act_params =    (self.fw_bits, self.bw_bits,   self.add_noise)
        self.weight_copy = weight_copy

    def forward(self, input):

        if self.weight_copy:
            weight = quant_fb(self.weight, *(self.weight_params))
            if self.bias is not None:
                bias = self.bias
            else:
                bias = None
            output = F.conv2d(input, weight, bias, self.stride,
                              self.padding, self.dilation, self.groups)
        else:
            self.weight.data = quant_fb(self.weight, *(self.weight_params))
            if self.bias is not None:
                self.bias.data = quant_fb(self.bias, *(self.bias_params))
            else:
                self.bias = None

            output = F.conv2d(input, self.weight, self.bias, self.stride,
                              self.padding, self.dilation, self.groups)

        output = quant_fb(output, *(self.act_params))
        return output

    def extra_repr(self):
        s = ('DFX_CONV, {in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}, dfx = {{ {fw_bits}, {bw_bits}, add_noise={add_noise}}}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class QBatchNorm2d(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, weight_copy=True):
        super(QBatchNorm2d, self).__init__()
        # quantization parameters:

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        # params define
        self.num_bits = 8
        self.add_noise = False
        self.weight_params = (self.num_bits, self.num_bits,   self.add_noise)
        self.act_params =    (self.num_bits, self.num_bits,   self.add_noise)
        self.weight_copy = weight_copy


        if self.affine:
            self.bias = nn.Parameter(torch.Tensor(num_features))
            self.weight = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weights', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_buffer('num_batches_tracked', None)
        self.reset_parameters()


    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.weight_copy:
            # weight = quant_fb(self.weight, *(self.weight_params))
            weight = self.weight
            bias = self.bias 

            out = F.batch_norm(
                input, self.running_mean, self.running_var, weight, bias,
                self.training or not self.track_running_stats,
                exponential_average_factor, self.eps)
        else:

            self.weight.data = quant_fb(self.weight, *(self.weight_params))
            self.bias.data = quant_fb(self.bias, *(self.bias_params))

            out = F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training or not self.track_running_stats,
                exponential_average_factor, self.eps)

        out = quant_fb(out, *(self.act_params))

        return out

    def extra_repr(self):
        s = 'DFX_BN, {num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
            'track_running_stata={track_running_stats}, '
        s += ' dfx = {{ {num_bits}, add_noise={add_noise} }}'
        return s.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(QBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

