from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from .quant_func import fix_quant as quant


### ==============================================================================### 
###             quant for different data types                                    ###
### ==============================================================================### 
act_quant = lambda x : quant(x, 3, 4, "act")
weight_quant = lambda x : quant(x, 2, 5, "weight")
bias_quant = lambda x : quant(x, 7, 8, "weight")

### ===============================================================================### 
###             Quantization Modules                                               ###
### ===============================================================================### 

class QReLu(nn.ReLU):
    __constants__ = ['inplace']

    def __init__(self, inplace=False):
        super(nn.ReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        out =  F.relu(input, inplace=self.inplace)
        out =  act_quant(out)

        return out

    def extra_repr(self):
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


class QLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(QLinear, self).__init__(in_features, out_features, bias=bias)

        self.ia_quant = lambda x : quant(x, 5, 2, "act")
        self.weight_quant = lambda x : quant(x, 3, 4, "weight")
        self.bias_quant = lambda x : quant(x, 5, 2, "weight")
        self.oa_quant = lambda x : quant(x, 5, 2, "act")

    def forward(self, input):
        input = self.ia_quant(input)

        weight = self.weight_quant(self.weight)

        if self.bias is not None :
            bias = self.bias_quant(self.bias)
        else : 
            bias = None

        output = F.linear(input, weight, None)
        
        output = self.oa_quant(output)  # post bias 

        if self.bias is not None :
            output = output + bias 
            output = self.oa_quant(output)

        # output = F.linear(input, self.weight, self.bias)
        return output

class QAveragePool2d(nn.AvgPool2d):
    __constants__ = ['kernel_size', 'stride', 'padding', 'ceil_mode', 'count_include_pad', 'divisor_override']

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True, divisor_override=None):
        super(QAveragePool2d, self).__init__(kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True, divisor_override=None)


    def forward(self, input):
        input = act_quant(input)

        out = F.avg_pool2d(input, self.kernel_size, self.stride,
                            self.padding, self.ceil_mode, self.count_include_pad, self.divisor_override)

        out =  act_quant(out)
        return out

class QConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 ):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, input):
        weight = weight_quant(self.weight)
        
        if self.bias is not None: 
            bias = bias_quant(self.bias)

        output = F.conv2d(input, weight, None, self.stride,
                              self.padding, self.dilation, self.groups)
        
        output = act_quant(output)

        if self.bias is not None : 
            output = output + bias.view(1, -1, 1, 1) 
            output = act_quant(output)

        return output

class _QConvBnNd(nn.modules.conv._ConvNd):

    _version = 2

    def __init__(self,
                 # ConvNd args
                 in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups,
                 bias,
                 padding_mode,
                 # BatchNormNd args
                 # num_features: out_channels
                 eps=1e-05, momentum=0.1,
                 # affine: True
                 # track_running_stats: True
                 # Args for this module
                 freeze_bn=True,
                 ):
        nn.modules.conv._ConvNd.__init__(self, in_channels, out_channels, kernel_size,
                                         stride, padding, dilation, transposed,
                                         output_padding, groups, False, padding_mode)


        self.freeze_bn = freeze_bn if self.training else True

        # if self.training : 
        norm_layer = nn.BatchNorm2d
        # else : 
            # norm_layer = IdentityBN
        self.bn = norm_layer(out_channels, eps, momentum, True, True)


        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_bn_parameters()

        # this needs to be called after reset_bn_parameters,
        # as they modify the same state
        if self.training:
            if freeze_bn:
                self.freeze_bn_stats()
            else:
                self.update_bn_stats()
        else:
            self.freeze_bn_stats()

    def reset_running_stats(self):
        self.bn.reset_running_stats()

    def reset_bn_parameters(self):
        self.bn.reset_running_stats()
        nn.init.uniform_(self.bn.weight)
        nn.init.zeros_(self.bn.bias)
        # note: below is actully for conv, not BN
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def get_params(self):
        running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
        scale_factor = self.bn.weight / running_std
        scaled_weight = self.weight_fake_quant(self.weight * scale_factor.reshape([-1, 1, 1, 1]))

        # scaled bias : 
        if self.bias is not None :
            scaled_bias =  scale_factor * (self.bias - self.bn.running_mean)  + self.bn.bias
        else :
            scaled_bias =  - scale_factor * self.bn.running_mean  + self.bn.bias
        scaled_bias_q = self.bias_fake_quant(scaled_bias)

        return scaled_weight, scaled_bias_q

    def reset_parameters(self):
        super(_QConvBnNd, self).reset_parameters()

    def update_bn_stats(self):
        self.freeze_bn = False
        self.bn.training = True
        return self

    def freeze_bn_stats(self):
        self.freeze_bn = True
        self.bn.training = False
        return self

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)

        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def _forward(self, input):
        input = act_quant(input)

        running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
        scale_factor = self.bn.weight / running_std
        scaled_weight = weight_quant(self.weight * scale_factor.reshape([-1, 1, 1, 1]))

        # scaled bias : 
        # with torch.no_grad():
        if self.bias is not None :
            scaled_bias =   scale_factor *(self.bias - self.bn.running_mean)  + self.bn.bias
        else : 
            scaled_bias =   - scale_factor * self.bn.running_mean  + self.bn.bias
        scaled_bias_q = bias_quant(scaled_bias)

        # this does not include the conv bias
        conv = self._conv_forward(input, scaled_weight)

        conv = act_quant(conv)
        
        conv_bias = conv + scaled_bias_q.reshape([1, -1, 1, 1])
        conv_bias = act_quant(conv_bias)

        if self.training : 
            conv_bias_orig = conv_bias - scaled_bias.reshape([1, -1, 1, 1])
            conv_orig = conv_bias_orig / scale_factor.reshape([1, -1, 1, 1])

            conv_orig = conv / scale_factor.reshape([1, -1, 1, 1])
            if self.bias is not None:
                conv_orig = conv_orig + self.bias.reshape([1, -1, 1, 1])
            conv = self.bn(conv_orig)
            return conv
        else : 
            return conv_bias

    def extra_repr(self):
        # TODO(jerryzh): extend
        return super(_QConvBnNd, self).extra_repr()

    def forward(self, input):
        return act_quant(self._forward(input))

    def train(self, mode=True):
        """
        Batchnorm's training behavior is using the self.training flag. Prevent
        changing it if BN is frozen. This makes sure that calling `model.train()`
        on a model with a frozen BN will behave properly.
        """
        self.training = mode
        if not self.freeze_bn:
            for module in self.children():
                module.train(mode)
        return self

    # ===== Serialization version history =====
    #
    # Version 1/None
    #   self
    #   |--- weight : Tensor
    #   |--- bias : Tensor
    #   |--- gamma : Tensor
    #   |--- beta : Tensor
    #   |--- running_mean : Tensor
    #   |--- running_var : Tensor
    #   |--- num_batches_tracked : Tensor
    #
    # Version 2
    #   self
    #   |--- weight : Tensor
    #   |--- bias : Tensor
    #   |--- bn : Module
    #        |--- weight : Tensor (moved from v1.self.gamma)
    #        |--- bias : Tensor (moved from v1.self.beta)
    #        |--- running_mean : Tensor (moved from v1.self.running_mean)
    #        |--- running_var : Tensor (moved from v1.self.running_var)
    #        |--- num_batches_tracked : Tensor (moved from v1.self.num_batches_tracked)
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        if version is None or version == 1:
            # BN related parameters and buffers were moved into the BN module for v2
            v2_to_v1_names = {
                'bn.weight': 'gamma',
                'bn.bias': 'beta',
                'bn.running_mean': 'running_mean',
                'bn.running_var': 'running_var',
                'bn.num_batches_tracked': 'num_batches_tracked',
            }
            for v2_name, v1_name in v2_to_v1_names.items():
                if prefix + v1_name in state_dict:
                    state_dict[prefix + v2_name] = state_dict[prefix + v1_name]
                    state_dict.pop(prefix + v1_name)
                elif strict:
                    missing_keys.append(prefix + v2_name)

        super(_QConvBnNd, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class QConvBn2d(_QConvBnNd, nn.Conv2d):
    r"""
    A ConvBn2d module is a module fused from Conv2d and BatchNorm2d,
    attached with FakeQuantize modules for both output activation and weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv2d` and
    :class:`torch.nn.BatchNorm2d`.

    Implementation details: https://arxiv.org/pdf/1806.08342.pdf section 3.2.2

    Similar to :class:`torch.nn.Conv2d`, with FakeQuantize modules initialized
    to default.

    Attributes:
        freeze_bn:
        activation_post_process: fake quant module for output activation
        weight_fake_quant: fake quant module for weight

    """

    def __init__(self,
                 # ConvNd args
                 in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=None,
                 padding_mode='zeros',
                 # BatchNorm2d args
                 # num_features: out_channels
                 eps=1e-05, momentum=0.1,
                 # affine: True
                 # track_running_stats: True
                 # Args for this module
                 freeze_bn=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        _QConvBnNd.__init__(self, in_channels, out_channels, kernel_size, stride,
                           padding, dilation, False, _pair(0), groups, bias, padding_mode,
                           eps, momentum, freeze_bn)

class QConvBnReLU2d(QConvBn2d):
    r"""
    A ConvBnReLU2d module is a module fused from Conv2d, BatchNorm2d and ReLU,
    attached with FakeQuantize modules for both output activation and weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv2d` and
    :class:`torch.nn.BatchNorm2d` and :class:`torch.nn.ReLU`.

    Implementation details: https://arxiv.org/pdf/1806.08342.pdf

    Similar to `torch.nn.Conv2d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        observer: fake quant module for output activation, it's called observer
            to align with post training flow
        weight_fake_quant: fake quant module for weight

    """

    def __init__(self,
                 # Conv2d args
                 in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=None,
                 padding_mode='zeros',
                 # BatchNorm2d args
                 # num_features: out_channels
                 eps=1e-05, momentum=0.1,
                 # affine: True
                 # track_running_stats: True
                 # Args for this module
                 freeze_bn=False):
        super(QConvBnReLU2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                           padding, dilation, groups, bias,
                                           padding_mode, eps, momentum,
                                           freeze_bn)
        self.relu = nn.ReLU()
    def forward(self, input):
        return act_quant(self.relu(QConvBn2d._forward(self, input)))
