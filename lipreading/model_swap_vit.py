import torch
import torch.nn as nn
import math
import numpy as np
import timm
# from lipreading.models.resnet import ResNet, BasicBlock
from lipreading.models.resnet_swap import ResNet_swap, BasicBlock_swap
from lipreading.models.conv_mixer import *
import einops
from lipreading.models.resnet1D import ResNet1D, BasicBlock1D
from lipreading.models.shufflenetv2 import ShuffleNetV2
from lipreading.models.tcn import MultibranchTemporalConvNet, TemporalConvNet

# -- auxiliary functions
def threeD_to_2D_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch*s_time, n_channels, sx, sy)


def _average_batch(x, lengths, B):
    return torch.stack( [torch.mean( x[index][:,0:i], 1 ) for index, i in enumerate(lengths)],0 )


class MultiscaleMultibranchTCN(nn.Module):
    def __init__(self, input_size, num_channels, num_classes, tcn_options, dropout, relu_type, dwpw=False):
        super(MultiscaleMultibranchTCN, self).__init__()

        self.kernel_sizes = tcn_options['kernel_size']
        self.num_kernels = len( self.kernel_sizes )

        self.mb_ms_tcn = MultibranchTemporalConvNet(input_size, num_channels, tcn_options, dropout=dropout, relu_type=relu_type, dwpw=dwpw)
        self.tcn_output = nn.Linear(num_channels[-1], num_classes)

        self.consensus_func = _average_batch

    def forward(self, x, lengths, B):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        xtrans = x.transpose(1, 2)
        out = self.mb_ms_tcn(xtrans)
        out = self.consensus_func( out, lengths, B )
        return self.tcn_output(out)


class TCN(nn.Module):
    """Implements Temporal Convolutional Network (TCN)
    __https://arxiv.org/pdf/1803.01271.pdf
    """

    def __init__(self, input_size, num_channels, num_classes, tcn_options, dropout, relu_type, dwpw=False):
        super(TCN, self).__init__()
        self.tcn_trunk = TemporalConvNet(input_size, num_channels, dropout=dropout, tcn_options=tcn_options, relu_type=relu_type, dwpw=dwpw)
        self.tcn_output = nn.Linear(num_channels[-1], num_classes)

        self.consensus_func = _average_batch

        self.has_aux_losses = False

    def forward(self, x, lengths, B):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        x = self.tcn_trunk(x.transpose(1, 2))
        x = self.consensus_func( x, lengths, B )
        return self.tcn_output(x)


class Lipreading(nn.Module):
    def __init__( self, modality='video', hidden_dim=256, backbone_type='resnet', num_classes=500,
                  relu_type='prelu', tcn_options={}, width_mult=1.0, extract_feats=False):
        super(Lipreading, self).__init__()
        self.extract_feats = extract_feats
        self.backbone_type = backbone_type
        self.modality = modality
        
        self.backend_out = 384
        self.trunk = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
#         tcn_class = TCN if len(tcn_options['kernel_size']) == 1 else MultiscaleMultibranchTCN
#         self.tcn = tcn_class( input_size=self.backend_out,
#                               num_channels=[hidden_dim*len(tcn_options['kernel_size'])*tcn_options['width_mult']]*tcn_options['num_layers'],
#                               num_classes=num_classes,
#                               tcn_options=tcn_options,
#                               dropout=tcn_options['dropout'],
#                               relu_type=relu_type,
#                               dwpw=tcn_options['dwpw'],
#                             )
    def forward(self, x, lengths):
        if self.modality == 'video':
            x = torch.squeeze(x,1)
            x = x.permute(0, 4, 1, 2, 3)

            B, C, T, H, W = x.size()
#                 print(x.size())
            x_new = threeD_to_2D_tensor( x )
#                 print(x_new.size())
#                 new_tensor = torch.zeros(x_new.shape[0],384)
#             x_new = einops.repeat(x_new, 'b c h w -> b (repeat c) h w', repeat=3)
#                 print(x_new.size())
#                 T = 10
#                 for i in range(0,x_new.shape[0]-T, T):
#                     new_tensor[i:i+T] = self.trunk(x_new[i:i+T])
#                     print('llll')
            x_new = self.trunk(x_new)
            print(x_new.size())
#                 print(new_tensor.size(), new_tensor.requires_grad, x_new.requires_grad)
#             x_new = x_new.view(B, T, x_new.size(1))
#                 x = new_tensor.view(B, T, new_tensor.size(1))

#         return x if self.extract_feats else self.tcn(x, lengths, B)
        return x_new
