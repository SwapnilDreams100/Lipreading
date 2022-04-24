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

        if self.modality == 'raw_audio':
            self.frontend_nout = 1
            self.backend_out = 512
            self.trunk = ResNet1D(BasicBlock1D, [2, 2, 2, 2], relu_type=relu_type)
        elif self.modality == 'video':
            if self.backbone_type == 'resnet':
                self.frontend_nout = 64
                self.backend_out = 512
#                 self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)
                self.trunk = ResNet_swap(BasicBlock_swap, [2, 2, 2, 2], relu_type=relu_type)
                
            elif self.backbone_type == 'shufflenet':
                assert width_mult in [0.5, 1.0, 1.5, 2.0], "Width multiplier not correct"
                shufflenet = ShuffleNetV2( input_size=96, width_mult=width_mult)
                self.trunk = nn.Sequential( shufflenet.features, shufflenet.conv_last, shufflenet.globalpool)
                self.frontend_nout = 24
                self.backend_out = 1024 if width_mult != 2.0 else 2048
                self.stage_out_channels = shufflenet.stage_out_channels[-1]
                
            elif self.backbone_type == 'vit':
                self.backend_out = 384
                self.trunk = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
            elif self.backbone_type == 'mixer':
                self.backend_out = 512
                self.frontend_nout = 64
                self.trunk = ConvMixer(dim=self.backend_out, depth=25, frontend_out = self.frontend_nout, kernel_size=9, patch_size=3)
            
            if self.backbone_type=='resnet' or self.backbone_type=='shufflenet' or self.backbone_type=='mixer' :
                frontend_relu = nn.PReLU(num_parameters=self.frontend_nout) if relu_type == 'prelu' else nn.ReLU()
                self.frontend3D = nn.Sequential(
                    nn.Conv3d(1, self.frontend_nout, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
                    nn.BatchNorm3d(self.frontend_nout),
                    frontend_relu,
                    nn.MaxPool3d( kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
                )
        else:
            raise NotImplementedError

        tcn_class = TCN if len(tcn_options['kernel_size']) == 1 else MultiscaleMultibranchTCN
        self.tcn = tcn_class( input_size=self.backend_out,
                              num_channels=[hidden_dim*len(tcn_options['kernel_size'])*tcn_options['width_mult']]*tcn_options['num_layers'],
                              num_classes=num_classes,
                              tcn_options=tcn_options,
                              dropout=tcn_options['dropout'],
                              relu_type=relu_type,
                              dwpw=tcn_options['dwpw'],
                            )
        # -- initialize
        self._initialize_weights_randomly()

    def forward(self, x, lengths):
        if self.modality == 'video':
            if self.backbone_type=='resnet' or self.backbone_type=='shufflenet' :
                x = torch.squeeze(x,1)
                x = x.permute(0, 4, 1, 2, 3)
                B, C, T, H, W = x.size()
                print(x.size())
                x = self.frontend3D(x)
                print(x.size())
                Tnew = x.shape[2]    # outpu should be B x C2 x Tnew x H x W
                x = threeD_to_2D_tensor( x )
                print(x.size())
                x = self.trunk(x)
                print(x.size())
                if self.backbone_type == 'shufflenet':
                    x = x.view(-1, self.stage_out_channels)
                x = x.view(B, Tnew, x.size(1))
                print(x.size())
                print('--------------------------------------------')
            elif self.backbone_type=='mixer':
                B, C, T, H, W = x.size()
#                 print(x.size())
                x = self.frontend3D(x)
                x_new = threeD_to_2D_tensor( x )
#                 print(x_new.size())
#                 new_tensor = torch.zeros(x_new.shape[0],384)
#                 x_new = einops.repeat(x_new, 'b c h w -> b (repeat c) h w', repeat=3)
#                 print(x_new.size())
#                 T = 10
#                 for i in range(0,x_new.shape[0]-T, T):
#                     new_tensor[i:i+T] = self.trunk(x_new[i:i+T])
#                     print('llll')
                x_new = self.trunk(x_new)
#                 print(x_new.size())
#                 print(new_tensor.size(), new_tensor.requires_grad, x_new.requires_grad)
                x = x_new.view(B, T, x_new.size(1))
#                 print(x.size())
            else:
                B, C, T, H, W = x.size()
#                 print(x.size())
                x_new = threeD_to_2D_tensor( x )
#                 print(x_new.size())
#                 new_tensor = torch.zeros(x_new.shape[0],384)
                x_new = einops.repeat(x_new, 'b c h w -> b (repeat c) h w', repeat=3)
#                 print(x_new.size())
#                 T = 10
#                 for i in range(0,x_new.shape[0]-T, T):
#                     new_tensor[i:i+T] = self.trunk(x_new[i:i+T])
#                     print('llll')
                x_new = self.trunk(x_new)
#                 print(x_new.size())
#                 print(new_tensor.size(), new_tensor.requires_grad, x_new.requires_grad)
                x = x_new.view(B, T, x_new.size(1))
#                 x = new_tensor.view(B, T, new_tensor.size(1))
                
        elif self.modality == 'raw_audio':
            B, C, T = x.size()
            x = self.trunk(x)
            x = x.transpose(1, 2)
            lengths = [_//640 for _ in lengths]

        return x if self.extract_feats else self.tcn(x, lengths, B)


    def _initialize_weights_randomly(self):

        use_sqrt = True

        if use_sqrt:
            def f(n):
                return math.sqrt( 2.0/float(n) )
        else:
            def f(n):
                return 2.0/float(n)

        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                n = np.prod( m.kernel_size ) * m.out_channels
                m.weight.data.normal_(0, f(n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                n = float(m.weight.data[0].nelement())
                m.weight.data = m.weight.data.normal_(0, f(n))
