import functools

import torch
import torch.nn as nn

from .layer_utils import get_norm_layer, ResNetBlock, MinibatchDiscrimination
from base.base_model import BaseModel


class ResNetGenerator(BaseModel):
    """Define a generator using ResNet"""

    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9, norm_type='instance', padding_type='reflect',
                 use_dropout=True, learn_residual=True):
        super(ResNetGenerator, self).__init__()
                                                                              # ngf: number of filters in generator (channels)
        self.learn_residual = learn_residual

        norm_layer = get_norm_layer(norm_type)                                # get the proper norm layer
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d         # turn off the bias if BatchNorm2d is used
            
                                                                                # the first conv layer to expand the channels (1/3 -> ngf)
        sequence = [                                        # padding(x+6) -> conv2d (7, 1 -> x-6) -> instance norm (apply to each image(H*W)) -> ReLU
            nn.ReflectionPad2d(3),                          # the size of image not change (x -> x)
            nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0, bias=use_bias),
            norm_layer(ngf),                                # point out the channels in (N*C*H*W)
            nn.ReLU(True)                                   # use the ReLU as activation function
        ]

        n_downsampling = 2
        for i in range(n_downsampling):  # downsample the feature map           # front part of the U-net
            mult = 2 ** i
            sequence += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),           # double the channel and half the size (x -> x/2)
                norm_layer(ngf * mult * 2),                                                                         # norm layer
                nn.ReLU(True)                                                           # do the relu calculation in place to save some memory                                                                   
            ]

        for i in range(n_blocks):  # ResNet                                     # stack of multiple resnets, keep size and channel still
            sequence += [
                ResNetBlock(ngf * 2 ** n_downsampling, norm_layer, padding_type, use_dropout, use_bias)
            ]

        for i in range(n_downsampling):  # upsample the feature map             # back part of the U-net, upsampling, make it has same layer with as front part
            mult = 2 ** (n_downsampling - i)                                    # calculate the channels
            sequence += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,         # half the channel and double the size (x -> 2x)
                                   output_padding=1, bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]

        sequence += [                                                           # the final conv layer to map the channels to output, size donot change (x -> x)
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0),
            nn.Tanh()                                                           # use the tanh as activation function to control the output range (-1, 1)
        ]                                                                       

        self.model = nn.Sequential(*sequence)                                   # get the whole model

    def forward(self, x):
        out = self.model(x)                                                     # get the output
        if self.learn_residual:                                                 # put a high layer skip connection at higest level
            out = x + out                                                       # use output = x + x_range * out to adjust the proper range
                                                                                # or also use this with a proper normalization
            out = torch.clamp(out, min=-1, max=1)  # clamp to [-1,1] according to normalization(mean=0.5, var=0.5)      # clamp the output for a proper data 
        return out                                                              # mean=0.5, var=0.5 means -0.5 and then /0.5, to make data in range [-1, 1] 
    
class ResNetGeneratorReduced(BaseModel):
    """
    Define a generator using ResNet
    
    Less channels and layers: 10 -> 10 -> 5 -> 5x 5 -> 10 -> 10, with a skip connection
    """

    def __init__(self, input_nc, output_nc, ngf=16, n_blocks=5, norm_type='instance', padding_type='reflect',
                 use_dropout=True, learn_residual=True):
        super(ResNetGenerator, self).__init__()
                                                                              # ngf: number of filters in generator (channels)
        self.learn_residual = learn_residual

        norm_layer = get_norm_layer(norm_type)                                # get the proper norm layer
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d         # turn off the bias if BatchNorm2d is used
            
                                                                                # the first conv layer to expand the channels (1/3 -> ngf)
        sequence = [                                        # padding(x+6) -> conv2d (7, 1 -> x-6) -> instance norm (apply to each image(H*W)) -> ReLU
            nn.ReflectionPad2d(3),                          # the size of image not change (x -> x)
            nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0, bias=use_bias),
            norm_layer(ngf),                                # point out the channels in (N*C*H*W)
            nn.ReLU(True)                                   # use the ReLU as activation function
        ]

        n_downsampling = 1
        for i in range(n_downsampling):  # downsample the feature map           # front part of the U-net
            mult = 2 ** i
            sequence += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),           # double the channel and half the size (x -> x/2)
                norm_layer(ngf * mult * 2),                                                                         # norm layer
                nn.ReLU(True)                                                           # do the relu calculation in place to save some memory                                                                   
            ]

        for i in range(n_blocks):  # ResNet                                     # stack of multiple resnets, keep size and channel still
            sequence += [
                ResNetBlock(ngf * 2 ** n_downsampling, norm_layer, padding_type, use_dropout, use_bias)
            ]

        for i in range(n_downsampling):  # upsample the feature map             # back part of the U-net, upsampling, make it has same layer with as front part
            mult = 2 ** (n_downsampling - i)                                    # calculate the channels
            sequence += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,         # half the channel and double the size (x -> 2x)
                                   output_padding=1, bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]

        sequence += [                                                           # the final conv layer to map the channels to output, size donot change (x -> x)
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0),
            nn.Tanh()                                                           # use the tanh as activation function to control the output range (-1, 1)
        ]                                                                       

        self.model = nn.Sequential(*sequence)                                   # get the whole model

    def forward(self, x):
        out = self.model(x)                                                     # get the output
        if self.learn_residual:                                                 # put a high layer skip connection at higest level
            out = x + out                                                       # use output = x + x_range * out to adjust the proper range
                                                                                # or also use this with a proper normalization
            out = torch.clamp(out, min=-1, max=1)  # clamp to [-1,1] according to normalization(mean=0.5, var=0.5)      # clamp the output for a proper data 
        return out  
            


class NLayerDiscriminator(BaseModel):
    """Define a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_type='instance', use_sigmoid=False,               # try to get the representation for images
                 use_minibatch_discrimination=False):                                           # 360(*640) -> 180 -> 90/45/22 -> 21 -> 20
        super(NLayerDiscriminator, self).__init__()                                   # finally use a matrix (1-channel feature tensor) as the classification result

        self.use_minibatch_discrimination = use_minibatch_discrimination

        norm_layer = get_norm_layer(norm_type)                                  # get the normalization layer
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kernel_size = 4
        padding = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kernel_size, stride=2, padding=padding),   # first layer, increase the channel and half the size with stride 2    
            nn.LeakyReLU(0.2, True)                                                         # use LeakyReLU as activation 
        ]

        nf_mult = 1
        for n in range(1, n_layers):  # gradually increase the number of filters            # increase the channel and half the size (3 layers)
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kernel_size, stride=2, padding=padding,
                          bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [                                                                       # clamp the channels and try to keep a almost same size (x -> x-1)
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kernel_size, stride=1, padding=padding,
                      bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kernel_size, stride=1, padding=padding) # reduce the channel to 1 and maintain size (x -> x-1)
        ]  # output 1 channel prediction map

        if use_sigmoid:
            sequence += [nn.Sigmoid()]                                                      # clamp the result to [0, 1] inorder to use the GAN loss

        self.model = nn.Sequential(*sequence)                                               # get the whole model

    def forward(self, x):
        out = self.model(x)
        if self.use_minibatch_discrimination:
            out = out.view(out.size(0), -1)
            a = out.size(1)
            out = MinibatchDiscrimination(a, a, 3)(out)
        return out
    
    class NLayerDiscriminatorReduced(BaseModel):
    """
    Define a PatchGAN discriminator
    
    less layer and channel: 10 -> 2x 5 -> 5 -> 5
    
    """

    def __init__(self, input_nc, ndf=16, n_layers=2, norm_type='instance', use_sigmoid=False,               # try to get the representation for images
                 use_minibatch_discrimination=False):                                           # 360(*640) -> 180 -> 90/45/22 -> 21 -> 20
        super(NLayerDiscriminator, self).__init__()                                   # finally use a matrix (1-channel feature tensor) as the classification result

        self.use_minibatch_discrimination = use_minibatch_discrimination

        norm_layer = get_norm_layer(norm_type)                                  # get the normalization layer
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kernel_size = 4
        kernel_size_2 = 3
        padding = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kernel_size, stride=2, padding=padding),   # first layer, increase the channel and half the size with stride 2    
            nn.LeakyReLU(0.2, True)                                                         # use LeakyReLU as activation 
        ]

        nf_mult = 1
        for n in range(1, n_layers):  # gradually increase the number of filters            # increase the channel and leave the size still
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kernel_size_2, stride=1, padding=padding,
                          bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [                                                                       # clamp the channels and try to keep a almost same size (x -> x-1)
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kernel_size_2, stride=1, padding=padding,
                      bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kernel_size_2, stride=1, padding=padding) # reduce the channel to 1 and maintain size (x -> x-1)
        ]  # output 1 channel prediction map

        if use_sigmoid:
            sequence += [nn.Sigmoid()]                                                      # clamp the result to [0, 1] inorder to use the GAN loss

        self.model = nn.Sequential(*sequence)                                               # get the whole model

    def forward(self, x):
        out = self.model(x)
        if self.use_minibatch_discrimination:
            out = out.view(out.size(0), -1)
            a = out.size(1)
            out = MinibatchDiscrimination(a, a, 3)(out)
        return out
