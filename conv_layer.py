"""Convolutional layer

PyTorch implementation of CapsNet in Sabour, Hinton et al.'s paper
Dynamic Routing Between Capsules. NIPS 2017.
https://arxiv.org/abs/1710.09829

Author: Cedric Chee
"""

import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    """
    Conventional Conv2d layer
    """

    def __init__(self, in_channel, out_channel, kernel_size, relu6=False):
        super(ConvLayer, self).__init__()

        self.conv0 = nn.Conv2d(in_channels=in_channel,
                               out_channels=out_channel,
                               kernel_size=kernel_size,
                               stride=1)
        if relu6:
            self.act = nn.ReLU6(inplace=True)
        else:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass"""
        # x shape: [128, 1, 28, 28]
        # out_conv0 shape: [128, 256, 20, 20]
        out_conv0 = self.conv0(x)
        # out_relu shape: [128, 256, 20, 20]
        out_relu = self.act(out_conv0)
        return out_relu
