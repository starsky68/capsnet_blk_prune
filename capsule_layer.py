"""Capsule layer

PyTorch implementation of CapsNet in Sabour, Hinton et al.'s paper
Dynamic Routing Between Capsules. NIPS 2017.
https://arxiv.org/abs/1710.09829

Author: Cedric Chee
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import utils
from agprune import *


class CapsuleLayer_blk_pru(nn.Module):
    """
    The core implementation of the idea of capsules
    """

    def __init__(self, in_unit, in_channel, num_unit, unit_size, use_routing,
                 num_routing, cuda_enabled, use_blk=False, t=1, prune_w=False, prune_b=False):
        super(CapsuleLayer_blk_pru, self).__init__()

        self.in_unit = in_unit
        self.in_channel = in_channel
        self.num_unit = num_unit
        self.unit_size=unit_size
        self.use_routing = use_routing
        self.num_routing = num_routing
        self.cuda_enabled = cuda_enabled

        self.use_blk = use_blk
        self.prune_w = prune_w
        self.prune_b = prune_b

        if self.use_routing:
            """
            Based on the paper, DigitCaps which is capsule layer(s) with
            capsule inputs use a routing algorithm that uses this weight matrix, Wij
            """
            # weight shape:
            # [1 x primary_unit_size x num_classes x output_unit_size x num_primary_unit]
            # == [1 x 1152 x 10 x 16 x 8]
            self.weight = nn.Parameter(torch.randn(1, in_channel, num_unit, unit_size, in_unit))

        else:
            """
            According to the CapsNet architecture section in the paper,
            we have routing only between two consecutive capsule layers (e.g. PrimaryCapsules and DigitCaps).
            No routing is used between Conv1 and PrimaryCapsules.

            This means PrimaryCapsules is composed of several convolutional units.
            """
            # Define 8 convolutional units.
            if self.use_blk:
                self.conv_units = nn.ModuleList([
                    blk(self.in_channel, 32, 9, 2, t) for u in range(self.num_unit)
                ])
            else:
                self.conv_units = nn.ModuleList([
                    nn.Conv2d(self.in_channel, 32, 9, 2) for u in range(self.num_unit)
                ])

    # def get_weight(self, ep):
    #     ts = AgPrune(0.1, 0.1, ep, 0, 10, 5).AgpPruningRate()
    #     return Prune.apply(self.weight, ts)

    def forward(self, x, ep):
        if self.use_routing:
            # Currently used by DigitCaps layer.
            return self.routing(x, ep)
        else:
            # Currently used by PrimaryCaps layer.
            return self.no_routing(x)

    def routing(self, x, ep):
        """
        Routing algorithm for capsule.

        :input: tensor x of shape [128, 8, 1152]

        :return: vector output of capsule j
        """
        batch_size = x.size(0)

        x = x.transpose(1, 2) # dim 1 and dim 2 are swapped. out tensor shape: [128, 1152, 8]

        # Stacking and adding a dimension to a tensor.
        # stack ops output shape: [128, 1152, 10, 8]
        # unsqueeze ops output shape: [128, 1152, 10, 8, 1]
        x = torch.stack([x] * self.num_unit, dim=2).unsqueeze(4)

        # Convert single weight to batch weight.
        # [1 x 1152 x 10 x 16 x 8] to: [128, 1152, 10, 16, 8]
        # prune_w = AgPrune(0.1, 0.1, ep, 0, 10, 5)
        # pruner_W = prune_w.apply_prune(self.weight)
        # w = self.get_weight(ep)
        if self.prune_w:
            ts_w = AgPrune(0.3, 0.6, ep, 0, 50, 1).AgpPruningRate()
            w = Prune.apply(self.weight, ts_w)
            batch_weight = torch.cat([w] * batch_size, dim=0)
        else:
            batch_weight = torch.cat([self.weight] * batch_size, dim=0)
        # u_hat is "prediction vectors" from the capsules in the layer below.
        # Transform inputs by weight matrix.
        # Matrix product of 2 tensors with shape: [128, 1152, 10, 16, 8] x [128, 1152, 10, 8, 1]
        # u_hat shape: [128, 1152, 10, 16, 1]
        u_hat = torch.matmul(batch_weight, x)

        # All the routing logits (b_ij in the paper) are initialized to zero.
        # self.in_channel = primary_unit_size = 32 * 6 * 6 = 1152
        # self.num_unit = num_classes = 10
        # b_ij shape: [1, 1152, 10, 1]
        b_ij = Variable(torch.zeros(1, self.in_channel, self.num_unit, 1))
        if self.cuda_enabled:
            b_ij = b_ij.cuda()

        # From the paper in the "Capsules on MNIST" section,
        # the sample MNIST test reconstructions of a CapsNet with 3 routing iterations.
        num_iterations = self.num_routing

        for iteration in range(num_iterations):
            # Routing algorithm

            # Calculate routing or also known as coupling coefficients (c_ij).
            # c_ij shape: [1, 1152, 10, 1]
            c_ij = F.softmax(b_ij, dim=2)  # Convert routing logits (b_ij) to softmax.
            # c_ij shape from: [128, 1152, 10, 1] to: [128, 1152, 10, 1, 1]

            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)


            # Implement equation 2 in the paper.
            # s_j is total input to a capsule, is a weigthed sum over all "prediction vectors".
            # u_hat is weighted inputs, prediction ˆuj|i made by capsule i.
            # c_ij * u_hat shape: [128, 1152, 10, 16, 1]
            # s_j output shape: [batch_size=128, 1, 10, 16, 1]
            # Sum of Primary Capsules outputs, 1152D becomes 1D.
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)

            # Squash the vector output of capsule j.
            # v_j shape: [batch_size, weighted sum of PrimaryCaps output,
            #             num_classes, output_unit_size from u_hat, 1]
            # == [128, 1, 10, 16, 1]
            # So, the length of the output vector of a capsule is 16, which is in dim 3.
            v_j = utils.squash(s_j, dim=3)

            # in_channel is 1152.
            # v_j1 shape: [128, 1152, 10, 16, 1]
            v_j1 = torch.cat([v_j] * self.in_channel, dim=1)

            # The agreement.
            # Transpose u_hat with shape [128, 1152, 10, 16, 1] to [128, 1152, 10, 1, 16],
            # so we can do matrix product u_hat and v_j1.
            # u_vj1 shape: [1, 1152, 10, 1]
            u_vj1 = torch.matmul(u_hat.transpose(3, 4), v_j1).squeeze(4).mean(dim=0, keepdim=True)

            # Update routing (b_ij) by adding the agreement to the initial logit.


            if self.prune_b:
                ts_b = AgPrune(0.2, 0.6, ep, 0, 50, 1).AgpPruningRate()
                b_ij = Prune.apply(b_ij + u_vj1, ts_b)
            else:
                b_ij = b_ij + u_vj1
        return v_j.squeeze(1) # shape: [128, 10, 16, 1]

    def no_routing(self, x):
        """
        Get output for each unit.
        A unit has batch, channels, height, width.
        An example of a unit output shape is [128, 32, 6, 6]

        :return: vector output of capsule j
        """
        # Create 8 convolutional unit.
        # A convolutional unit uses normal convolutional layer with a non-linearity (squash).
        unit = [self.conv_units[i](x) for i, l in enumerate(self.conv_units)]

        # Stack all unit outputs.
        # Stacked of 8 unit output shape: [128, 8, 32, 6, 6]
        unit = torch.stack(unit, dim=1)

        batch_size = x.size(0)

        # Flatten the 32 of 6x6 grid into 1152.
        # Shape: [128, 8, 1152]
        unit = unit.view(batch_size, self.num_unit, -1)

        # Add non-linearity
        # Return squashed outputs of shape: [128, 8, 1152]
        return utils.squash(unit, dim=2) # dim 2 is the third dim (1152D array) in our tensor

class blk(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, t):
        super(blk, self).__init__()

        expand_channels = 256 * t
        self.conv1 = nn.Conv2d(in_channel, expand_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_channels)
        self.nolinear1 = nn.ReLU6(inplace=True)
        self.conv2 = nn.Conv2d(expand_channels, expand_channels, kernel_size=kernel_size, stride=stride,
                               groups=expand_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_channels)
        self.nolinear2 = nn.ReLU6(inplace=True)
        self.conv3 = nn.Conv2d(expand_channels, out_channel, kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)

    def forward(self,x):
        x = self.nolinear1(self.bn1(self.conv1(x)))
        x = self.nolinear2(self.bn2(self.conv2(x)))
        out = self.bn3(self.conv3(x))
        return out