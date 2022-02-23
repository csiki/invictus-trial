import torch
from torch.nn import Module, Sequential, ModuleList, \
    Conv2d, BatchNorm2d, ReLU, Dropout2d, MaxPool2d, UpsamplingNearest2d, Identity


class EncoderBlock(Module):

    def __init__(self, in_chans, out_chans, dropout, pool_size):
        super().__init__()
        kernel_size = (3, 2)
        stride = (1, 1)

        self.conv = Conv2d(in_chans, out_chans, kernel_size, stride, bias=True, padding='same')
        self.bachnorm = BatchNorm2d(out_chans)
        self.relu = ReLU()
        self.dropout = Dropout2d(dropout)
        self.maxpool = MaxPool2d(pool_size, pool_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.bachnorm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.maxpool(x)
        return x


class DecoderBlock(Module):

    def __init__(self, in_chans, out_chans, upsample_factor):
        super().__init__()
        kernel_size = (3, 2)
        stride = (1, 1)

        self.conv = Conv2d(in_chans, out_chans, kernel_size, stride, bias=True, padding='same')
        self.bachnorm = BatchNorm2d(out_chans)
        self.relu = ReLU()
        # no dropout for decoder
        self.upsample = UpsamplingNearest2d(scale_factor=upsample_factor)  # TODO try bilinear

    def forward(self, x):
        x = self.conv(x)
        x = self.bachnorm(x)
        x = self.relu(x)
        x = self.upsample(x)
        return x


class ResidualBlock(Module):

    def __init__(self, in_chans, out_chans, kernel_size):
        super().__init__()
        stride = (1, 1)
        self.nsubblocks = 3
        
        self.first_skip_layer = Identity() if in_chans == out_chans \
            else Conv2d(in_chans, out_chans, kernel_size, stride, bias=True, padding='same')

        # TODO self.skip_layers = ModuleList([self.first_skip_layer, torch.nn.Identity(), torch.nn.Identity()])
        self.activations = ModuleList([ReLU() for _ in range(self.nsubblocks)])
        self.layers = ModuleList()  # the rest

        for i in range(self.nsubblocks):
            conv = Conv2d(in_chans, out_chans, kernel_size, stride, bias=True, padding='same')
            bachnorm = BatchNorm2d(out_chans)
            self.layers.append(Sequential(conv, bachnorm))
            in_chans = out_chans

    def forward(self, x):
        res = self.first_skip_layer(x)
        for i in range(self.nsubblocks):
            x = self.layers[i](x)
            x = self.activations[i](x)
        return x + res


class EMGModel(Module):

    def __init__(self, dropout, res_block_params):
        super().__init__()
        # TODO

    def forward(self, x):
        pass  # TODO
