import numpy as np
import torch
from torch.nn import Module, Sequential, ModuleList, \
    Conv2d, BatchNorm2d, ReLU, Dropout2d, MaxPool2d, UpsamplingNearest2d, Identity, LayerNorm, GELU, \
    UpsamplingBilinear2d, ConvTranspose2d


class bReLU(Module):
    def __init__(self, max_val):
        super().__init__()
        self.max_val = max_val

    def forward(self, x):
        return torch.clamp(torch.relu(x), None, self.max_val)


class EncoderBlock(Module):

    def __init__(self, in_chans, out_chans, dropout, pool_size, act_fun: Module = ReLU()):
        super().__init__()
        kernel_size = (3, 2)
        stride = (1, 1)

        self.conv = Conv2d(in_chans, out_chans, kernel_size, stride, bias=True, padding='same')
        self.bachnorm = BatchNorm2d(out_chans)
        self.relu = act_fun
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

    def __init__(self, in_chans, out_chans, dropout, upsample_factor, kernel_size=(3, 2), act_fun: Module = ReLU()):  # TODO try snake
        super().__init__()
        stride = (1, 1)

        self.conv = Conv2d(in_chans, out_chans, kernel_size, stride, bias=True, padding='same')
        self.bachnorm = BatchNorm2d(out_chans)
        self.relu = act_fun
        # self.dropout = Dropout2d(dropout)
        self.upsample = UpsamplingNearest2d(size=upsample_factor)
        # UpsamplingBilinear2d(scale_factor=upsample_factor)  # UpsamplingNearest2d(scale_factor=upsample_factor)

        # upstride = upsample_factor
        # out_pad = (upsample_factor[0] - 1, upsample_factor[1] - 1)
        # dilation = (self._find_dilation(kernel_size[0]), self._find_dilation(self._find_dilation(kernel_size[1])))
        # padding = (dilation[0] * (kernel_size[0] - 1) // 2, dilation[1] * (kernel_size[1] - 1) // 2)
        # self.upsample = ConvTranspose2d(out_chans, out_chans, kernel_size, upstride,
        #                                 padding=padding, output_padding=out_pad, dilation=1)

    @staticmethod
    def _find_dilation(kernel):
        for dilation in range(1, 5):
            pad = dilation * (kernel - 1) / 2
            if pad.is_integer():
                return dilation
        raise ValueError(f'no dilation for kernel {kernel}')

    def forward(self, x):
        x = self.conv(x)
        x = self.bachnorm(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.upsample(x)
        return x


class ResidualBlock(Module):

    def __init__(self, in_chans, out_chans, kernel_size, act_fun_cls=ReLU):
        super().__init__()
        stride = (1, 1)
        self.nsubblocks = 3
        
        self.first_skip_layer = Identity() if in_chans == out_chans \
            else Conv2d(in_chans, out_chans, kernel_size, stride, bias=True, padding='same')

        # TODO self.skip_layers = ModuleList([self.first_skip_layer, torch.nn.Identity(), torch.nn.Identity()])
        self.activations = ModuleList([act_fun_cls() for _ in range(self.nsubblocks)])
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

        act_fun_cls = ReLU

        self.encoder = Sequential(
            EncoderBlock(1, 32, dropout, (5, 2), act_fun_cls()),
            EncoderBlock(32, 128, dropout, (4, 2), act_fun_cls()),
            EncoderBlock(128, 256, dropout, (2, 2), act_fun_cls()),
        )

        self.resnet = Sequential(*[ResidualBlock(*p, act_fun_cls=act_fun_cls) for p in res_block_params])

        # this is for output of 16
        self.decoder = Sequential(
            # DecoderBlock(256, 128, (5, 5), kernel_size=(3, 1), act_fun=act_fun_cls()),  # TODO why 3,2 here? when only 1 chan
            # DecoderBlock(128, 32, (4, 2), kernel_size=(3, 2), act_fun=act_fun_cls()),
            # DecoderBlock(32, 1, (2, 1), kernel_size=(3, 2), act_fun=bReLU(1.)),

            DecoderBlock(256, 128, dropout, (125, 2), kernel_size=(3, 1), act_fun=act_fun_cls()),
            DecoderBlock(128, 64, dropout, (250, 4), kernel_size=(3, 2), act_fun=act_fun_cls()),
            DecoderBlock(64, 32, dropout, (500, 8), kernel_size=(3, 2), act_fun=act_fun_cls()),
            DecoderBlock(32, 32, dropout, (500, 10), kernel_size=(3, 2), act_fun=act_fun_cls()),
            DecoderBlock(32, 1, dropout, (1000, 10), kernel_size=(3, 2), act_fun=bReLU(1.)),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.resnet(x)
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    res_block_params = [(256, 256, (5, 1)) for _ in range(5)]
    model = EMGModel(0.4, res_block_params)
    x = torch.zeros((32, 1, 1000, 8))
    y = model(x)
    print(y.shape)

    br = bReLU(5)
    y = br(torch.arange(-5, 10, 1))
    print(y)

    # speed measurement
    repetitions = 100
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    dummy_input = torch.zeros((1, 1, 1000, 8))
    measurements = []

    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            meas = starter.elapsed_time(ender)
            measurements.append(meas)
            print('.', end='')

    print('mean std runtime:', np.mean(measurements[1:]), np.std(measurements[1:]))
