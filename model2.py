import numpy as np
import torch
from torch.nn import Module, Sequential, ModuleList, \
    Conv2d, BatchNorm2d, ReLU, Dropout2d, MaxPool2d, UpsamplingNearest2d, Identity, LayerNorm, GELU, \
    UpsamplingBilinear2d, ConvTranspose2d, Conv1d, ConvTranspose1d, BatchNorm1d, Dropout
from model import bReLU

# structure: conv1d encoder, transposed conv1d decoder, variational bottleneck
#   try dilation on conv1ds


class EMGModel2(Module):

    def __init__(self, dropout, res_block_params=None):
        super().__init__()

        act_fun_cls = ReLU

        # encoder
        in_chan = 8
        chans = [32, 32, 64, 128, 256]
        kernel = [5, 3, 3, 3, 3]
        stride = [2, 2, 2, 2, 2]
        padding = [0, 0, 0, 0, 0]
        dilation = [1, 1, 1, 1, 1]

        convs = []
        for i in range(len(chans)):
            convs.append(Conv1d(in_chan, chans[i], kernel[i], stride[i], padding[i], dilation[i]))
            in_chan = chans[i]

        encoder_blocks = []
        for i in range(len(chans)):
            encoder_blocks.append(Sequential(
                convs[i],
                BatchNorm1d(chans[i]),
                act_fun_cls(),
                Dropout(dropout),
            ))
        self.encoder = ModuleList(encoder_blocks)

        # variational
        variational_in = 128
        variational_dim = 256
        self.kld_loss = None
        self.mean_lin = torch.nn.Linear(variational_in, variational_dim)
        self.var_lin = torch.nn.Linear(variational_in, variational_dim)

        # decoder
        in_chan = 256
        chans = chans[-2::-1] + [10]
        kernel = kernel[::-1]
        stride = stride  # [2, 2, 2, 2]
        padding = padding  # [0, 0, 0, 0]
        dilation = [0, 0, 1, 1, 1]

        convs = []
        for i in range(len(chans)):  # TODO transposed
            convs.append(ConvTranspose1d(in_chan, chans[i], kernel[i], stride[i], padding[i], dilation[i]))
            in_chan = chans[i]

        decoder_blocks = []
        for i in range(len(chans)):
            decoder_blocks.append(Sequential(
                convs[i],
                BatchNorm1d(chans[i]),
                act_fun_cls() if i == len(chans) - 1 else bReLU(1.),
                Dropout(dropout),
            ))
        self.decoder = ModuleList(decoder_blocks)

        # TODO conv to reshape to right channel size

    def forward(self, x):

        for module in self.encoder:
            x = module(x)

        # var
        h_x_shape = x.shape
        x = torch.permute(x, (0, 2, 1))  # batch x t x c
        x = torch.reshape(x, (-1, h_x_shape[1]))

        mean = self.mean_lin(x[:, :128])
        log_var = self.var_lin(x[:, 128:])
        x = self._variational(mean, log_var)  # batch x seq x loc_ctx_dim  # TODO add var error

        x = torch.reshape(x, (-1, h_x_shape[2], h_x_shape[1]))
        x = torch.permute(x, (0, 2, 1))

        # TODO dec
        for module in self.decoder:
            x = module(x)

        # (out - out_target) = (kernel - 1) / 2

        return x

    def _variational(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        self.kld_loss = (-0.5 * torch.sum(log_var - mu ** 2 - torch.exp(log_var) + 1, 1)).mean().squeeze()
        return mu + std * eps


if __name__ == '__main__':
    model = EMGModel2(0.4)
    x = torch.zeros((32, 8, 1000))
    y = model(x)
    print(y.shape)
