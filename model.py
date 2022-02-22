import torch


class EncoderBlock(torch.nn.Module):

    def __init__(self, in_chans, out_chans, dropout, pool_size):
        super().__init__()
        kernel_size = (3, 2)
        stride = (1, 1)

        self.conv = torch.nn.Conv2d(in_chans, out_chans, kernel_size, stride, bias=True)
        self.bachnorm = torch.nn.BatchNorm2d(out_chans)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout2d(dropout)
        self.maxpool = torch.nn.MaxPool2d(pool_size, pool_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.bachnorm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.maxpool(x)
        return x


class DecoderBlock(torch.nn.Module):

    def __init__(self, in_chans, out_chans, upsample_factor):
        super().__init__()
        kernel_size = (3, 2)
        stride = (1, 1)

        self.conv = torch.nn.Conv2d(in_chans, out_chans, kernel_size, stride, bias=True)
        self.bachnorm = torch.nn.BatchNorm2d(out_chans)
        self.relu = torch.nn.ReLU()
        # no dropout for decoder
        self.upsample = torch.nn.UpsamplingNearest2d(scale_factor=upsample_factor)  # TODO try bilinear

    def forward(self, x):
        x = self.conv(x)
        x = self.bachnorm(x)
        x = self.relu(x)
        x = self.upsample(x)
        return x


class ResidualBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # TODO
