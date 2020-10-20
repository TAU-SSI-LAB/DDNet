from BaseLayers import *


class MultiGpuModel(nn.DataParallel):
    """
    Wrapper to model class for using multi-gpu model.
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class DDNet(nn.Module):

    def __init__(self, params):
        super().__init__()
        self.inputs_channels = params['inputs_channels_list']
        self.kernels = params['kernels_list']
        self.num_levels = params['num_levels']
        self.base_num_filters = params['base_num_filters']
        self.bottleneck_layers = params['bottleneck_layers']
        self.out_channels = params['out_channels']
        self.attention = params['attention']
        self.attention_heads = params['attention_heads']

        assert len(self.kernels) == len(self.inputs_channels), 'Kernels list and inputs channels list must be the ' \
                                                               'same length '

        # Encoder
        self.encoders = nn.ModuleList([])
        for c, k in zip(self.inputs_channels, self.kernels):
            axes = {(1, 9): 'x', (9, 1): 'y', (3, 3): 'xy'}
            self.encoders.append(Encoder(c, k, self.num_levels, self.base_num_filters, self.attention, self.attention_heads, axes=axes[k]))

        # Bottleneck
        self.bottleneck = Bottleneck(self.bottleneck_layers, 2**(self.num_levels - 1)*self.base_num_filters * len(self.inputs_channels),
                                     2**self.num_levels*self.base_num_filters)

        # Decoder
        self.decoder = Decoder((2**self.num_levels)*self.base_num_filters, len(self.inputs_channels), self.num_levels, 3)
        self.final_layer = FinalLayer(self.base_num_filters, self.out_channels)

    def forward(self, x):
        enc_out = None
        skips = [[]] * self.num_levels
        for i, enc in enumerate(self.encoders):
            s, e = enc(x[i])

            for j, level in enumerate(s):
                skips[j] = level if not len(skips[j]) else torch.cat([skips[j], level], dim=1)

            enc_out = e if enc_out is None else torch.cat([enc_out, e], dim=1)

        bottleneck_out = self.bottleneck(enc_out)
        dec_out = self.decoder(bottleneck_out, skips)
        out_cube = self.final_layer(dec_out)

        return out_cube



