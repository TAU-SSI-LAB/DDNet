from Imports import *


class Noise:

    def __init__(self, noise, **kwargs):
        self.noise = noise
        if 'Gaussian' in self.noise:
            assert 'sigma' in kwargs, 'Missing sigma for gaussian noise'
            self.sigma = kwargs['sigma']
        if 'Quantization' in self.noise:
            assert 'bits' in kwargs, 'Missing bits for quantization noise'
            self.bits = kwargs['bits']

    def __call__(self, mono):
        mono = mono[-1].unsqueeze(0) if mono[-1].dim() == 3 else mono[-1]
        for noise in self.noise:
            if noise == 'Gaussian':
                mono += torch.randn_like(mono) * self.sigma
            if noise == 'Poisson':
                sigma = torch.sqrt(mono)
                mono += torch.randn_like(mono) * sigma
            if noise == 'Quantization':
                mono = mono * 2 ** self.bits
                mono = torch.round(mono)
                mono = mono / (2 ** self.bits)

        mono_out = [mono[:, :3, ...], torch.cat((mono[:, 0, ...].unsqueeze(1), mono[:, -2:, ...]), dim=1), mono]
        mono_out = [m.squeeze(0) for m in mono_out]
        return mono_out










