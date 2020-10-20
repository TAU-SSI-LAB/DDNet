from Imports import *
from Metrics import *
from GeneralUtils import *
from pytorch_msssim import ssim
from torch.autograd import Variable


class Losses:

    def __init__(self, **args):

        self.params = args['params']

    @staticmethod
    def mse_loss(pred, target):
        return mse(pred, target)

    @staticmethod
    def rmse_loss(pred, target, normalization='cube'):
        return rmse(pred, target, normalization)

    def rgb_loss(self, pred, target, metric='ssim'):
        rgb_pred = hs_to_rgb(pred, self.params['wave_lengths'])
        rgb_target = hs_to_rgb(target, self.params['wave_lengths'])
        if metric == 'ssim':
            return self.ssim_loss(rgb_pred, rgb_target)
        elif metric == 'rmse':
            return self.rmse_loss(rgb_pred, rgb_target)

    @staticmethod
    def ssim_loss(pred, target, window_size=11, size_average=True):
        return 1 - ssim(pred, target, data_range=1, size_average=size_average, win_size=window_size)


    @staticmethod
    def blur_loss(pred, target, patch_size=9):
        b, c, h, w = pred.shape
        kernel = Variable(torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])).to(pred.device).type_as(pred).view(1, 1, 3, 3)
        kernel = kernel.expand(c, 1, 3, 3)
        pred_lap = F.conv2d(pred, kernel, groups=c, padding=1)
        target_lap = F.conv2d(target, kernel, groups=c, padding=1)
        pred_lap_patches = torch.nn.functional.unfold(pred_lap.contiguous(), kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size),
                                                  padding=(patch_size // 2, patch_size // 2)).view(b, c, patch_size * patch_size, -1)
        target_lap_patches = torch.nn.functional.unfold(target_lap.contiguous(), kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size),
                                                  padding=(patch_size // 2, patch_size // 2)).view(b, c, patch_size * patch_size, -1)
        pred_var = torch.var(pred_lap_patches, dim=[2])
        target_var = torch.var(target_lap_patches, dim=[2])
        blur_loss = torch.abs(pred_var - target_var).mean()

        return blur_loss

    @staticmethod
    def spectral_loss(pred, target):

        pred_grad = pred[:, 2:, ...] - pred[:, :-2, ...]
        target_grad = target[:, 2:, ...] - target[:, :-2, ...]
        return (pred_grad - target_grad).abs().sum(1).mean()

    def calc_loss(self, pred, target, metrics):

        loss = 0

        for loss_function in self.losses.keys():
            if loss_function == 'rmse':
                rmse_loss = self.rmse_loss(pred, target, normalization='pixel')
                loss += rmse_loss * self.losses[loss_function]
                metrics[loss_function] += rmse_loss.item() * target.size(0)
            elif loss_function == 'mse':
                mse_loss = self.mse_loss(pred, target)
                loss += mse_loss * self.losses[loss_function]
                metrics[loss_function] += mse_loss.item() * target.size(0)
            elif loss_function == 'rgb':
                rgb_loss = self.rgb_loss(pred, target)
                loss += rgb_loss * self.losses[loss_function]
                metrics[loss_function] +=rgb_loss.item() * target.size(0)
            elif loss_function == 'ssim':
                ssim_loss = self.ssim_loss(pred, target)
                loss += ssim_loss * self.losses[loss_function]
                metrics[loss_function] += ssim_loss.item() * target.size(0)
            elif loss_function == 'spectral':
                spectral_loss = self.spectral_loss(pred, target)
                loss += spectral_loss * self.losses[loss_function]
                metrics[loss_function] += spectral_loss.item() * target.size(0)
            elif loss_function == 'blur':
                blur_loss = self.blur_loss(pred, target)
                loss += blur_loss * self.losses[loss_function]
                metrics[loss_function] += blur_loss.item() * target.size(0)

        metrics['loss'] += loss.item() * target.size(0)

        for metric in self.metrics:
            if metric == 'psnr':
                metrics[metric] += psnr(pred, target).item() * target.size(0)
            if metric == 'ssim':
                metrics[metric] += ssim_score(pred, target).item() * target.size(0)

        return loss




