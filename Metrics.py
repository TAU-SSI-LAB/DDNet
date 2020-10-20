from Imports import *
from pytorch_msssim import ssim

def image_to_tensor(img):

    tensor = torch.from_numpy(np.flip(img, axis=0).copy())
    while tensor.ndim != 4:
        tensor.unsqueeze_(0)
    tensor = tensor.permute(0, -1, 1, 2)
    return tensor


def mse(pred, target):
    if torch.is_tensor(pred):
        return torch.mean((pred - target) ** 2)
    else:
        return np.mean((pred - target) ** 2)


def rmse(pred, target, normalization='cube'):

    if pred.ndim == 1:
        pred = pred / pred.max()
        target = target / target.max()

    elif normalization == 'cube':
        if torch.is_tensor(pred):
            if pred.ndim == 3:
                pred = pred.unsqueeze(0)
                target = target.unsqueeze(0)

            pred = pred / torch.max(pred.reshape(pred.shape[0], -1).max(-1)[0][:, None, None, None],
                             torch.tensor(1e-7).to(pred.device).expand_as(pred))
            target = target / torch.max(target.reshape(target.shape[0], -1).max(-1)[0][:, None, None, None],
                               torch.tensor(1e-7).to(target.device).expand_as(target))
        else:

            if pred.ndim == 3:
                pred = np.expand_dims(pred, 0)
                target = np.expand_dims(target, 0)
            pred = pred / pred.max(tuple(range(1, len(pred.shape))))[:, None, None, None]
            target = target / target.max(tuple(range(1, len(target.shape))))[:, None, None, None]

    elif normalization == 'pixel':
        assert torch.is_tensor(pred), 'Inputs must be Pytorch tensors in pixel normalization'
        pred = pred / torch.max(pred.max(1)[0].unsqueeze(1), torch.tensor(1e-7).to(pred.device).expand_as(pred))
        target = target / torch.max(target.max(1)[0].unsqueeze(1), torch.tensor(1e-7).to(target.device).expand_as(target))

    if torch.is_tensor(pred):
        return torch.sqrt(mse(pred, target))
    else:
        return np.sqrt(mse(pred, target))


def psnr(pred, target, normalization='cube'):
    if torch.is_tensor(pred):
        return -20 * torch.log10(rmse(pred, target, normalization))
    else:
        return -20 * np.log10(rmse(pred, target, normalization))


def psnr_map(pred, target):
    pred = pred / pred.max(0)[0]
    target = target / target.max(0)[0]
    return -20 * torch.log10(((pred - target) ** 2).mean(0).sqrt())


def ssim_score(pred, target, window_size=11, size_average=True):

    assert pred.shape[0] == 1

    if not torch.is_tensor(pred):
        pred = image_to_tensor(pred)
    if not torch.is_tensor(target):
        target = image_to_tensor(target)
    pred = pred.type(torch.float32) / pred.max()
    target = target.type(torch.float32) / target.max()
    return ssim(pred, target, data_range=1, size_average=size_average, win_size=window_size)


def sam(pred, target):

    if torch.is_tensor(pred):
        if pred.ndim == 3:
            pred = pred.permute(1, 2, 0)
        pred = pred.squeeze().cpu().numpy()

    if torch.is_tensor(target):
        if target.ndim == 3:
            target = target.permute(1, 2, 0)
        target = target.squeeze().cpu().numpy()

    axis = -1 if pred.ndim == 3 else None

    prod = (pred * target).sum(axis=axis)
    cos = prod / (np.linalg.norm(pred, axis=axis) * np.linalg.norm(target, axis=axis))

    radians = np.arccos(cos)
    angle = radians * 180 / np.pi

    if angle.ndim > 1:
        angle = angle.mean()
    return angle





