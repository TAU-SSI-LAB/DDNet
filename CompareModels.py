import hdf5storage
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from SSIM import SSIMLoss
import cv2
#from TestUtils import show_hs_predictions_results

def image_to_tensor(img):

    tensor = torch.from_numpy(np.flip(img, axis=0).copy())
    while tensor.ndim != 4:
        tensor.unsqueeze_(0)
    tensor = tensor.permute(0, -1, 1, 2)
    return tensor
def ssim_score(pred, target, window_size=11, size_average=True, stride=1):

    if pred.shape[0] != 1:
        pred = np.expand_dims(pred, 0)
        target = np.expand_dims(target, 0)


    if not torch.is_tensor(pred):
        pred = image_to_tensor(pred)
    if not torch.is_tensor(target):
        target = image_to_tensor(target)
    pred = pred.type(torch.float32) / pred.max()
    target = target.type(torch.float32) / target.max()
    return 1 - 2 * SSIMLoss(window_size=window_size, size_average=size_average, stride=stride)(pred, target)


def hs_to_rgb(cube, wave_lengths_range=None, gamma=False):

    # These values correspond to 400-720 nm (33 spectral bands with 10nm bandwidth)
    x = np.array([0.0143, 0.0435, 0.1344, 0.2839, 0.3483, 0.3362, 0.2908, 0.1954, 0.0956, 0.032, 0.0049, 0.0093, 0.0633, 0.1655,
                  0.2904, 0.4335, 0.5945, 0.7621, 0.9163, 1.0263, 1.0622, 1.0026, 0.8545, 0.6424, 0.4479, 0.2835,
                  0.1649, 0.0874, 0.0468, 0.0227, 0.0114, 0.0058, 0.0029])

    y = np.array([0.000396, 0.0012, 0.004, 0.0116, 0.023, 0.038, 0.06, 0.091, 0.139, 0.208, 0.323, 0.503, 0.71, 0.862, 0.954, 0.995,
                  0.995, 0.952, 0.87, 0.757, 0.631, 0.503, 0.381, 0.265, 0.175, 0.107, 0.061, 0.032, 0.017, 0.0082,
                  0.0041, 0.0021, 0.0010])
    z = np.array([0.0679, 0.2074, 0.6456, 1.3856, 1.7471, 1.7721, 1.6692, 1.2876, 0.8130, 0.4652, 0.272, 0.1582, 0.0783, 0.0422,
                  0.0203, 0.0088, 0.0039, 0.0021, 0.0017, 0.0011, 0.0008, 0.00034, 0.00019, 0.00005, 0.00002, 0.,
                  0., 0., 0., 0., 0., 0., 0.])




    min_index = (max(400, wave_lengths_range.min()) - 400) // 10
    max_index = (min(wave_lengths_range.max(), 720) - 400) // 10


    spectral_range = range(min_index, max_index + 1)

    x = x[spectral_range]
    y = y[spectral_range]
    z = z[spectral_range]

    if 400 in wave_lengths_range:
        min_cube_index = np.where(wave_lengths_range == 400)[0][0]
    else:
        min_cube_index = 0

    if 720 in wave_lengths_range:
        max_cube_index = np.where(wave_lengths_range == 720)[0][0]
    else:
        max_cube_index = -1

    if isinstance(cube, torch.Tensor):
        bs = cube.shape[0]
        x, y, z = map(lambda p: torch.from_numpy(p).type(cube.dtype).to(cube.device), [x, y, z])
        cube = cube.transpose(0, 1)
        X, Y, Z = map(lambda p: torch.mul(cube, p.unsqueeze(1).unsqueeze(2).unsqueeze(3)).sum(0), [x, y, z])
        max_val = torch.cat([X.unsqueeze(1), Y.unsqueeze(1), Z.unsqueeze(1)], dim=1).view(bs, -1).max(-1)[0]
        X, Y, Z = map(lambda p: p / max_val[:, None, None], [X, Y, Z])
        X, Y, Z = map(lambda p: torch.clamp(p, min=0), [X, Y, Z])

        R = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
        G = - 0.9689 * X + 1.87582 * Y + 0.0414 * Z
        B = 0.0557 * X - 0.2040 * Y + 1.0570 * Z

        R, G, B = map(lambda p: torch.clamp(p, min=0, max=1).unsqueeze(1), [R, G, B])

        return torch.cat([R, G, B], dim=1) ** ((1 / 0.45) if gamma else 1)

    elif isinstance(cube, np.ndarray):
        X, Y, Z = map(lambda p: np.dot(cube, np.expand_dims(p, -1)), [x, y, z])
        max_val = np.max(np.concatenate([X, Y, Z], axis=-1))
        X, Y, Z = map(lambda p: p / max_val, [X, Y, Z])
        X, Y, Z = map(lambda p: np.clip(p, 0, None), [X, Y, Z])

        R = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
        G = - 0.9689 * X + 1.87582 * Y + 0.0414 * Z
        B = 0.0557 * X - 0.2040 * Y + 1.0570 * Z

        R, G, B = map(lambda p: np.clip(p, 0, 1), [R, G, B])

        return np.concatenate([R, G, B], axis=-1) ** ((1 / 0.45) if gamma else 1)


def psnr_map(pred, target):

    if not torch.is_tensor(pred):
        pred = torch.tensor(pred).permute(2, 0, 1)
        target = torch.tensor(target).permute(2, 0, 1)
    pred = pred / pred.max(0)[0]
    target = target / target.max(0)[0]
    return -20 * torch.log10(((pred - target) ** 2).mean(0).sqrt())

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
    pred = pred / pred.max()
    target = target / target.max()
    return -20 * np.log10(np.sqrt(np.mean((pred - target) ** 2)))


def show_spectral_graphs_(cubes, ref_cube, save_dir, strings, normalize=False, points=None):

    colors = ['blue', 'springgreen', 'lime', 'yellow', 'pink']
    if points is None:
        points = [(50, 50), (100, 100), (150, 150), (200, 200)]

    n_channels = np.shape(cubes[0])[-1]
    lambdas = np.linspace(400 + (31 - n_channels) * 10, 700, n_channels)

    fig = plt.figure(figsize=(5, 4))

    for idx, point in enumerate(points):
        ax = fig.add_subplot(1, len(points), idx + 1)
        spectras = []
        psnrs = []
        sams = []
        title = ''
        legend_strings = ['Reference']
        ref_spectral_vector = ref_cube[point[1], point[0], ...]
        ax.plot(lambdas, ref_spectral_vector / np.max(ref_spectral_vector) if normalize else ref_spectral_vector, 'r--')
        for i, cube in enumerate(cubes):
            pred_spectral_vector = cube[point[1], point[0], ...]
            pred_spectral_vector[pred_spectral_vector < 0] = 0
            psnrs.append(psnr(pred_spectral_vector, ref_spectral_vector))
            sams.append(sam(pred_spectral_vector, ref_spectral_vector))
            if normalize:
                pred_spectral_vector /= np.max(pred_spectral_vector)
            spectras.append(pred_spectral_vector)

            ax.plot(lambdas, pred_spectral_vector, colors[i], marker='.')
            legend_strings.append(strings[i])
            title +='%s: %0.2f dB / %0.2f$^\circ$\n' % (strings[i],psnrs[i], sams[i])

        plt.title(title, fontsize='xx-large')
        plt.xlabel('$\lambda$[nm]', fontsize='xx-large')
        plt.xticks([450, 500, 550, 600, 650, 700], fontsize='xx-large')
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0],  fontsize='xx-large')
        if normalize:
            plt.ylabel('Norm. Int', fontsize='xx-large')
        else:
            plt.ylabel('Int', fontsize='xx-large')

        plt.grid(False)
        plt.ylim(0, 1)
        plt.xlim(lambdas[0], lambdas[-1])
        #if idx == 1:
        #    ax.legend(legend_strings, loc='upper center', bbox_to_anchor=(0.5, -0.05),
        #              ncol=3, fancybox=True, shadow=True, fontsize='medium')

    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, '{}_compare.tiff'.format(cube_name)))
    plt.close()

    return


def show_hs_predictions_results_(pred, target, save_dir, wave_lengths, show=False, test_channels=None):

    if test_channels is None:
        test_channels = [1, 4, 8, 13, 15, 17, 25]

    assert len(pred.shape) == 3
    if pred.shape[-1] == 31:
        pred = pred[..., 2:]

    assert len(target.shape) == 3
    if target.shape[-1] == 31:
        target = target[..., 2:]

    fig = plt.figure(figsize=(50, 50))
    columns = len(test_channels)
    rows = 2
    for idx, c in enumerate(test_channels):

        lambda_ = 420 + c * 10
        pred_channel = pred[..., c]
        target_channel = target[..., c]
        color = np.zeros_like(pred)
        color[..., c] = 1
        color = hs_to_rgb(color, wave_lengths_range=wave_lengths, gamma=False)
        color = np.array([color[0, 0, 0], color[0, 0, 1], color[0, 0, 2]])

        pred_color = pred_channel[:, :, None] * color[None, None, :]
        target_color = target_channel[:, :, None] * color[None, None, :]

        cv2.imwrite(os.path.join(save_dir, 'ref{}nm.tiff'.format(lambda_)),
                    cv2.cvtColor(np.uint8(target_color * 255), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_dir, 'pred{}nm.tiff'.format(lambda_)),
                    cv2.cvtColor(np.uint8(pred_color * 255), cv2.COLOR_RGB2BGR))

        psnr_ = psnr(pred[..., c], target[..., c])
        ssim = ssim_score(pred[..., c], target[..., c])

        ax = fig.add_subplot(rows, columns, idx + 1)
        ax.text(0.95, 0.01, str(wave_lengths[c]) + 'nm',
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes,
                color='white', fontsize=10)
        plt.axis('off')
        plt.imshow(target_color, vmin=0, vmax=target.max())
        plt.title('Ref')
        ax = fig.add_subplot(rows, columns, idx + len(test_channels) + 1)
        ax.text(0.95, 0.01, str(wave_lengths[c]) + 'nm',
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes,
                color='white', fontsize=10)
        plt.imshow(pred_color, vmin=0, vmax=pred.max())
        plt.title('Rec' + '\n' + '%0.2f/%0.3f' % (psnr_, ssim))
        plt.axis('off')
        plt.tight_layout()
    if show:
        plt.show()
    plt.savefig(os.path.join(save_dir, 'color_spectral_plots.tiff'))
    plt.close(fig)


if __name__ == '__main__':

    cube_name = 'Im_Mono_Name_9064_Row_4'
    cube_ref = hdf5storage.loadmat('/Users/amitz/Documents/thesis/DDNetPaper/figures/2DIpad/{}/cube2DRef.mat'.format(cube_name))['Cube']
    cube_2 = hdf5storage.loadmat('/Users/amitz/Documents/thesis/DDNetPaper/figures/2DIpad/{}/cube2D.mat'.format(cube_name))['Cube']
    cube_1 = hdf5storage.loadmat('/Users/amitz/Documents/thesis/DDNetPaper/figures/1DIpad/{}/cube2D.mat'.format(cube_name))['Cube']
    psnr2 = psnr_map(cube_2, cube_ref)

    ref_rgb = hs_to_rgb(cube_ref, wave_lengths_range=np.array(range(420, 710, 10)))
    cube_1_rgb = hs_to_rgb(cube_1, wave_lengths_range=np.array(range(420, 710, 10)))
    cube_2_rgb = hs_to_rgb(cube_2, wave_lengths_range=np.array(range(420, 710, 10)))

    print(ssim_score(ref_rgb, cube_1_rgb))
    print(ssim_score(ref_rgb, cube_2_rgb))

    #plt.subplot(121)
    #plt.imshow(psnr2, cmap='plasma')
    #plt.subplot(122)
    #plt.imshow(ref_rgb)
    #plt.show()
    #points = [(173, 45)]
    save_dir = '/Users/amitz/Documents/thesis/DDNetPaper/figures/paper_revision'
    os.makedirs(save_dir, exist_ok=True)
    show_hs_predictions_results_(cube_1, cube_ref, save_dir, np.array(range(420, 710, 10)))
    #show_spectral_graphs_([cube_1, cube_2], cube_ref, save_dir, ['1D', '2D'], normalize=True, points=points)








