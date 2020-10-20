from Metrics import *
from GeneralUtils import *
from Imports import *
import pandas as pd


def calc_test_metrics(pred, target, test_metrics):
    out_dict = defaultdict(float)

    for metric in test_metrics:
        if metric == 'psnr':
            out_dict[metric] = psnr(pred, target).item()
        if metric == 'rmse':
            out_dict[metric] = rmse(pred, target).item()
        if metric == 'ssim':
            out_dict[metric] = ssim_score(pred.unsqueeze(0), target.unsqueeze(0)).item()
        if metric == 'sam':
            out_dict[metric] = sam(pred, target)

    return out_dict


def save_rgb_from_hs(target, save_dir, is_ref):

    if target.dim() == 3:
        target = target.unsqueeze(0)
    target_rgb = hs_to_rgb(target, gamma=True)
    target_rgb = tensor_to_image(target_rgb)

    save_path = os.path.join(save_dir, 'rgb_{}.tiff'.format('ref' if is_ref else 'pred'))
    cv2.imwrite(save_path, cv2.cvtColor(target_rgb, cv2.COLOR_RGB2BGR))


def show_rgb_and_metrics(pred, target, save_dir, metrics, wavelengths, points, show=False):

    if pred.dim() == 3:
        pred = pred.unsqueeze(0)

    if target.dim() == 3:
        target = target.unsqueeze(0)

    pred_rgb = hs_to_rgb(pred, wave_lengths_range=wavelengths)
    pred_rgb = tensor_to_image(pred_rgb)
    target_rgb = hs_to_rgb(target, wave_lengths_range=wavelengths)
    target_rgb = tensor_to_image(target_rgb)

    fig = plt.figure()
    fig.add_subplot(1, 2, 1)

    for idx, point in enumerate(points):
        pred_rgb[point[1], point[0], :] = (0, 1, 0)
        pred_rgb[point[1] + 1, point[0] + 1, :] = (0, 1, 0)
        pred_rgb[point[1] - 1, point[0] - 1, :] = (0, 1, 0)
        pred_rgb[point[1] + 1, point[0] - 1, :] = (0, 1, 0)
        pred_rgb[point[1] - 1, point[0] + 1, :] = (0, 1, 0)
        target_rgb[point[1], point[0], :] = (0, 1, 0)
        target_rgb[point[1] + 1, point[0] + 1, :] = (0, 1, 0)
        target_rgb[point[1] - 1, point[0] - 1, :] = (0, 1, 0)
        target_rgb[point[1] + 1, point[0] - 1, :] = (0, 1, 0)
        target_rgb[point[1] - 1, point[0] + 1, :] = (0, 1, 0)

    plt.imshow(pred_rgb)
    plt.title('Prediction')
    fig.add_subplot(1, 2, 2)
    plt.imshow(target_rgb)
    plt.title('Reference')

    title = ''
    for metric, value in metrics.items():
        title += '{}: {}'.format(metric, value)
        title += '\n'
    plt.suptitle(title)
    if show:
        plt.show()
    plt.savefig(os.path.join(save_dir, 'rgb_ref_pred.tiff'))
    plt.close(fig)


def plot_average_scores(metrics, ids, logger, save_dir):

    n_samples = len(metrics)

    avg_metrics = defaultdict(float)
    all_metrics = defaultdict(list)

    for image_metrics in metrics:
        for metric, value in image_metrics.items():
            avg_metrics[metric] += value
            all_metrics[metric].extend([value])
    for id in ids:
        all_metrics['ids'].extend([id])

    logger.info('Average results over {} images:'.format(n_samples))
    avg_metrics_file = open(os.path.join(save_dir, 'totals.txt'), 'w')
    for metric, value in avg_metrics.items():
        logger.info('Average {}: {}'.format(metric, value / n_samples))
        avg_metrics_file.write('{}: {} \n'.format(metric, value / n_samples))
    avg_metrics_file.close()
    df = pd.DataFrame(all_metrics)
    df.to_csv(os.path.join(save_dir, 'metrics.csv'))


def show_spectral_graphs(pred, target, save_dir, normalize=False, show=False, points=None):

    if points is None:
        points = [(50, 50), (100, 100), (150, 150), (200, 200)]
    pred = tensor_to_image(pred)
    target = tensor_to_image(target)

    n_channels = np.shape(target)[-1]
    lambdas = np.linspace(400 + (31 - n_channels) * 10, 700, n_channels)

    for idx, point in enumerate(points):
        plt.figure()
        ref_spectral_vector = target[point[1], point[0], ...]
        pred_spectral_vector = pred[point[1], point[0], ...]

        psnr_ = psnr(pred_spectral_vector, ref_spectral_vector)
        sam_ = sam(pred_spectral_vector, ref_spectral_vector)
        if normalize:
            ref_spectral_vector /= np.max(ref_spectral_vector)
            pred_spectral_vector /= np.max(pred_spectral_vector)
        plt.plot(lambdas, ref_spectral_vector, 'r--')
        plt.plot(lambdas, pred_spectral_vector, 'b', marker='.')
        plt.xlabel('wavelength(nm)')
        if normalize:
            plt.ylabel('normalized intensity')
        else:
            plt.ylabel('intensity')
        plt.legend(['Reference', 'Prediction, PSNR: %0.2f, SAM: %0.2f$^\circ$' % (psnr_, sam_)], loc='lower right')

        plt.grid(False)
        plt.ylim(0, 1)
        plt.xlim(lambdas[0], lambdas[-1])
        if show:
            plt.show()
        plt.savefig(os.path.join(save_dir, '{}_{}.tiff'.format(point[0], point[1])))
        plt.close()

    return


def show_hs_predictions_results(pred, target, save_dir, wave_lengths, show=False, test_channels=None):

    if test_channels is None:
        test_channels = [1, 4, 8, 13, 15, 17, 25]
    pred = tensor_to_image(pred)
    target = tensor_to_image(target)
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


def show_psnr_map(pred, target, save_dir, show=False):

    psnr_map_tensor = psnr_map(pred, target)
    psnr_map_ = psnr_map_tensor.cpu().numpy()
    plt.figure()
    plt.imshow(psnr_map_, cmap='plasma')
    plt.colorbar()
    plt.title('PSNR Map')
    if show:
        plt.show()
    plt.savefig(os.path.join(save_dir, 'psnr_map.tiff'))
    plt.close()


def rotate_tensor(tensor):
    return torch.rot90(torch.rot90(torch.rot90(tensor, dims=(1, 2)), dims=(1, 2)), dims=(1, 2))

