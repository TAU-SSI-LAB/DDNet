from TestUtils import *
from DataLoader import HSDataLoader
from DataTransforms import *
import time
import hdf5storage


class Testing:

    def __init__(self, train=False, logger=None, **args):

        super().__init__()
        self.params = args['params']
        if logger is None:
            logging.basicConfig(format='%(asctime)s %(message)s')
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        if not train:
            self.get_model()
        self.display_on = False
        self.test_options = self.params['test_options']

    def get_model(self):

        self.model = self.params['model_type'](self.params['architecture_params'])
        state_dict = torch.load(self.params['test_weights_path'])
        state_dict = fix_state_dict(state_dict)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        return

    def test_scheme(self, data_loader, **noise_params):
        total_time = 0
        label_exists = False
        steps = len(data_loader)
        metrics = [defaultdict(float)] * steps
        ids = []
        with torch.no_grad():

            with tqdm(total=steps) as progress:
                idx = 0
                for data in data_loader:
                    if len(data) == 3:
                        image, label, id = data
                        ids.append(id[0][0].split('.')[0])
                        if self.params['single_image_test'] is not None and not ids[idx] in self.params['single_image_test']:
                            idx += 1
                            continue
                        if noise_params:
                            noise = Noise(['Gaussian', 'Quantization'], bits=noise_params['bits'],
                                        sigma=noise_params['sigma'])
                            image = noise(image)

                        label_exists = True
                    else:
                        image, id = data
                    if image[0].dim() == 3:
                        image = [m.unsqueeze(0) for m in image]
                    image_to_model = list(map(lambda p: p.to(self.device), image))
                    label = label.to(self.device).squeeze()

                    debug_dir = os.path.join(self.params['debug_dir'], id[0][0].split('.')[0])
                    os.makedirs(debug_dir, exist_ok=True)

                    if idx == 0:    # Warm up
                        _ = self.model(image_to_model).squeeze()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    pred = self.model(image_to_model)
                    end.record()
                    torch.cuda.synchronize()
                    total_time += start.elapsed_time(end)
                    pred = pred.squeeze()

                    if label_exists:
                        metrics[idx] = calc_test_metrics(pred, label, self.params['test_metrics'])

                        if self.test_options['save_predictions']:
                            export_pred = pred.permute(1, 2, 0).cpu().numpy()
                            out_path = os.path.join(debug_dir, 'cube2D.mat')
                            hdf5storage.savemat(out_path, {'Cube': export_pred})
                            export_ref = label.permute(1, 2, 0).cpu().numpy()
                            out_path = os.path.join(debug_dir, 'cube2DRef.mat')
                            hdf5storage.savemat(out_path, {'Cube': export_ref})

                        if self.test_options['save_rgb']:
                            save_rgb_from_hs(label, save_dir=debug_dir, is_ref=True)
                            save_rgb_from_hs(pred, save_dir=debug_dir, is_ref=False)
                        if self.test_options['compare_rgb']:
                            show_rgb_and_metrics(pred, label, debug_dir, metrics[idx], self.params['wave_lengths'],
                                                    points=self.params['points'], show=self.display_on)
                        if self.test_options['spectral_graphs']:
                            show_spectral_graphs(pred, label, save_dir=debug_dir, normalize=True,
                                                    show=self.display_on, points=self.params['points'])
                        if self.test_options['psnr_map']:
                            show_psnr_map(pred, label, save_dir=debug_dir, show=self.display_on)
                        if self.test_options['color_spectral_channels']:
                            show_hs_predictions_results(pred, label, debug_dir, self.params['wave_lengths']
                                                        , show=self.display_on)

                    else:
                        if self.test_options['save_predictions']:
                            export_pred = pred.permute(1, 2, 0).cpu().numpy()
                            out_path = os.path.join(debug_dir, 'cube2D.mat')
                            hdf5storage.savemat(out_path, {'Cube': export_pred})

                        if self.test_options['save_rgb']:
                            save_rgb_from_hs(pred, save_dir=debug_dir, is_ref=False)

                    idx += 1
                    progress.update()
        self.logger.info('average inference time: {} ms'.format(total_time / idx))
        if self.params['single_image_test'] is None:
            plot_average_scores(metrics, ids, self.logger, save_dir=self.params['debug_dir'])
        return metrics

    def test(self):
        start = time.time()

        # Initialize data loader
        data_loader = HSDataLoader(self.params, logger=self.logger, mode='test')
        # Evaluate predictions and results
        scores = self.test_scheme(data_loader())
        self.logger.info('Time elapsed for testing: {}'.format(time.time() - start))

        return scores

    def noise_test(self, data_loader):

        sigma = [1e-8, 1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0,  1e1]
        bits = list(range(16, 0, -1))
        psnr_table = np.zeros((len(sigma), len(bits)))
        for i, sig in enumerate(sigma):
            for j, b in enumerate(bits):
                self.logger.info('Evaluation with sigma={}, bits={}'.format(self.sigma, self.bits))
                scores = self.test_scheme(data_loader(), bits=bits, sigma=sigma)
                psnr = 0
                for score in scores:
                    psnr += score['psnr']
                psnr /= len(scores)
                psnr_table[i, j] = psnr
        plt.imshow(psnr_table, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.xticks(np.arange(len(bits)), np.array(bits))
        plt.yticks(np.arange(len(sigma)), np.round(np.log10(np.array(sigma)), 2))
        plt.title('PSNR Map')
        plt.xlabel('bits')
        plt.ylabel('log_10(sigma)')

        plt.show()
        print(psnr_table)


