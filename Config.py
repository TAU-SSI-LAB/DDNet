from Models import *


class Parameters(object):

    def __init__(self):
        self.params = dict()

        """ Hyper Parameters """
        self.params['learning_rate'] = 1e-4
        self.params['optimizer'] = Adam
        self.params['learning_rate_decay_frequency'] = 8    # Reduce learning rate every _ epochs
        self.params['learning_rate_decay_rate'] = 0.5       # Reduce learning rate by this factor
        self.params['batch_size'] = 4
        self.params['epochs'] = 40
        self.params['train_steps'] = None       # How many training iterations to perform

        """ Model Weights Parameters """
        self.params['checkpoints'] = 4      # Save model checkpoint frequency (in epochs)
        self.params['train_weights_path'] = None        # Pretrained weights
        self.params['model_weights_save_dir'] = '/media/cs-dl/HD_6TB/Data/Trained_Models/pytorch/2D_ICVL/'

        """ Model Parameters """
        self.params['model_type'] = DDNet

        # Which replicas to extract from the original DD image and in which order
        self.params['mono_replicas_location'] = [['center', 'right', 'left'], ['center', 'up', 'down'],
                                                 ['center', 'right', 'left', 'up', 'down']]

        self.params['min_wavelength'] = 420
        self.params['max_wavelength'] = 700
        self.params['spectral_band_width'] = 10

        # Don't touch these 2 lines -->
        self.params['wave_lengths'] = range(self.params['min_wavelength'], self.params['max_wavelength'] +
                                            self.params['spectral_band_width'], self.params['spectral_band_width'])
        inputs_channels_list = [len(p) for p in self.params['mono_replicas_location']]


        self.params['architecture_params'] = {'num_levels': 3,  # Number of U-Net levels.
                                              'inputs_channels_list': inputs_channels_list,
                                              'base_num_filters': 64,   # First layer output filters
                                              # Kernel sizes for each of the inputs
                                              'kernels_list': [(1, 9), (9, 1), (3, 3)],
                                              'bottleneck_layers': 4,       # How many cnn layers in bottleneck
                                              'out_channels': len(self.params['wave_lengths']),
                                              'attention': True, 'attention_heads': 8}

        self.params['model_name'] = '2D_ICVL'

        """ Data Parameters """

        self.params['training_data_sets'] = [       # (DD images dir, HS dir)
            ('/media/cs-dl/HD_6TB/Data/ICVL/2D/TrainData/Vanilla/Sim/',
             '/home/cs-dl/Data_local/ICVL/TrainData/Cubes/Vanilla/HS/'),
            ('/media/cs-dl/HD_6TB/Data/ICVL/2D/TrainData/augmented/Sim/',
             '/home/cs-dl/Data_local/ICVL/TrainData/Cubes/augmented/HS/'),
            ('/media/cs-dl/HD_6TB/Data/ICVL/2D/TrainData/augmented_noLightening/Sim/',
             '/home/cs-dl/Data_local/ICVL/TrainData/Cubes/augmented_noLightening/HS/'),
                                             ]

        # Insert validation data sets or keep 'None' to use portion of train set
        self.params['validation_data_sets'] = None
        self.params['train_val_ratio'] = 0.97
        self.params['train_data_percentage'] = 1

        self.params['mono_replicas'] = list(map(lambda p: len(p), self.params['mono_replicas_location']))

        self.params['crop_center_coord_x'] = 1168      # Where to crop the main replica from the original DD image.
        self.params['crop_center_coord_y'] = 844
        self.params['crop_size'] = (256, 256)
        self.params['hs_normalize_factor'] = 1
        self.params['mono_normalize_factor'] = 45

        self.params['debug_dir'] = '/home/cs-dl/Results/2d_ICVL'

        """ Loss Parameters """
        self.params['losses'] = {'rmse': 0.5, 'rgb': 0.2, 'ssim': 0.1, 'spectral': 0.1, 'blur': 0.1}
        self.params['metrics'] = ['psnr']
        """ Test Parameters """
        self.params['test_data_sets'] = [('/media/cs-dl/HD_6TB/Data/ICVL/2D/TestData/Hand_cropped/Sim',
                                          '/media/cs-dl/HD_6TB/Data/ICVL/2D/TestData/Hand_cropped/HS')]
        # Model test weights
        self.params['test_weights_path'] = '/media/cs-dl/HD_6TB/Data/Trained_Models/pytorch/2D_ICVL/2D_ICVL.pt'

        self.params['test_metrics'] = ['psnr', 'rmse', 'ssim', 'sam']
        # Which experiments to perform
        self.params['test_options'] = {'save_predictions': False, 'save_rgb': False, 'compare_rgb': True,
                                       'spectral_graphs': False, 'psnr_map': False, 'color_spectral_channels': False}
        # If different than 'None', only this image will be tested (use the image name without suffix)
        self.params['single_image_test'] = None
        # specific points to analyze on test
        self.params['points'] = [(125, 30), (20, 170)]



