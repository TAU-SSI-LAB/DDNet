from DataUtils import *
from DataTransforms import *


class HSDataSet(Dataset):

    def __init__(self, ids, params, logger, mode='test'):

        self.ids = ids
        self.mode = mode
        self.width = None
        self.height = None
        self.logger = logger
        self.batch_size = params['batch_size']
        self.mono_replicas_location = params['mono_replicas_location']
        self.mono_replicas = list(map(lambda p: len(p), self.mono_replicas_location))
        self.crop_center_coord_x = params['crop_center_coord_x']
        self.crop_center_coord_y = params['crop_center_coord_y']
        self.crop_size_x, self.crop_size_y = params['crop_size']

        self.crop_center_coord_x = params['crop_center_coord_x']
        self.crop_center_coord_y = params['crop_center_coord_y']

        self.hs_normalize_factor = params['hs_normalize_factor']
        self.mono_normalize_factor = params['mono_normalize_factor']

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx, add_noise=False):

        _id = self.ids[idx]
        mono, hs = self.data_generator(_id)

        for mono_replica in mono:

            assert mono_replica.shape[:1] == hs.shape[:1]

        mono = list(map(lambda p: np.transpose(p, (2, 0, 1)), mono))
        mono = list(map(lambda p: torch.tensor(p, dtype=torch.float32), mono))
        hs = np.transpose(hs, (2, 0, 1))
        hs = torch.tensor(hs, dtype=torch.float32)

        if add_noise:
            if random.random() < 0.2:
                noise = Noise(['Gaussian', 'Quantization'], sigma=1e-3, bits=10)
                mono = noise(mono)
        return [mono, hs, _id]

    def data_generator(self, mono_hs_pair):

        _id, mono_dir, hs_dir = mono_hs_pair
        hs_id = _id.split('.')[0] + '.mat'
        if 'Im_Mono' in hs_id:
            hs_id = hs_id.replace('Im_Mono', 'RefCube')
        mono_path = os.path.join(mono_dir, _id)
        label_path = os.path.join(hs_dir, hs_id)
        # Fix Hs extension if needed.
        if label_path.split('.')[-1] == 'mat':
            label_path_2 = label_path.split('.')[:-1][0]
        else:
            label_path_2 = label_path + '.mat'
        if mono_path.split('.')[-1] == 'mat':
            mono_path_2 = mono_path.split('.')[:-1][0]
        else:
            mono_path_2 = mono_path + '.mat'
        # Mono image loading.
        if not os.path.isfile(mono_path):
            mono_path = mono_path_2
            if not os.path.isfile(mono_path):
                self.logger.error("Can't open %s", mono_path)
                raise FileNotFoundError
        try:
            img = cv2.imread(mono_path, cv2.IMREAD_ANYDEPTH)
        except Exception as e:
            self.logger.error('{} - {}'.format(mono_path, e))
        # Hyper spectral label image loading.
        if not os.path.isfile(label_path):
            label_path = label_path_2
            if not os.path.isfile(label_path):
                self.logger.error("Can't open %s", label_path)
                raise FileNotFoundError
        try:
            hs = hdf5storage.loadmat(label_path)['RefCube'][..., 2:]
            assert hs.ndim == 3
        except Exception as e:
            self.logger.error('{} - {}'.format(label_path, e))

        mono_list = self.concat_input_image(img)
        # Data augmentation
        if self.mode == 'train' and np.random.random() < 0.5:
            mono_list, hs = list(map(lambda p: p[:, ::-1, :], mono_list)), hs[:, ::-1,
                                                                                  :]  # random horizontal flip
        if self.mode == 'train' and np.random.random() < 0.5:
            mono_list, hs = list(map(lambda p: p[::-1, :, :], mono_list)), hs[::-1, :,
                                                                                  :]  # random vertical flip
        mono_list = list(map(lambda p: np.array(p), mono_list))

        hs = np.array(hs, np.float32)
        mono_list, hs = self.normalize_batch(mono_list, hs)

        return mono_list, hs

    def concat_input_image(self, img):
        mono_inputs = [None] * len(self.mono_replicas)

        center_range_x = slice(self.crop_center_coord_x, self.crop_center_coord_x + self.crop_size_x)
        right_range_x = slice(self.crop_center_coord_x + self.crop_size_x, self.crop_center_coord_x + 2 * self.crop_size_x)
        right_right_range_x = slice(self.crop_center_coord_x + 2 * self.crop_size_x, self.crop_center_coord_x + 3 * self.crop_size_x)
        left_range_x = slice(self.crop_center_coord_x - self.crop_size_x, self.crop_center_coord_x)
        left_left_range_x = slice(self.crop_center_coord_x - 2 * self.crop_size_x, self.crop_center_coord_x - self.crop_size_x)
        center_range_y = slice(self.crop_center_coord_y, self.crop_center_coord_y + self.crop_size_y)
        up_range_y = slice(self.crop_center_coord_y - self.crop_size_y, self.crop_center_coord_y)
        down_range_y = slice(self.crop_center_coord_y + self.crop_size_y, self.crop_center_coord_y + 2 * self.crop_size_y)

        for i, n_replicas in enumerate(self.mono_replicas):
            mono_inputs[i] = np.zeros((self.crop_size_y, self.crop_size_x, n_replicas))

            for idx, loc in enumerate(self.mono_replicas_location[i]):

                if loc == 'center':
                    mono_inputs[i][..., idx] = img[center_range_y, center_range_x]
                    continue
                elif loc == 'right':
                    mono_inputs[i][..., idx] = img[center_range_y, right_range_x]
                    continue
                elif loc == 'left':
                    mono_inputs[i][..., idx] = img[center_range_y, left_range_x]
                    continue
                elif loc == 'up':
                    mono_inputs[i][..., idx] = img[up_range_y, center_range_x]
                    continue
                elif loc == 'down':
                    mono_inputs[i][..., idx] = img[down_range_y, center_range_x]
                    continue
                elif loc == 'left-left':
                    mono_inputs[i][..., idx] = img[center_range_y, left_left_range_x]
                    continue
                elif loc == 'right-right':
                    mono_inputs[i][..., idx] = img[center_range_y, right_right_range_x]
                    continue
                elif loc == 'up-right':
                    mono_inputs[i][..., idx] = img[up_range_y, right_range_x]
                    continue
                elif loc == 'down-right':
                    mono_inputs[i][..., idx] = img[down_range_y, right_range_x]
                    continue
                elif loc == 'down-left':
                    mono_inputs[i][..., idx] = img[down_range_y, left_range_x]
                    continue
                elif loc == 'up-left':
                    mono_inputs[i][..., idx] = img[up_range_y, left_range_x]
                    continue

        return mono_inputs

    def normalize_batch(self, x_batch, y_batch):
        y_batch = np.divide(y_batch, self.hs_normalize_factor)
        y_batch = np.clip(y_batch, 0, 1)

        if y_batch.ndim > 4:
            y_batch = np.squeeze(y_batch, axis=0)

        x_batch = list(map(lambda p: np.divide(p, self.mono_normalize_factor), x_batch))
        return x_batch, y_batch


class HSDataLoader:

    def __init__(self, params, logger, mode='train'):

        self.params = params
        self.validation_sets = self.params['validation_data_sets'] if 'validation_data_sets' in self.params.keys() \
            else None
        self.train_val_ratio = self.params['train_val_ratio'] if self.validation_sets is None else 1.0
        self.mode = mode
        self.batch_size = self.params['batch_size'] if mode == 'train' else 1
        self.batches_per_gpu = int(self.batch_size / (torch.cuda.device_count() if torch.cuda.is_available() else 1))
        # Set number of workers to be number of batches processed per GPU.
        self.workers = self.batches_per_gpu

        self.logger = logger

    def __call__(self):

        if self.mode == 'train':

            # Create list of images and labels for training and validation.
            self.train_list, self.val_list, self.train_dirs_sizes = \
                get_train_val_images_list(train_sets=self.params['training_data_sets'],
                                          validation_sets=self.validation_sets,
                                          train_val_ratio=self.train_val_ratio,
                                          data_percentage=self.params['train_data_percentage'])
            # Get train and validation data sets.

            train_set = HSDataSet(self.train_list,  self.params, logger=self.logger, mode='train')

            val_set = HSDataSet(self.val_list, self.params, logger=self.logger, mode='test')

            # Create data loaders with the data sets and sampler.
            data_loaders = {
                'train': DataLoader(train_set,
                                    num_workers=self.workers,
                                    worker_init_fn=worker_init_fn,
                                    pin_memory=True,
                                    shuffle=True,
                                    batch_size=self.batch_size),
                'val': DataLoader(val_set,
                                  batch_size=self.batch_size,
                                  shuffle=False,
                                  num_workers=self.workers,
                                  worker_init_fn=worker_init_fn,
                                  pin_memory=True)
            }
            return data_loaders

        elif self.mode == 'test':

            # Create list of images and labels for testing.
            self.test_list = get_test_images_list(test_sets=self.params['test_data_sets'])
            # Create test data set with the test list images.
            test_set = HSDataSet(self.test_list, self.params,
                                           logger=self.logger, mode='test')
            return DataLoader(test_set,
                              batch_size=self.batch_size,
                              shuffle=False,
                              num_workers=self.workers,
                              worker_init_fn=worker_init_fn,
                              pin_memory=True)

