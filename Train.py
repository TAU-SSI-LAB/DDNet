from TrainUtils import *
from Losses import Losses
from DataLoader import HSDataLoader
from Models import *
from DataTransforms import Noise

import torch


class Training(Losses):

    def __init__(self, logger=None, **args):

        super().__init__(**args)
        self.params = args['params']
        if logger is None:
            logging.basicConfig(format='%(asctime)s %(message)s')
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.architecture_params = self.params['architecture_params']
        self.get_model()
        self.model_weights_save_dir = self.params['model_weights_save_dir']
        self.model_name = self.params['model_name']
        self.losses = self.params['losses']
        self.metrics = self.params['metrics']
        self.train_steps = self.params['train_steps'] if 'train_steps' in self.params.keys() else None
        self.batch_size = self.params['batch_size']
        self.epochs = self.params['epochs']

        self.data_loaders = HSDataLoader(self.params, logger=self.logger, mode='train')

        self.checkpoints = self.params['checkpoints']
        self.display_on = False

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def get_model(self):

        self.model = self.params['model_type'](self.params['architecture_params'])
        if self.params['train_weights_path'] is not None:
            state_dict = torch.load(self.params['train_weights_path'])
            state_dict = fix_state_dict(state_dict)
            self.model.load_state_dict(state_dict)
        else:
            self.model.apply(self.weights_init)
        if torch.cuda.device_count() > 1:
            self.model = MultiGpuModel(self.model)
            self.logger.info('Using multi GPU mode with {} GPUS'.format(torch.cuda.device_count()))
        self.model = self.model.to(self.device)
        return

    def training_scheme(self, optimizer, scheduler, data_loaders, add_noise=False):
        noise = Noise(['Gaussian', 'Quantization'], sigma=1e-3, bits=10
                      )

        best_model_weights = copy.deepcopy(self.model.state_dict())
        best_loss = 1e10

        epochs_losses = defaultdict(list)

        if not len(data_loaders['val']):
            test_phase = 'train'
            test_loss = 'loss_train'
        else:
            test_phase = 'val'
            test_loss = 'loss_val'
        for epoch in range(1, self.epochs + 1):
            self.logger.info('Epoch %s/%s', epoch, self.epochs)

            since = time.time()

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                metrics = defaultdict(float)
                epoch_samples = 0

                steps = len(data_loaders[phase]) if self.train_steps is None or phase != 'train' else\
                    min(self.train_steps, len(data_loaders[phase]))
                if phase == 'val' and not steps:  # If no validation samples, skip validation phase.
                    continue

                with tqdm(total=steps) as progress:
                    step = 0
                    for inputs, labels, _ in data_loaders[phase]:
                        if step > steps and phase == 'train':
                            break
                        inputs = list(map(lambda p: p.to(self.device), inputs))

                        if phase == 'train' and add_noise:
                            inputs = noise(inputs)
                        labels = labels.to(self.device)
                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = self.model(inputs)
                            loss = self.calc_loss(outputs, labels, metrics)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        # statistics
                        epoch_samples += inputs[0].size(0)
                        progress.update()
                        step += 1
                if phase == 'train':
                    scheduler.step()
                    for param_group in optimizer.param_groups:
                        self.logger.info('LR: %s', param_group['lr'])

                epoch_losses = defaultdict(float)
                print_metrics(metrics, epoch_samples, phase, self.logger)

                for metric in metrics.keys():

                    epoch_losses[metric + '_' + phase] = metrics[metric] / epoch_samples
                    epochs_losses[metric + '_' + phase].append(metrics[metric] / epoch_samples)
                epochs_losses['loss' + '_' + phase].append(metrics['loss'] / epoch_samples)

                # deep copy the model
                if phase == test_phase and epoch_losses[test_loss] < best_loss:
                    self.logger.info("saving current model as best model")
                    best_loss = epoch_losses[test_loss]
                    best_model_weights = copy.deepcopy(self.model.state_dict())

                if self.checkpoints:
                    if phase == 'train' and not np.mod(epoch, self.params['checkpoints']):
                        temp_weights_file = os.path.join(self.model_weights_save_dir,
                                                         self.model_name + '_{epochs}_epochs_weights.pt'.format(
                                                              epochs=epoch))
                        self.logger.info('Saving checkpoints weights: %s', temp_weights_file)
                        torch.save(self.model.state_dict(), temp_weights_file)

            time_elapsed = time.time() - since
            self.logger.info('Elapsed time: %.0fm %.0fs', time_elapsed // 60, time_elapsed % 60)

        if self.display_on:
            show_loss_statistics(epochs_losses, valid=bool(len(data_loaders['val'])))

        self.logger.info('Best ' + test_phase + ' loss: {:4f}'.format(best_loss))

        # load best model weights
        self.model.load_state_dict(best_model_weights)

        return self.model, epochs_losses

    def train(self):

        # Set an optimizer
        optimizer_ft = self.params['optimizer'](self.model.parameters(), lr=self.params['learning_rate'])

        # Set learning rate decay scheduler
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer=optimizer_ft,
                                               step_size=self.params['learning_rate_decay_frequency'],
                                               gamma=self.params['learning_rate_decay_rate'])

        # Run training cycle
        model, losses = self.training_scheme(optimizer_ft, exp_lr_scheduler, self.data_loaders())

        # Save model
        final_weights_path = os.path.join(self.model_weights_save_dir, self.model_name + '.pt')
        self.logger.info('Saving final weights: %s', final_weights_path)
        torch.save(model.state_dict(), final_weights_path)

        return model, losses




