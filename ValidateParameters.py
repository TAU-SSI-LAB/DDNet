from Imports import *


def validate_params(params, logger, train=True, test=False):

    def validate_train_params():

        assert isinstance(params['learning_rate'], float), 'Learning rate must be a float'
        assert issubclass(params['optimizer'], torch.optim.Optimizer), \
            'Optimizer must be a torch.nn.Optimizer sub class'
        assert isinstance(params['learning_rate_decay_frequency'], int), \
            'learning rate decay frequency must be an int'
        assert isinstance(params['learning_rate_decay_rate'], float) and \
               1 >= params['learning_rate_decay_rate'] >= 0, 'Learning rate decay rate must be a float between 0 to 1'
        assert isinstance(params['batch_size'], int), 'Batch size must be an int'

        if 'train_steps' in params.keys():
            assert isinstance(params['train_steps'], int) or params['train_steps'] is None, \
                'train steps must be an int or None'
        else:
            logger.info('train steps not defined, setting train steps to default None(Iterate over all data batches)')
        assert isinstance(params['checkpoints'], int), 'checkpoints must be an int'

        if 'train_weights_path' in params.keys():
            assert isinstance(params['train_weights_path'], str) or params['train_weights_path'] is None, \
                'train weights path must be a string or None'
            if params['train_weights_path'] is not None:
                assert os.path.isfile(params['train_weights_path']), \
                    'weights file {} is not exist.'.format(params['train_weights_path'])
        else:
            logger.info('train weights path not defined,'
                        ' setting train weights to default None (Train model from scratch)')
        assert isinstance(params['model_weights_save_dir'], str), 'model weights save dir must be an int'
        assert os.path.isdir(params['model_weights_save_dir']), \
            'model weights save dir: {} is not a directory'.format(params['model_weights_save_dir'])
        assert issubclass(params['model_type'], torch.nn.Module), 'model type must be a torch.nn.Module sub class'
        assert isinstance(params['architecture_params'], dict) or params['architecture_params'] is None,\
            'architecture_params must be a dictionary'
        assert isinstance(params['model_name'], str), 'model name must be a string'

        assert isinstance(params['training_data_sets'], list), 'training data sets must be a list of strings'
        for data_set in params['training_data_sets']:
            assert isinstance(data_set, tuple), 'training data sets must be a list of tuples'
            assert len(data_set) == 2, 'Each training data set must be a tuple of (image_dir, label_dir)'
            assert os.path.isdir(data_set[0]) and os.path.isdir(data_set[1]), \
                'train data set directories: {} is not a directories'.format(data_set)

        if 'validation_data_sets' in params.keys() and params['validation_data_sets'] is not None:
            assert isinstance(params['validation_data_sets'], list) \
                , 'validation data sets must be a list of strings or None'
            for data_set in params['validation_data_sets']:
                assert isinstance(data_set, tuple), 'validation data sets must be a list of tuple or None'
                assert len(data_set) == 2, 'Each validation data set must be a tuple of (image_dir, label_dir)'
                assert os.path.isdir(data_set[0]) and os.path.isdir(data_set[1]), \
                    'Validation data set directories: {} is not a directories'.format(data_set)
        else:
            logger.info('Validation data sets are not defined,'
                        ' setting validation sets to default None (Take validation set from'
                        ' training data with train_val_ratio)')

        assert isinstance(params['train_val_ratio'], float) or params['train_val_ratio'] == 1 and \
               1 >= params['train_val_ratio'] >= 0, 'train val ratio must be a float between 0 to 1'

        assert isinstance(params['train_data_percentage'], float) or params['train_data_percentage'] == 1 \
               and 1 >= params['train_data_percentage'] >= 0, 'train data percentage must be float between 0 to 1'

        assert isinstance(params['losses'], dict), 'losses must be a dictionary'
        for key, value in params['losses'].items():
            assert isinstance(key, str), 'losses keys must be a strings'
            assert isinstance(value, float) or value == 0, 'losses values must be a floats or zero'

    # Train Parameters Validation
    if train:
        validate_train_params()

    # Test Parameters Validation
    if test:

        if not train:
            assert isinstance(params['test_weights_path'], str), 'test weights path must be a string'
            assert os.path.isfile(params['test_weights_path']), \
                'test weights file {} is not exist.'.format(params['test_weights_path'])
            assert issubclass(params['model_type'], torch.nn.Module), 'model type must be a torch.nn.Module sub class'

        assert isinstance(params['test_data_sets'], list), 'test sets dirs must be a list'
        for data_set in params['test_data_sets']:
            assert isinstance(data_set, tuple), 'Test data sets must be a list of tuple or None'
            assert len(data_set) == 2, 'Each test data set must be a tuple of (image_dir, label_dir)'
            assert os.path.isdir(data_set[0]) and os.path.isdir(data_set[1]), \
                'Test data set directories: {} is not a directories'.format(data_set)
