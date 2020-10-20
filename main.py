from ValidateParameters import validate_params
from Train import Training
from Test import Testing
import argparse
import importlib
import os
import logging
import sys


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=False, help='Path to .py configuration file')
    parser.add_argument('--train', action="store_true", required=False, help='Whether to apply Training or not')
    parser.add_argument('--test', action="store_true", required=False, help='Whether to apply Testing or not')
    parser.add_argument('--display', action="store_true", required=False, help='Whether to show visual objects on GUI')
    args = parser.parse_args()

    # Logging control
    if args.config is not None:
        log_path = args.config.split('.')[0] + '.log'
    else:
        log_path = '.log'

    logging.basicConfig(filename=log_path, filemode='a', format='%(asctime)s %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logger.addHandler(console)

    # Get parameter file from arguments or from default source
    if args.config is not None:
        assert args.config.split('.')[-1] == 'py', 'Config path must be a python script file'
        config_name = os.path.split(args.config)[-1].split('.')[0]
        params = importlib.import_module(config_name, args.config).Parameters().params
    else:
        logger.info('No config file inserted. Using default config file in {}'
                    .format(os.path.join('Config', 'config.py')))
        from Config import Parameters
        params = Parameters().params

    for k, v in params.items():
        logger.info('{}: {}'.format(k, v))

    # Parameters validation
    try:
        validate_params(params, logger, args.train, args.test)
    except AssertionError as e:
        logger.error('Parameters Error: {}'.format(e))
        sys.exit()

    # Train
    trained_model = None
    if args.train:

        trainer = Training(logger=logger, params=params)
        if args.display:
            trainer.display_on = True
            logger.info('Number of trainable parameters: {}'.format(sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)))
        trained_model, losses = trainer.train()

    # Test
    if args.test:

        if args.train:
            tester = Testing(logger=logger, params=params, train=True)
            tester.model = trained_model
            tester.model.eval()
        else:
            tester = Testing(logger=logger, params=params)
        if args.display:
            tester.display_on = True

        scores = tester.test()

    logger.info('Done!')

