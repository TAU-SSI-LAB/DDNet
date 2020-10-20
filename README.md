# DDNet
Pytorch repository for Hyper-spectral image reconstruction from Dispersed and Diffused monochrome images.

## Usage

All configuration is set via python config file. A sample file with the paper publicaiton configuration for HS reconstruction from 2D diffuser DD image on ICVL cubes
is located at config.py.

  - To preform training with the default configuration on config.py:
  
    python main.py --train
  
  - To perform test with the default configuration:
  
    python main.py --test

### Training

python main.py --train --config <config_file.py>


### Testing

python main.py --test --config <config_file.py>

- For visualy display results on screen rather then only store them to drive:

  python main.py --test --config <config_file.py> --display
  
 
