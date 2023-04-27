# Coefficient Diffusion for Electron Density Estimation

## Requirements

All codes are run with Python 3.9.15 and CUDA 11.6. Similar environment should also work, as this project does not rely on some rapidly changing packages. Other required packages are listed in `requirements.txt`.

## Data

The datasets used in this repo are freely available at the Quantum Machine [website](http://www.quantum-machine.org/datasets/). Specifically, **ethane and malonaldehyde** densities come from [here](https://arxiv.org/abs/1609.02815) by Brockherde et al., and **benzene, ethanol, phenol, and resorcinol** densities come from [here](https://www.nature.com/articles/s41467-020-19093-1) by Bogojeski et al. 

We assume the data is stored in the `<data_root>/<mol_name>/<mol_name>_<split>/` directory, where `mol_name` should be one of the molecules mentioned above and `split` should be either `train` or `test`. The directory should contain the following files:

- `structures.npy` contains the coordinates of the atoms.
- `dft_densities.npy` contains the voxelized electron charge density data.

This is the format for the latter four molecules (there are other files which you can safely ignore). For the former two molecules, run `python generate_dataset.py` to generate the correctly formatted data. You can also specify the data directory with `--root` and the output directory with `--out`.

All datasets assume a cubic box with side length of 20 Bohr and 50 grids per side. The densities are store as Fourier coefficients, and we will handle them properly in the `DensityDataset` class. 

## Training

To train a model, run `python train.py configs/diffusion.yml`. More specification of the arguments is listed below.

```bash
usage: main.py [-h] [--mode {train,inf}] [--device DEVICE] [--logdir LOGDIR] [--savename SAVENAME] [--resume RESUME] [--resume-predictor-only] config

positional arguments:
  config                config file

optional arguments:
  -h, --help            show this help message and exit
  --mode {train,inf}    running mode
  --device DEVICE       running device
  --logdir LOGDIR       log directory
  --savename SAVENAME   running save name
  --resume RESUME       resume from checkpoint
  --resume-predictor-only
                        only resume predictor but not diffnet
```

Training hyperparameters and model hypermeters can be found in the config YAML file. Most hyperparameters are self-explanatory. Note that the `model.diffusion` field will overwrite the original predictor hyperparameters (if provided). In this way, you can use a different architecture for the diffusion network. Feel free to change them to see how they affect the performance.