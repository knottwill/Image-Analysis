# Image analysis

<a href="https://hub.docker.com/r/milesial/unet"><img src="https://img.shields.io/badge/docker%20image-available-blue?logo=Docker&style=for-the-badge" /></a>

## Description

This project solves various problems relating to image analysis, as described in `ImageAnalysis.pdf`. The analysis report can be found in `report/`. 

Topics:
- Image segmentation
- Inverse problems
- Compressed-sensing
- Wavelet compression
- Optimization
- CT reconstruction (TV-regularization, learned gradient descent)

<b>Project structure</b>

```bash
├── report/    # contains report
├── data/      # contains provided data
├── figures/   # directory for storing plots used in the report
├── scripts/   # directory containing all scripts for reproducing results in report
├── src/       # package containing re-usable components used in the scripts
|
├── .gitignore         # specifies untracked files to ignore
├── ImageAnalysis.pdf  # coursework assignment
├── Dockerfile         # containerisation instructions
├── LICENSE            # license for project
├── README.md          # this file
├── environment.yml    # environment specifications
├── env_light.yml      # lightweight environment (not including odl)
├── env_very_light.yml # environment not including odl, pywavelets OR version specs
```

## Usuage

#### Re-creating environment

Attempt to re-create the full environment for the project:

```bash
# Try re-create exact conda environment
$ conda env create -f environment.yml -n <env-name>
```

If this fails due to conflicts relating to the `odl` package, you will have to re-create the smaller environment (not containg `odl`), and the CT reconstruction part of the project can be done using `CT_reconstruction.ipynb` in google colab (More details below). The smaller environment can be re-created using one of the following:

```bash
# Option 1: Conda
$ conda env create -f env_light.yml -n <env-name>

# If env_light didnt work, try:
$ conda env create -f env_very_light.yml -n <env-name>
$ conda activate <env-name>
$ pip install pywavelets

# Option 2: Generate docker image and run container
$ docker build -t <image_name> .
$ docker run -ti <image_name>
```

#### Reproducing all results EXCEPT problem 3.2

To re-produce all results/figures presented in the report, use the following commands

```bash
# 1.1 CT segmentation
$ python ./scripts/ct_segmentation.py --img ./data/CT.png --output_dir ./figures

# 1.2 Flowers segmentation
$ python scripts/flower_segmentation.py --img ./data/noisy_flower.jpg --output_dir ./figures

# 1.3 Coins segementation
$ python scripts/coins_segmentation.py --img ./data/coins.png --output_dir ./figures

# 2.1 Line fitting
$ python scripts/line_fitting.py --y_line ./data/y_line.txt --y_outlier_line ./data/y_outlier_line.txt --output ./figures/line_fits.png

# 2.2 Compressed sensing
$ python scripts/compressed_sensing.py

# 2.3 Wavelet compression
$ python ./scripts/wavelet_compression.py --img ./data/river_side.jpeg --output_dir ./figures

# 3.1 Gradient descent
$ python ./scripts/gradient_descent.py
```

In each case, plots will be saved to `--output_dir` and other results will be printed to the terminal.

#### Reproducing problem 3.2

If you were able to reproduce the full environment from `environment.yml`, then run the script below. Also try running the script if you happen to have an environment with the relevant packages installed.

```bash
$ python scripts/ct_reconstruction.py --output_dir ./figures
```

Otherwise, use the notebook `CT_reconstruction.ipynb` in google colab. 

### Timing

I ran all scripts on my personal laptop with the following specifications:
- Chip:	Apple M1 Pro
- Total Number of Cores: 8 (6 performance and 2 efficiency)
- Memory (RAM): 16 GB
- Operating System: macOS Sonoma v14.0

The `ct_reconstruction.py` script took ~20 minutes to run. All other scripts ran in less than 2 minutes.