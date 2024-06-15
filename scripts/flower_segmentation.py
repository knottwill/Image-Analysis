"""

"""

import os
from os.path import join
import sys
import argparse
from skimage import io, restoration, color, filters, morphology
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--img', type=str, default='./data/noisy_flower.jpg', help='Path to image')
parser.add_argument('--output_dir', type=str, default='./figures', help='Path to output directory')
args = parser.parse_args()

# Load image
image_path = './data/noisy_flower.jpg'
image = io.imread(image_path)

# RGBA -> RGB since alpha=255 everywhere. 
image = image[:,:,:3]