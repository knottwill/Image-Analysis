"""
Script to perform the coins segmentation (Module 1)

Usage:
python scripts/coins_segmentation.py --img ./data/coins.png --output_dir ./figures
"""

import sys
import os
import argparse
from os.path import join
from skimage import io, exposure, filters, filters, morphology, measure
import numpy as np 
import matplotlib.pyplot as plt
from skimage.feature import canny
from scipy.ndimage import binary_fill_holes

# add project path to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.custom_segmentation import otsu_threshold, find_regions, create_disk, binary_opening
from src.plotting import plot_image_mask_overlay

# Parse the arguments
parser = argparse.ArgumentParser(description='Segment the lungs in a CT image')
parser.add_argument('--img', default='./data/coins.png', type=str, help='Path to CT image')
parser.add_argument('--output_dir', default='./figures', type=str, help='Path to output directory')
args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)

# Load the image
image = io.imread(args.img, as_gray=True)
orig_image = image.copy()

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Median filter to remove noise
image = filters.median(image, np.ones((4,4)))

axes[0].imshow(image, cmap='gray')
axes[0].text(0, 1.05, '(a)', fontsize=25, transform=axes[0].transAxes, fontweight="bold")

# Find the edges using Canny
image = canny(image, sigma=3)

axes[1].imshow(image, cmap='gray')
axes[1].text(0, 1.05, '(b)', fontsize=25, transform=axes[1].transAxes, fontweight="bold")

# Fill the holes in the edges and remove small objects
image = binary_fill_holes(image)
mask = morphology.remove_small_objects(image)

axes[2].imshow(mask, cmap='gray')
axes[2].text(0, 1.05, '(c)', fontsize=25, transform=axes[2].transAxes, fontweight="bold")

# Find the centroids of the coins
label_image = measure.label(mask)
regions = measure.regionprops(label_image)
centroids = [region.centroid for region in regions]

axes[3].imshow(mask, cmap='gray')
for centroid in centroids:
    axes[3].plot(centroid[1], centroid[0], 'ro')
axes[3].text(0, 1.05, '(d)', fontsize=25, transform=axes[3].transAxes, fontweight="bold")

for ax in axes:
    ax.axis('off')

plt.tight_layout()
fig.savefig(join(args.output_dir, 'coins_steps.png'), bbox_inches='tight')

# Assume grid is 4x6
# Find the grid position of each coin
grid_positions = []
for centroid in centroids:
    grid_position = [0,0]
    for r in range(4):
        # Check if centroid is in the r'th row
        if centroid in sorted(centroids, key=lambda x: x[0])[6*r:6*r + 6]:
            grid_position[0] = r+1
    for c in range(6):
        # Check if centroid is in the c'th column
        if centroid in sorted(centroids, key=lambda x: x[1])[4*c:4*c + 4]:
            grid_position[1] = c+1

    grid_positions.append(grid_position)

# get segmentation of coin [1,1], [2,2], [3,3], [4,4]
diagonal_coin_labels = [grid_positions.index([i,i])+1 for i in range(1,5)]

# Get rid of all other coins in cleaned image
label_image = measure.label(mask)
regions = measure.regionprops(label_image)
for region in regions:
    if region.label not in diagonal_coin_labels:
        for coordinates in region.coords:
            mask[coordinates[0], coordinates[1]] = 0

fig, ax = plt.subplots(1, 1, figsize=(5, 5))

plot_image_mask_overlay(orig_image, mask, ax, dim_factor=0.3, cmap='gray', border_color='yellow')
ax.axis('off')

plt.tight_layout()
fig.savefig(join(args.output_dir, 'coins_segmentation.png'), bbox_inches='tight')