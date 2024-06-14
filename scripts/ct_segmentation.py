"""
Script to segment the lungs in a CT image.

Usage:
python ./scripts/ct_segmentation.py --img ./data/CT.png --output_dir ./figures
"""

import sys
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from os.path import join

# add project path to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.custom_segmentation import otsu_threshold, find_regions, create_disk, binary_opening
from src.plotting import plot_image_mask_overlay

# Parse the arguments
parser = argparse.ArgumentParser(description='Segment the lungs in a CT image')
parser.add_argument('--img', default='./data/CT.png', type=str, help='Path to CT image')
parser.add_argument('--output_dir', default='./figures', type=str, help='Path to output directory')
args = parser.parse_args()

# Load the image
image_path = args.img
image = plt.imread(image_path)
image = np.mean(image, axis=-1)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Otsu's thresholding
threshold = otsu_threshold(image)
binary_image = image > threshold
axes[0].imshow(binary_image, cmap='gray')
axes[0].text(0, 1.05, '(a)', fontsize=25, transform=axes[0].transAxes, fontweight="bold")

# Find foreground regions
labelled_image, _ = find_regions(binary_image, forground_only=True)

# Remove small regions
sizes = np.bincount(labelled_image.flatten())

mask_sizes = sizes >= 500
mask_sizes[0] = 0  # Background should not be counted

img = mask_sizes[labelled_image]

axes[1].imshow(img, cmap='gray')
axes[1].text(0, 1.05, '(b)', fontsize=25, transform=axes[1].transAxes, fontweight="bold")

# Binary opening
selem = create_disk(3)
img = binary_opening(img, selem)
axes[2].imshow(img, cmap='gray')
axes[2].text(0, 1.05, '(c)', fontsize=25, transform=axes[2].transAxes, fontweight="bold")

# Find lung regions
img, n_regions = find_regions(1 - img, forground_only=True)
regions = list(range(1, n_regions+1))

# remove background label from regions
background_label = img[0, 0]
regions.remove(background_label)

# get two largest remaining regions (lungs)
if len(regions) > 2:
    regions.sort(key=lambda x: -np.sum(img == x))
    regions = regions[:2]
        
# create segmentation mask of the two lungs
mask = np.zeros_like(img)
for region in regions:
    mask[img == region] = 1

axes[3].imshow(mask, cmap='gray')
axes[3].text(0, 1.05, '(d)', fontsize=25, transform=axes[3].transAxes, fontweight="bold")

for ax in axes:
    ax.axis('off')

fig.savefig(join(args.output_dir, 'CT_steps.png'))

# Plot the original image and the segmentation mask overlay
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(image, cmap='gray')
axes[0].text(0, 1.05, '(a)', fontsize=20, transform=axes[0].transAxes, fontweight="bold")

plot_image_mask_overlay(image, mask, axes[1], dim_factor=0.3, cmap='gray', border_color='yellow')
axes[1].text(0, 1.05, '(b)', fontsize=20, transform=axes[1].transAxes, fontweight="bold")

for ax in axes:
    ax.axis('off')

fig.savefig(join(args.output_dir, 'CT_segmentation.png'))