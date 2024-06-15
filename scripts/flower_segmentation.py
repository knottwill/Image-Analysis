"""
Script to segment the purple flowers in the image using KMeans clustering and colour thresholding.

Usage:
python scripts/flower_segmentation.py --img ./data/noisy_flower.jpg --output_dir ./figures
"""

import os
from os.path import join
import sys
import argparse
from skimage import io, restoration, color, filters, morphology
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# add project path to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.plotting import plot_image_mask_overlay

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--img', type=str, default='./data/noisy_flower.jpg', help='Path to image')
parser.add_argument('--output_dir', type=str, default='./figures', help='Path to output directory')
args = parser.parse_args('--img ./data/noisy_flower.jpg'.split(' '))

# Load image
image_path = args.img
image = io.imread(image_path)

# RGBA -> RGB since alpha=255 everywhere. 
image = image[:,:,:3]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# gaussian blur denoising
denoised_image = restoration.denoise_bilateral(image, win_size=8, sigma_color=0.15, sigma_spatial=15, channel_axis=-1)

axes[0].imshow(denoised_image)
axes[0].text(0, 1.05, '(a)', fontsize=20, transform=axes[0].transAxes, fontweight="bold")

################
# Method 1: KMeans clustering
################

# Convert image to Lab and remove the lightness channel
image_lab = color.rgb2lab(denoised_image)
image_ab = image_lab[:,:,1:]
pixels = image_ab.reshape((-1, 2))

# Fit KMeans clustering
kmeans = KMeans(n_clusters=4)
kmeans.fit(pixels)

# Get the cluster assignments and cluster centers
cluster_assignments = kmeans.labels_
cluster_centers = kmeans.cluster_centers_
cluster_image = cluster_assignments.reshape(image.shape[:2])

# Lab value for purple (source: https://www.e-paint.co.uk/lab-hlc-rgb-lrv-values.asp?cRange=BS+5252&cRef=02+D+45&cDescription=Purple)
purple = [34.64, 34.45, -4.07]
purple = purple[1:] # remove lightness channel

assert len(purple) == cluster_centers.shape[1], "Color vector must have the same number of dimensions as the cluster centers"

# Find the cluster center that is closest to purple
closest_cluster_index = np.argmin(np.linalg.norm(cluster_centers - purple, axis=1))
km_mask = cluster_image == closest_cluster_index

axes[1].imshow(km_mask, cmap='gray')
axes[1].text(0, 1.05, '(b)', fontsize=20, transform=axes[1].transAxes, fontweight="bold")

# Morphological operations for cleaning
km_mask = morphology.remove_small_objects(km_mask, 200) # remove small objects
km_mask = morphology.remove_small_holes(km_mask, 100) # remove small holes

# binary opening
selem = morphology.disk(2)
km_mask = morphology.binary_opening(km_mask, selem)

axes[2].imshow(km_mask, cmap='gray')
axes[2].text(0, 1.05, '(c)', fontsize=20, transform=axes[2].transAxes, fontweight="bold")

for ax in axes:
    ax.axis('off')

################
# Method 2: Colour thresholding
################

# Convert image to HSV and threshold based on hue
image_hsv = color.rgb2hsv(denoised_image)
purple_hue_range = [0.7, 0.96]
cf_mask = (image_hsv[..., 0] > purple_hue_range[0]) & (image_hsv[..., 0] < purple_hue_range[1])

cf_mask = morphology.remove_small_objects(cf_mask, 200)
cf_mask = morphology.remove_small_holes(cf_mask, 100)

# binary opening
selem = morphology.disk(2)
cf_mask = morphology.binary_opening(cf_mask, selem)

fig.savefig(join(args.output_dir, 'flower_steps.png'))

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# axes[0].imshow(image, cmap='gray')
plot_image_mask_overlay(image, km_mask, axes[0], dim_factor=0.15, cmap='jet', border_color='yellow', border_width=0.7)
# axes[0].imshow(km_mask, alpha=0.2, cmap='jet')
axes[0].text(0, 1.05, '(a) KMeans', fontsize=12, transform=axes[0].transAxes, fontweight="bold")

plot_image_mask_overlay(image, cf_mask, axes[1], dim_factor=0.15, cmap='jet', border_color='yellow', border_width=0.7)
axes[1].text(0, 1.05, '(b) Colour Threshold', fontsize=12, transform=axes[1].transAxes, fontweight="bold")

for ax in axes:
    ax.axis('off')

plt.tight_layout()
fig.savefig(join(args.output_dir, 'flower_segmentation.png'))