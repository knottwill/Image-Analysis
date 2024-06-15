"""
Script to solve the wavelet compression problem (Exercise 2.3) of coursework.

Usage:
python ./scripts/wavelet_compression.py --img ./data/river_side.jpeg --output_dir ./figures
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os
from os.path import join
import pywt
import argparse
import pandas as pd

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--img', default='./data/river_side.jpeg', type=str, help='Path to river_side.jpeg')
parser.add_argument('--output_dir', default='./figures', type=str, help='Path to output directory')
args = parser.parse_args()

# Load the image
A = io.imread(args.img)
image = np.mean(A, -1); # Convert RGB to grayscale

# Crop out the white image borders
mask = image < 225 # (Find the white regions)
coords = np.argwhere(mask)
y0, x0 = coords.min(axis=0) 
y1, x1 = coords.max(axis=0) + 1
image = image[y0:y1, x0:x1]

# Parameters for wavelet decomposition
n = 4 # Number of levels
mode='periodization'
w = 'db4'

##################
# Wavelet transform and non-compressed reconstruction
##################

coeffs = pywt.wavedec2(image,wavelet=w,mode=mode,level=n)

# normalize the coefficient arrays
coeffs[0] /= np.abs(coeffs[0]).max()
for detail in range(n):
    coeffs[detail + 1] = [d/np.abs(d).max() for d in coeffs[detail + 1]]

coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)

# Plot the coefficients and highlight the largest 15%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(coeff_arr, cmap='gray',vmin=-0.4,vmax=0.75)
ax.axis('off')
fig.savefig(join(args.output_dir, 'e2p3_transform.png'), dpi=300)

coeffs = pywt.wavedec2(image,wavelet=w,mode=mode,level=n)

# Reconstruction
recon = pywt.waverec2(coeffs,wavelet=w,mode=mode)

abs_diff = np.abs(image-recon)

# Calculate MSE, PSNR, and SSIM
vmin = min(image.min(),recon.min())
vmax = max(image.max(),recon.max())

mse = np.mean(abs_diff**2)
psnr = peak_signal_noise_ratio(image, recon, data_range=vmax-vmin)
ssim = structural_similarity(image, recon, data_range=vmax-vmin)
print(f'Un-compressed reconstruction PSNR: {psnr:.5f}')
print(f'Un-compressed reconstruction SSIM: {ssim:.5f}')

# Show original, reconstruction, and difference
fig, axes = plt.subplots(1,3,figsize=(8,8))
axes[0].imshow(image,cmap='gray',vmin=vmin,vmax=vmax)
axes[0].text(0., 1.05, '(a) Original', fontsize=12, transform=axes[0].transAxes, fontweight="bold")

axes[1].imshow(recon,cmap='gray',vmin=vmin,vmax=vmax)
axes[1].text(0., 1.05, '(b) Reconstructed', fontsize=12, transform=axes[1].transAxes, fontweight="bold")

axes[2].imshow(abs_diff,cmap='gray')
axes[2].text(0., 1.05, '(c) Difference (abs)', fontsize=12, transform=axes[2].transAxes, fontweight="bold")

for ax in axes:
    ax.axis('off')

plt.tight_layout()

fig.savefig(join(args.output_dir, 'e2p3_uncompressed.png'))

coeff_sort = np.array(sorted(np.abs(coeff_arr).flatten(), reverse=True))

# plot the sorted coefficients and log of the sorted coefficients
fig, axes = plt.subplots(1,2,figsize=(12,6))

axes[0].plot(coeff_sort)
axes[0].text(0.1, 0.9, '(a)', fontsize=15, transform=axes[0].transAxes, fontweight="bold")
axes[0].set_ylabel(r'Coefficient magnitude $|\alpha_i|$', fontsize=15)
axes[0].set_xlabel(r'Index $i$', fontsize=15)
axes[0].grid()

log_coeff = np.log(coeff_sort[coeff_sort > 0.00001])
axes[1].plot(log_coeff)
axes[1].text(0.1, 0.9, '(b)', fontsize=15, transform=axes[1].transAxes, fontweight="bold")
axes[1].set_ylabel(r'$\log|\alpha_i|$', fontsize=15)
axes[1].set_xlabel(r'Index $i$', fontsize=15)
axes[1].grid()

fig.savefig(join(args.output_dir, 'e2p3_coefficients.png'))

##################
# Threshold to retain largest 15% of coefficients
##################

## Wavelet Compression
coeffs = pywt.wavedec2(image,wavelet=w,mode=mode,level=n)

def threshold_coefficients(coeffs, keep):
    """
    Threshold the wavelet coefficients by keeping the largest `keep` fraction of coefficients.
    """
    coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs) # Convert to array
    coeff_sort = np.sort(np.abs(coeff_arr.reshape(-1))) # Sort the coefficients
    thresh_idx = int(np.floor((1 - keep) * len(coeff_sort))) # Find the threshold index
    thresh = coeff_sort[thresh_idx] # Find the threshold
    idx = np.abs(coeff_arr) > thresh # Find the indices of the large coefficients
    coeff_thresh = coeff_arr * idx # Threshold small indices
    coeff_thresh = pywt.array_to_coeffs(coeff_thresh,coeff_slices,output_format='wavedec2')
    return coeff_thresh


coeff_thresh = threshold_coefficients(coeffs,0.15)

# Reconstruction
recon = pywt.waverec2(coeff_thresh,wavelet=w,mode=mode)
abs_diff = np.abs(image-recon)

# Calculate MSE, PSNR, and SSIM
vmin = min(image.min(),recon.min())
vmax = max(image.max(),recon.max())

mse = np.mean(abs_diff**2)
psnr = peak_signal_noise_ratio(image, recon, data_range=vmax-vmin)
ssim = structural_similarity(image, recon, data_range=vmax-vmin)
print(f'15% threshold PSNR: {psnr:.5f}')
print(f'15% threshold SSIM: {ssim:.5f}')

# Show original, reconstruction, and difference
fig, axes = plt.subplots(1,3,figsize=(8,8))

axes[0].imshow(image,cmap='gray',vmin=vmin,vmax=vmax)
axes[0].text(0., 1.05, '(a) Original', fontsize=12, transform=axes[0].transAxes, fontweight="bold")

axes[1].imshow(recon,cmap='gray',vmin=vmin,vmax=vmax)
axes[1].text(0., 1.05, '(b) Reconstructed', fontsize=12, transform=axes[1].transAxes, fontweight="bold")

axes[2].imshow(abs_diff,cmap='gray')
axes[2].text(0., 1.05, '(c) Difference (abs)', fontsize=12, transform=axes[2].transAxes, fontweight="bold")

for ax in axes:
    ax.axis('off')

plt.tight_layout()

fig.savefig(join(args.output_dir, 'e2p3_threshold_15.png'))

# Plot the coefficients and highlight the largest 15%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

retained = pywt.coeffs_to_array(coeff_thresh)[0] > 0

ax.imshow(retained, cmap='gray')
ax.axis('off')

fig.savefig(join(args.output_dir, 'e2p3_threshold_transform.png'))

#####################
# Try different thresholds
#####################

coeffs = pywt.wavedec2(image,wavelet=w,mode=mode,level=n)

keep_proportions = [0.2, 0.1, 0.05, 0.025, 0.005]

fig, axes = plt.subplots(2, len(keep_proportions), figsize=(15, 8))

for i, keep in enumerate(keep_proportions):

    # Threshold the coefficients
    orig_ = (pywt.coeffs_to_array(coeffs)[0] != 0).sum()
    coeff_thresh = threshold_coefficients(coeffs,keep)

    # Reconstruction
    recon = pywt.waverec2(coeff_thresh,wavelet=w,mode=mode)
    abs_diff = np.abs(image-recon)

    # Calculate MSE, PSNR, and SSIM
    vmin = min(image.min(),recon.min())
    vmax = max(image.max(),recon.max())
    psnr = peak_signal_noise_ratio(image, recon, data_range=vmax-vmin)
    ssim = structural_similarity(image, recon, data_range=vmax-vmin)

    # Show reconstruction and difference

    axes[0,i].imshow(recon,cmap='gray')
    axes[0,i].set_title(f'{keep*100}% Retention', fontsize=12)
    axes[0,i].axis('off')

    axes[1,i].imshow(abs_diff,cmap='gray')
    axes[1,i].set_title(f'PSNR: {psnr:.3f}, SSIM: {ssim:.3f}', fontsize=12)
    axes[1,i].axis('off')

plt.tight_layout()
fig.savefig(join(args.output_dir, 'e2p3_all_compression.png'))