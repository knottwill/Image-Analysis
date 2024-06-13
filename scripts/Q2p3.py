"""
Script to solve problem 3.3 of coursework.

Usage:
python ./scripts/Q2p3.py --img ./data/river_side.jpeg --output_dir ./figures
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import os
from os.path import join
import pywt
import argparse
import pandas as pd

# Parse the arguments
parser = argparse.ArgumentParser(description='Plot the data')
parser.add_argument('--img', default='./data/river_side.jpeg', type=str, help='Path to river_side.jpeg')
parser.add_argument('--output_dir', default='./figures', type=str, help='Path to output directory')
args = parser.parse_args()

# Load the image
A = io.imread('./data/river_side.jpeg')
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
fig.savefig(join(args.output_dir, 'wavelet_transform.png'), dpi=300)

coeffs = pywt.wavedec2(image,wavelet=w,mode=mode,level=n)

# Reconstruction
recon = pywt.waverec2(coeffs,wavelet=w,mode=mode)

abs_diff = np.abs(image-recon)

# Calculate MSE and PSNR and compression ratio
mse = np.mean((image-recon)**2)
psnr = 10*np.log10(255**2/mse)

print('Uncompressed reconstruction:')
print(f'MSE: {mse}')
print(f'PSNR: {psnr:.2f} dB')

# Show original, reconstruction, and difference
fig, axes = plt.subplots(1,3,figsize=(8,8))
vmin = min(image.min(),recon.min())
vmax = max(image.max(),recon.max())
axes[0].imshow(image,cmap='gray',vmin=vmin,vmax=vmax)
axes[0].set_title('Original', fontsize=15)

axes[1].imshow(recon,cmap='gray',vmin=vmin,vmax=vmax)
axes[1].set_title('Reconstructed', fontsize=15)

axes[2].imshow(abs_diff,cmap='gray')
axes[2].set_title('Difference (abs)', fontsize=15)

for ax in axes:
    ax.axis('off')

plt.tight_layout()

fig.savefig(join(args.output_dir, 'reconstruction.png'))

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

# Calculate MSE and PSNR
mse = np.mean((image-recon)**2)
psnr = 10*np.log10(255**2/mse)
print('Compressed reconstruction:')
print(f'MSE: {mse}')
print(f'PSNR: {psnr:.2f} dB')

# Show original, reconstruction, and difference
fig, axes = plt.subplots(1,3,figsize=(8,8))
vmin = min(image.min(),recon.min())
vmax = max(image.max(),recon.max())

axes[0].imshow(image,cmap='gray',vmin=vmin,vmax=vmax)
axes[0].set_title('Original', fontsize=15)

axes[1].imshow(recon,cmap='gray',vmin=vmin,vmax=vmax)
axes[1].set_title('Reconstructed', fontsize=15)

axes[2].imshow(abs_diff,cmap='gray')
axes[2].set_title('Difference (abs)', fontsize=15)

for ax in axes:
    ax.axis('off')

plt.tight_layout()

fig.savefig('./figures/reconstruction_15_compressed.png')

# Plot the coefficients and highlight the largest 15%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

retained = pywt.coeffs_to_array(coeff_thresh)[0] > 0

ax.imshow(retained, cmap='gray')
ax.axis('off')

fig.savefig(join(args.output_dir, 'thresholded_transform.png'))

#####################
# Try different thresholds
#####################

coeffs = pywt.wavedec2(image,wavelet=w,mode=mode,level=n)

keep_proportions = [0.2, 0.1, 0.05, 0.025, 0.005]

fig, axes = plt.subplots(len(keep_proportions), 2, figsize=(10, 15))

for i, keep in enumerate(keep_proportions):

    # Threshold the coefficients
    orig_ = (pywt.coeffs_to_array(coeffs)[0] != 0).sum()
    coeff_thresh = threshold_coefficients(coeffs,keep)

    # Reconstruction
    recon = pywt.waverec2(coeff_thresh,wavelet=w,mode=mode)
    abs_diff = np.abs(image-recon)

    # Calculate MSE and PSNR
    mse = np.mean((image-recon)**2)
    psnr = 10*np.log10(255**2/mse)

    # Show reconstruction and difference

    axes[i,0].imshow(recon,cmap='gray')
    axes[i,0].set_title(f'Keep: {keep}', fontsize=15)
    axes[i,0].axis('off')

    axes[i,1].imshow(abs_diff,cmap='gray')
    axes[i,1].set_title(f'MSE: {mse:.2f}, PSNR: {psnr:.2f}', fontsize=15)
    axes[i,1].axis('off')

plt.tight_layout()
fig.savefig(join(args.output_dir, 'all_compression_results.png'))