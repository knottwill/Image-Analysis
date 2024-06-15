"""
This script solves the compressed-sensing exercise (problem 2.2) of the coursework. 

Usage:
python scripts/compressed_sensing.py
"""

import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import argparse 

# tight layout
plt.rcParams.update({'figure.autolayout': True})

parser = argparse.ArgumentParser(description='Soft Thresholding')
parser.add_argument('--lam', type=float, default=0.05, help='Threshold value')
parser.add_argument('--iter', type=int, default=100, help='Number of iterations')
parser.add_argument('--output_dir', type=str, default='./figures', help='Output directory')
args = parser.parse_args()

def fftc(x):
    """ Centered Fast Fourier Transform (adapted from helper.py) """
    return 1 / np.sqrt(x.shape[0]) * np.fft.ifftshift(np.fft.fft(np.fft.ifftshift(x)))

def ifftc(y):
    """ Centered Inverse Fast Fourier Transform (adapted from helper.py) """
    return np.sqrt(y.shape[0]) * np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(y)))

def soft_thresh(x, lam):
    """ Soft thresholding function (adapted from helper.py) """
    if isinstance(x[0], complex):
        res = abs(x) - lam
        return (res > 0.) * res * x / abs(x)
    else:
        return np.sign(x) * np.maximum(np.abs(x) - lam, 0)

def IST(X_start, lam, nitr, X_ref=None):
    """ 
    Iterative Soft Thresholding (Adapted from helper.py)
    
    This is a projection over complex sets (POCS) type algorithm that solves the L1 minimization problem.
    """

    err = np.zeros((nitr,))
    Xi  = X_start.copy()
    for i in range(nitr):
        prev_est = ifftc(Xi)                # Inverse Fourier Transform
        est = soft_thresh(prev_est, lam)    # Soft thresholding
        Xi = fftc(est)                      # Fourier Transform
        Xi = Xi*(X_start==0) + X_start      # Keep the known values

        err[i] = np.linalg.norm(Xi - X_ref) # Compute the error

    recon = ifftc(Xi) # Reconstruct the signal
    return recon, err

# Generate the original signal with sparsity 10 and length 100.
np.random.seed(1)
L = 100
k = 10
true_s = np.zeros(L)

# Create a sparse signal
nzc = np.random.rand(k) + 1
true_s = np.concatenate((nzc, np.zeros(L - len(nzc))))
true_s = true_s[np.random.permutation(L)]

# Add Gaussian noise to the original signal.
sigma = 0.05
s = true_s + sigma * np.random.randn(L)

# Compute the Fourier transform to move to the frequency domain where we can subsample.
X = fftc(s)
keep = 32

# Create uniform and random masks for subsampling.
uniform_mask = np.zeros(L)
uniform_idx = np.arange(L, step=L // keep)[:keep]
uniform_mask[uniform_idx] = 1

random_mask = np.zeros(L)
random_idx = sorted(np.random.choice(L, keep, replace=False))
random_mask[random_idx] = 1

# Subsample by applying the masks.
X_uniform = X * uniform_mask
X_random = X * random_mask

# Reconstruct the signals (time domain) from the subsampled spectra.
s_uniform = ifftc(X_uniform) * 4
s_random = ifftc(X_random) * 4

# Visualize the reconstructed signals.
fig, axes = plt.subplots(2, 2, figsize=(10, 7))

axes[0, 0].stem(true_s)
axes[0, 0].text(0.03, 0.9, '(a)', fontsize=15, transform=axes[0, 0].transAxes, fontweight="bold")
axes[0, 0].set_ylabel('Real part', fontsize=12)

axes[0, 1].stem(s)
axes[0, 1].text(0.03, 0.9, '(b)', fontsize=15, transform=axes[0, 1].transAxes, fontweight="bold")

axes[1, 0].stem(np.real(s_uniform))
axes[1, 0].text(0.03, 0.9, '(c)', fontsize=15, transform=axes[1, 0].transAxes, fontweight="bold")
axes[1, 0].set_xlabel('Component', fontsize=12)
axes[1, 0].set_ylabel('Real part', fontsize=12)

axes[1, 1].stem(np.real(s_random))
axes[1, 1].text(0.03, 0.9, '(d)', fontsize=15, transform=axes[1, 1].transAxes, fontweight="bold")
axes[1, 1].set_xlabel('Component', fontsize=12)

fig.savefig(join(args.output_dir, 'CS_signals.png'))

#####################
# IST Reconstruction
#####################

s_rand_recon, err_rand = IST(X_random, args.lam, args.iter, fftc(s))
s_unif_recon, err_unif = IST(X_uniform, args.lam, args.iter, fftc(s))

# plot error
fig, ax = plt.subplots(1, 1, figsize=(5, 5))

ax.plot(err_rand, label='Random')
ax.plot(err_unif, label='Uniform')
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Error', fontsize=12)
ax.legend()

fig.savefig(join(args.output_dir, 'CS_error.png'))

# Plot reconstructions
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].stem(s)
axes[0].text(0.03, 0.9, '(a)', fontsize=15, transform=axes[0].transAxes, fontweight="bold")
axes[0].set_ylabel('Real part', fontsize=12)
axes[0].set_xlabel('Component', fontsize=12)

axes[1].stem(np.real(s_unif_recon))
axes[1].text(0.03, 0.9, '(b)', fontsize=15, transform=axes[1].transAxes, fontweight="bold")
axes[1].set_xlabel('Component', fontsize=12)

axes[2].stem(np.real(s_rand_recon))
axes[2].text(0.03, 0.9, '(c)', fontsize=15, transform=axes[2].transAxes, fontweight="bold")
axes[2].set_xlabel('Component', fontsize=12)

fig.savefig(join(args.output_dir, 'CS_reconstructions.png'))

# MSE
mse_rand = np.mean((s - np.abs(s_rand_recon)) ** 2)
mse_unif = np.mean((s - np.abs(s_unif_recon)) ** 2)
print("MSE random subsampling reconstruction and noisy signal:", mse_rand)
print("MSE uniform subsampling reconstruction and noisy signal:", mse_unif)

mse_rand_true = np.mean((true_s - np.abs(s_rand_recon)) ** 2)
mse_unif_true = np.mean((true_s - np.abs(s_unif_recon)) ** 2)
print("MSE random subsampling reconstruction and true signal:", mse_rand_true)
print("MSE uniform subsampling reconstruction and true signal:", mse_unif_true)