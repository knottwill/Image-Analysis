"""
Script to compare CT reconstruction methods: FBP, TV, and Learned Gradient Descent (LGD) networks.

This script was adapted from the provided notebook.

Usage:
python scripts/ct_reconstruction.py --output_dir ./figures
"""

# import libraries for CT and deep learning
import odl
import numpy as np
import matplotlib.pyplot as plt
import astra
astra.test()
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from time import time
import torch.nn as nn
import argparse
import os

import torch
import odl
import odl.contrib.torch as odl_torch

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', default='./figures', type=str, help='Path to save the plot')
args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)

# --- Set up the forward operator (ray transform) in ODL --- #
# Reconstruction space: functions on the rectangle [-20, 20]^2
img_size = 256 # discretized with 256 samples per dimension
reco_space = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20],
                               shape=[img_size, img_size], dtype='float32')
# Make a parallel beam geometry with flat detector, using number of angles = num_angles
num_angles = 30
geometry = odl.tomo.parallel_beam_geometry(reco_space, num_angles=num_angles)

# Create the forward operator, adjoint operator, and the FBO operator in ODL
fwd_op_odl = odl.tomo.RayTransform(reco_space, geometry)
fbp_op_odl = odl.tomo.fbp_op(fwd_op_odl,filter_type='Ram-Lak',
                             frequency_scaling=0.6)
adj_op_odl = fwd_op_odl.adjoint

# Create phantom and noisy projection data in ODL
phantom_odl = odl.phantom.shepp_logan(reco_space, modified=True) # ground-truth image
data_odl = fwd_op_odl(phantom_odl) # sinogram
data_odl += odl.phantom.white_noise(fwd_op_odl.range) * np.mean(data_odl) * 0.1 # noisy sinogram

# Reconstruct the image using FBP
t0 = time()
fbp_odl = fbp_op_odl(data_odl)
print('FBP reconstruction time = {:.3f} seconds'.format(time()-t0))

# convert the image and the sinogram to numpy arrays
phantom_np = phantom_odl.__array__()
fbp_np = fbp_odl.__array__()
data_np = data_odl.__array__()
print('sinogram size = {}'.format(data_np.shape))

data_range=np.max(phantom_np)-np.min(phantom_np)
psnr_fbp = compare_psnr(phantom_np,fbp_np,data_range=data_range)
ssim_fbp = compare_ssim(phantom_np,fbp_np,data_range=data_range)

# Display Shepp-Logan phantom and noisy sinogram
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(phantom_np.transpose(), cmap='bone')
axes[0].text(0, 1.05, '(a) Shepp-Logan phantom', fontsize=15, transform=axes[0].transAxes)
axes[0].axis('off')

axes[1].imshow(data_np, cmap='bone')
axes[1].text(0, 1.5, '(b) Noisy sinogram', fontsize=15, transform=axes[1].transAxes)
axes[1].axis('off')

fig.savefig(args.output_dir + '/phantom_sinogram.png', bbox_inches='tight', dpi=300)

# Let's solve the TV reconstruction problem using the linearized ADMM algorithm (implemented in ODL).

# In this example we solve the optimization problem: min_x  f(x) + g(Lx) = ||A(x) - y||_2^2 + lam * ||grad(x)||_1, Where ``A`` is a parallel beam ray transform, ``grad`` is the spatial gradient and ``y`` given noisy data.

# The problem is rewritten in decoupled form as: min_x g(L(x)) with a separable sum ``g`` of functionals and the stacked operator ``L``:

#     g(z) = ||z_1 - g||_2^2 + lam * ||z_2||_1,

#                ( A(x)    )
#     z = L(x) = ( grad(x) ).

# Gradient operator for the TV part
grad = odl.Gradient(reco_space)

# Stacking of the two operators
L = odl.BroadcastOperator(fwd_op_odl, grad)

# Data matching and regularization functionals
data_fit = odl.solvers.L2NormSquared(fwd_op_odl.range).translated(data_odl)
lam = 0.015
reg_func = lam * odl.solvers.L1Norm(grad.range)
g = odl.solvers.SeparableSum(data_fit, reg_func)

# We don't use the f functional, setting it to zero
f = odl.solvers.ZeroFunctional(L.domain)

# --- Select parameters and solve using ADMM --- #

# Estimated operator norm, add 10 percent for some safety margin
op_norm = 1.1 * odl.power_method_opnorm(L, maxiter=20)

niter = 200  # Number of iterations
sigma = 2.0  # Step size for g.proximal
tau = sigma / op_norm ** 2  # Step size for f.proximal

# Optionally pass a callback to the solver to display intermediate results
callback = (odl.solvers.CallbackPrintIteration(step=10) &
            odl.solvers.CallbackShow(step=10))

# Choose a starting point
x_admm_odl = L.domain.zero()

# Run the algorithm
t0 = time()
odl.solvers.admm_linearized(x_admm_odl, f, g, L, tau, sigma,
                            niter, callback=None)
x_admm_np = x_admm_odl.__array__()
print('ADMM reconstruction time = {:.3f} seconds'.format(time() - t0))

# Compute PSNR and SSIM
psnr_tv = compare_psnr(phantom_np,x_admm_np,data_range=data_range)
ssim_tv = compare_ssim(phantom_np,x_admm_np,data_range=data_range)

# Now, we will "train" learned gradient descent (LGD) network on this specific image to show the potential of data-driven reconstruction.
# Keep in mind that this example is unrealistic, in the sense that in practice you will train your learned reconstruction network on a dataset containing
# many images, and then test the performance on a new test image (that was not a part of your training dataset).

### first, let's create a CNN that replaces the prox operator in PGD.
class prox_net(nn.Module):
    def __init__(self, n_in_channels=2, n_out_channels = 1, n_filters=32, kernel_size=3):
        super(prox_net, self).__init__()
        self.pad = (kernel_size-1)//2
        self.conv1 = nn.Conv2d(n_in_channels, out_channels=n_filters, kernel_size=kernel_size, stride=1, padding=self.pad, bias=True)
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, stride=1, padding=self.pad, bias=True)
        self.conv3 = nn.Conv2d(n_filters, out_channels=1, kernel_size=kernel_size, stride=1, padding=self.pad, bias=True)

        self.act1 = nn.PReLU(num_parameters=1, init=0.25)
        self.act2 = nn.PReLU(num_parameters=1, init=0.25)

    def forward(self, x, u):
        ''' ################## YOUR CODE HERE #################### '''
        ### Note: here the two inputs denote the current iterate and the gradient
        inpt = torch.cat((x,u),dim=0)
        out = self.act1(self.conv1(inpt))
        out = self.act2(self.conv2(out))
        out = self.conv3(out)

        return out

# Let's compute a reasonable initial value for the step-size as step_size = 1/L, where L is the spectral norm of the forward operator.
op_norm = 1.1*odl.power_method_opnorm(fwd_op_odl)
step_size = 1/op_norm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fbp_op = odl_torch.OperatorModule(fbp_op_odl).to(device)
fwd_op = odl_torch.OperatorModule(fwd_op_odl).to(device)
adj_op = odl_torch.OperatorModule(adj_op_odl).to(device)

class LGD_net(nn.Module):
    def __init__(self, niter=5,step_size=step_size):
        super(LGD_net, self).__init__()
        self.niter = niter
        self.prox = nn.ModuleList([prox_net().to(device) for i in range(self.niter)])
        self.step_size = nn.Parameter(step_size * torch.ones(self.niter).to(device))
    def forward(self, y, x_init):
        x = x_init
        ''' ################## YOUR CODE HERE #################### '''
        #### Note: the gradient at a given x is A^T(Ax-y).
        for k in range(self.niter):
            f_grad = adj_op(fwd_op(x) - y)
            x = x - self.step_size[k] * self.prox[k](x,f_grad)

        return x

# Now, we can use the LGD network to reconstruct the image.
lgd_net = LGD_net().to(device) # realize the network and export it to GPU
num_learnable_params = sum(p.numel() for p in lgd_net.parameters() if p.requires_grad)
print('number of model parameters = {}'.format(num_learnable_params))
y = torch.from_numpy(data_np).to(device).unsqueeze(0) # noisy sinogram data as a torch tensor

# from odl.tomo import fbp_op
# initialization for the LGD net. Note that the input to a torch 2D CNN must be of size (num_batches x height x width).
# Initialise y as a RayTransform object

x_init = fbp_op(y)

# print(x_init.shape, y.shape)

ground_truth = torch.from_numpy(phantom_np).to(device).unsqueeze(0) # target ground-truth as a torch tensor

# define the loss and the optimizer
mse_loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(lgd_net.parameters(),lr = 1e-4)
num_epochs = 2000

##### training loop
t0 = time()
losses = []
print('Training LGD network...')
for epoch in range(0, num_epochs):
    optimizer.zero_grad()
    '''################## YOUR CODE HERE #################### '''
    #### You need to compute the reconstructed image by applying a forward pass through lgd_net, compute the loss, call the backward function on loss, and update the parameters.
    recon = lgd_net(y, x_init)
    loss = mse_loss(recon, ground_truth)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if(epoch%100 == 0):
        print('epoch = {}, loss = {}'.format(epoch, loss.item()))
print('Training time = {:.3f} seconds'.format(time()-t0))

# Time inference
t0 = time()
recon = lgd_net(y, x_init)
print('LGD Inference time = {:.3f} seconds'.format(time()-t0))

lgd_recon_np = recon.detach().cpu().numpy().squeeze() # convert the LGD reconstruction to numpy format

psnr_lgd = compare_psnr(phantom_np,lgd_recon_np,data_range=data_range)
ssim_lgd = compare_ssim(phantom_np,lgd_recon_np,data_range=data_range)

# Plot the reconstructions
fig, axes = plt.subplots(1, 4, figsize=(12, 6))

axes[0].imshow(phantom_np.transpose(), cmap='bone')
axes[0].text(0, 1.05, '(a) Ground-truth', fontsize=15, transform=axes[0].transAxes)

axes[1].imshow(fbp_np.transpose(), cmap='bone')
axes[1].text(0, 1.05, '(b) FBP', fontsize=15, transform=axes[1].transAxes)
axes[1].set_xlabel('PSNR: {:.2f} dB, SSIM: {:.2f}'.format(psnr_fbp, ssim_fbp))

axes[2].imshow(x_admm_np.transpose(), cmap='bone')
axes[2].text(0, 1.05, '(c) TV', fontsize=15, transform=axes[2].transAxes)
axes[2].set_xlabel('PSNR: {:.2f} dB, SSIM: {:.2f}'.format(psnr_tv, ssim_tv))

axes[3].imshow(lgd_recon_np.transpose(), cmap='bone')
axes[3].text(0, 1.05, '(d) LGD', fontsize=15, transform=axes[3].transAxes)
axes[3].set_xlabel('PSNR: {:.2f} dB, SSIM: {:.2f}'.format(psnr_lgd, ssim_lgd))

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

fig.savefig(args.output_dir + '/ct_reconstructions.png', bbox_inches='tight', dpi=300)