"""
Script to solve the problem 2.1 of the coursework.

Fitting straight lines to data in y_line.txt and y_outlier_line.txt using the l1 and l2 norms.

python scripts/Q2p1.py --y_line ./data/y_line.txt --y_outlier_line ./data/y_outlier_line.txt --output ./figures/linear_regression.png
"""

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import argparse

# Parse the arguments
parser = argparse.ArgumentParser(description='Plot the data')
parser.add_argument('--y_line', default='./data/y_line.txt', type=str, help='Path to y_line.txt')
parser.add_argument('--y_outlier_line', default='./data/y_outlier_line.txt', type=str, help='Path to y_outlier_line.txt')
parser.add_argument('--output', default='./figures/linear_regression.png', type=str, help='Path to save the plot')
args = parser.parse_args()

# Load the data
y_line = np.loadtxt('./data/y_line.txt')
y_outlier_line = np.loadtxt('./data/y_outlier_line.txt')
x = np.arange(0, len(y_line))

# Define the variables
a = cp.Variable()
b = cp.Variable()

# Define the objective functions
y_line_l1_obj = cp.Minimize(cp.norm(a * x + b - y_line, 1)) # l1 norm y_line
y_line_l2_obj = cp.Minimize(cp.norm(a * x + b - y_line, 2)) # l2 norm y_line
y_outlier_line_l1_obj = cp.Minimize(cp.norm(a * x + b - y_outlier_line, 1)) # l1 norm y_outlier_line
y_outlier_line_l2_obj = cp.Minimize(cp.norm(a * x + b - y_outlier_line, 2)) # l2 norm y_outlier_line

# Define the problems
y_line_l1 = cp.Problem(y_line_l1_obj)
y_line_l2 = cp.Problem(y_line_l2_obj)
y_outlier_line_l1 = cp.Problem(y_outlier_line_l1_obj)
y_outlier_line_l2 = cp.Problem(y_outlier_line_l2_obj)

# Solve the problems
y_line_l1.solve()
a_line_l1 = a.value
b_line_l1 = b.value

y_line_l2.solve()
a_line_l2 = a.value
b_line_l2 = b.value

y_outlier_line_l1.solve()
a_outlier_line_l1 = a.value
b_outlier_line_l1 = b.value

y_outlier_line_l2.solve()
a_outlier_line_l2 = a.value
b_outlier_line_l2 = b.value

# Plot the data
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].scatter(x, y_line, label='y_line')
axes[0].plot(x, a_line_l1 * x + b_line_l1, label=r'$\ell_1$', color='green')
axes[0].plot(x, a_line_l2 * x + b_line_l2, label=r'$\ell_2$', color='orange')
axes[0].grid()
axes[0].text(0, 1.05, '(a)', fontsize=15, transform=axes[0].transAxes, fontweight="bold")

axes[1].scatter(x, y_outlier_line, label='y_outlier_line')
axes[1].plot(x, a_outlier_line_l1 * x + b_outlier_line_l1, label=r'$\ell_1$', color='green')
axes[1].plot(x, a_outlier_line_l2 * x + b_outlier_line_l2, label=r'$\ell_2$', color='orange')
axes[1].grid()
axes[1].text(0, 1.05, '(b)', fontsize=15, transform=axes[1].transAxes, fontweight="bold")

for ax in axes:
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')

plt.tight_layout()
fig.savefig(args.output, bbox_inches='tight')