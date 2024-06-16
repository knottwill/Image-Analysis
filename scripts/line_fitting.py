"""
Script to solve the problem 2.1 of the coursework.

Fitting straight lines to data in y_line.txt and y_outlier_line.txt using the l1 and l2 norms.

python scripts/line_fitting.py --y_line ./data/y_line.txt --y_outlier_line ./data/y_outlier_line.txt --output ./figures/line_fits.png
"""

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import argparse

# Parse the arguments
parser = argparse.ArgumentParser(description='Plot the data')
parser.add_argument('--y_line', default='./data/y_line.txt', type=str, help='Path to y_line.txt')
parser.add_argument('--y_outlier_line', default='./data/y_outlier_line.txt', type=str, help='Path to y_outlier_line.txt')
parser.add_argument('--output', default='./figures/line_fits.png', type=str, help='Path to save the plot')
args = parser.parse_args()

# Load the data
y_line = np.loadtxt(args.y_line)
y_outlier_line = np.loadtxt(args.y_outlier_line)
x = np.arange(0, len(y_line))

# Function to solve for L2 norm (least squares) using closed-form solution
def solve_l2(x, y):
    n = len(x)
    S_x = np.sum(x)
    S_y = np.sum(y)
    S_xx = np.sum(x**2)
    S_xy = np.sum(x * y)
    
    m = (n * S_xy - S_x * S_y) / (n * S_xx - S_x**2)
    c = (S_y - m * S_x) / n
    return m, c

# Function to solve for L1 norm using linear programming
def solve_l1(x, y):
    m = cp.Variable()
    c = cp.Variable()
    u = cp.Variable(len(x))
    
    constraints = [u >= y - (m * x + c), u >= -(y - (m * x + c))]
    objective = cp.Minimize(cp.sum(u))
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    print("Solver: ", problem.solver_stats.solver_name)
    return m.value, c.value

# Solve the problems
a_line_l2, b_line_l2 = solve_l2(x, y_line)
a_outlier_line_l2, b_outlier_line_l2 = solve_l2(x, y_outlier_line)

a_line_l1, b_line_l1 = solve_l1(x, y_line)
a_outlier_line_l1, b_outlier_line_l1 = solve_l1(x, y_outlier_line)

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
