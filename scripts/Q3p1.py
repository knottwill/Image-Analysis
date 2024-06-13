"""
Script to solve problem 3.1 of the coursework.

Minimize the function f(x) = x1^2/2 + x2^2 using the gradient descent method with the step size alpha = 1/L, 
where L is the Lipschitz constant of the gradient of f.

python ./scripts/Q3p1.py
"""

def f(x):
    return x[0]**2/2 + x[1]**2

def grad_f(x):
    return [x[0], 2*x[1]]

def gradient_descent(f, grad_f, x0, x_min, alpha=1/2, eps=0.01):
    x = x0
    k = 0
    while f(x) - f(x_min) > eps:
        x = [x[i] - alpha*grad_f(x)[i] for i in range(2)]
        k += 1
    return k

if __name__ == '__main__':
    x0 = [1, 1]
    x_min = [0, 0]
    L = 2
    accuracy = 0.01

    K = gradient_descent(f, grad_f, x0, x_min, alpha=1/L, eps=accuracy)
    print(f'Number of iterations: {K}')