# Autograd & Numpy
import autograd
import autograd.numpy as np

# Pandas
import pandas as pd

# Matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 10] # [width, height] (inches). 

# Jupyter & IPython
from IPython.display import display


def grad(f):
    g = autograd.grad
    def grad_f(x, y):
        return np.array([g(f, 0)(x, y), g(f, 1)(x, y)])
    return grad_f

def J(f):
    j = autograd.jacobian
    def J_f(x, y):
        return np.array([j(f, 0)(x, y), j(f, 1)(x, y)]).T
    return J_f



def display_contour(f, x, y, levels):
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    fig, ax = plt.subplots()
    contour_set = plt.contour(
        X, Y, Z, colors="grey", linestyles="dashed", 
        levels=levels 
    )
    ax.clabel(contour_set)
    plt.grid(True)
    plt.xlabel("$x_1$") 
    plt.ylabel("$x_2$")
    plt.gca().set_aspect("equal")


def f1(x1, x2):
    # x1 = np.array(x1)
    # x2 = np.array(x2)
    return np.array([3.0 * x1 * x1 - 2.0 * x1 * x2 + 3.0 * x2 * x2, 1])

def f2(x1, x2):
    return (x1 - 1)**2 + (x1 - x2**2)**2

def f3(x, y):
    return np.sin(x + y) - np.cos(x * y) - 1 + 0.001 * (x * x + y * y)

def f4(x,y):
    return x**2 + y**2

def f5(x,y):
    return np.array([np.sin(x),np.sin(y)])

N=100
eps=10**(-2)

def Newton(F, x0, y0, eps=eps, N=N):
    JF=J(F)
    for i in range(N):
        X0 = np.array([x0,y0])
        X= X0 - np.linalg.inv(JF(x0,y0)).dot(F(x0,y0))
        x,y = X
        if np.sqrt((x - x0)**2 + (y - y0)**2) <= eps:
            return (x, y),f'atteint en {i} étapes'
        x0, y0 = x, y
    else:
        raise ValueError(f"no convergence in {N} steps.")

print(Newton(f1, 0.9, 0.8))