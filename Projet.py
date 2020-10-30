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
    return 3.0 * x1 * x1 - 2.0 * x1 * x2 + 3.0 * x2 * x2

def f2(x1, x2):
    return (x1 - 1)**2 + (x1 - x2**2)**2

def f3(x, y):
    return np.sin(x + y) - np.cos(x * y) - 1 + 0.001 * (x * x + y * y)

def f4(x,y):
    return x**2 + y**2

def f5(x,y):
    return np.array([np.sin(x),np.sin(y)])

N_newt=100
eps_newt=10**(-2)

def Newton(F, x0, y0, eps=eps_newt, N=N_newt):
    JF=J(F)
    for i in range(N):
        X0 = np.array([x0,y0])
        X= X0 - np.linalg.inv(JF(x0,y0)).dot(F(x0,y0))
        x,y = X
        if np.sqrt((x - x0)**2 + (y - y0)**2) <= eps:
            return (x, y), f'atteint en {i} étapes'
        x0, y0 = x, y
    else:
        raise ValueError(f"no convergence in {N} steps.")

def level_curve(f, x0, y0, delta=0.1, N=1000, eps=eps_newt):
    c = f(x0, y0)
    def f_delta(x,y):
        return np.array([f(x,y) - c, (x-x0)**2 + (y-y0)**2 - delta**2])
    grad_f = grad(f)
    Jf = J(f)
    X, Y = [x0], [y0]
    for i in range(N-1):
        dx, dy = grad_f(x0, y0)
        norme = np.sqrt(dx**2 + dy**2)
        x, y = Newton(f_delta, x0 + delta*dy/norme, y0 - delta*dx/norme, eps=eps)[0]
        X.append(x)
        Y.append(y)
        x0, y0 = x, y
    return np.array([X, Y])

# courbe = level_curve(f3, 0., 0., N=10)
# plt.scatter(courbe[0], courbe[1])
# plt.show()


def seg_inter(seg1 : np.ndarray, seg2 : np.ndarray):
    point1, point2 = seg1
    point3, point4 = seg2
    d1 = point2 - point1 # dy*y + dx*x = c
    d2 = point4 - point3
    c1 = np.vdot(d1, point1)
    c2 = np.vdot(d2, point3)
    # on veut déterminer le croisement des deux droites : AX = B
    A = np.array([d1, d2])
    B = np.array([c1, c2])
    try :
        inter = np.linalg.inv(A).dot(B)
    except numpy.linalg.LinAlgError:
        if point1 == point3 or point2 == point4 :
            print('exc')
            return True
        return False
    if (np.minimum(point1, point2) <= inter).all() and (inter <= np.maximum(point1, point2)).all():
        print('norm')
        return True
    return False

seg1 = np.array([[0,1],[1,0]])
seg2 = np.array([[0,0],[1,1]])
print(seg_inter(seg1, seg2))