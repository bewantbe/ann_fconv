# 

import sys
sys.path.append('/home/xyy/code/py/HIPS_autograd')
sys.path.append('/home/xyy/code/py/dnn_hess/')

import autograd as ag
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.misc.optimizers import adam
from autograd.numpy import abs, dot, sqrt
from autograd.numpy.linalg import norm

from pandas import DataFrame

# Custom ANN
from def_dnn import *

# Drawing
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import figure, clf, plot, scatter, xlabel, ylabel, xlim, ylim
# see mpl.rcParams for full list
mpl.rc('figure', figsize=(1600/240.0, 1200/240.0), dpi=240)
mpl.rc('lines', markersize=2.0)
mpl.rcParams['axes.formatter.limits'] = [-3,3]
plt.ion()

def normalize(v):
    return v / norm(v)

layer_dims = [1, 2, 1]

cost_l = lambda pm: cost_flat(ans_xy, pm, layer_dims)
cost_g = ag.jacobian(cost_l)
cost_h = ag.hessian (cost_l)

# ANN function
ann_theta_x = lambda pm, x: np.squeeze(net_predict(x, wb_flat2layers(pm, layer_dims), active_function = 'sigmoid'))
ann_h = ag.hessian(ann_theta_x, argnum=0)

# value for parameters and input
pm = np.array([1, 2, -0.3, 0.5, 2, 3, -1])
x = 0.1;

# a glance of the ANN function
figure(11);  clf()
s_x = np.linspace(-1, 1, 100)[:, np.newaxis]
plot(s_x, ann_theta_x(pm, s_x))

def permute2nodewise(A, layer_dims):
    """ permute A so that parameters about a neuron is adjacent """
    n_param = n_ann_param(layer_dims)
    wbidx = wb_flat2layers(range(n_param), layer_dims)
    assert(len(layer_dims)==3)
    idx = np.vstack([wbidx[1][0][:,0], wbidx[0][1], wbidx[0][0][0,:]])
    idx = idx.flatten(order='F')
    idx = np.append(idx, wbidx[1][1])
    if len(A.shape) == 1:
        A = A[idx]
    elif len(A.shape) == 2:
        A = A[np.ix_(idx, idx)]
    else:
        raise ValueError('No joking.')
    return A

h = ann_h(pm, x)
h_node = permute2nodewise(h, layer_dims)

print('The hess:\n', DataFrame(h_node))

def verify_f_hess():
    """ verify hand compute """
    s = lambda x: sigmoid(x)
    sd = ag.grad(s)
    sdd = ag.grad(sd)

    r = pm[0]
    b = pm[2]
    w = pm[4]

    uu = [
    [          0,      sd(r*x+b),    sd(r*x+b)*x],
    [  sd(r*x+b),   w*sdd(r*x+b), w*sdd(r*x+b)*x],
    [sd(r*x+b)*x, w*sdd(r*x+b)*x, w*sdd(r*x+b)*x*x]]
    print('hess err =', norm(h_node[0:3, 0:3] - uu), '(should be machine epsilon)')

def verify_f_hess_eigen_val():
    """ verify hand compute """
    s = lambda x: sigmoid(x)
    sd = ag.grad(s)
    sdd = ag.grad(sd)
    r = pm[0]
    b = pm[2]
    w = pm[4]
    p1 = w*sdd(r*x+b)
    p2 = sqrt((w*sdd(r*x+b))**2 + 4 * sd(r*x+b)**2 / (1+x*x))
    eigvals = (1+x*x)/2 * np.array([p1 - p2, 0, p1 + p2])
    
    eigvals_ref, eigvecs_ref = np.linalg.eigh(h_node[0:3, 0:3])  # ref ans
    print('eig val err =', norm(eigvals - eigvals_ref), '(should be machine epsilon)')

    eigvecs = np.array([
        [-eigvals[2], sd(r*x+b), sd(r*x+b)*x],
        [0, -x, 1],
        [-eigvals[0], sd(r*x+b), sd(r*x+b)*x]])
    eigvecs = eigvecs.T / sqrt(np.sum(eigvecs * eigvecs, axis=1))
    eigvecs = eigvecs * np.sign(eigvecs[1,:]*eigvecs_ref[1,:])

    print('eig vec err =', norm(eigvecs - eigvecs_ref), '(should be machine epsilon)')

verify_f_hess()
verify_f_hess_eigen_val()


# vim: et sw=4 sts=4
