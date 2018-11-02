#

#import sys
#sys.path.append('/home/xyy/code/py/HIPS_autograd')  # for autograd
#sys.path.append('/home/xyy/code/py/dnn_hess/')      # for def_dnn

import autograd as ag
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.misc.optimizers import adam
from autograd.numpy import dot, linspace, pi, diag, arange
from autograd.numpy import sign, abs, sqrt, exp, log, log10, sin, cos, tan, arccos, arcsin, arctan, sinh, cosh, tanh, arcsinh, arccosh, arctanh
from autograd.numpy.linalg import norm

from pandas import DataFrame

# Custom ANN
from utils.def_dnn import *

# Drawing
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import figure, clf, plot, scatter, xlabel, ylabel, xlim, ylim
# see mpl.rcParams for full list
mpl.rc('figure', figsize=(1600/240.0, 1200/240.0), dpi=240)
mpl.rc('lines', markersize=2.0)
mpl.rcParams['axes.formatter.limits'] = [-3,3]
plt.ion()

from utils.plot_util import pic_output as _pic_output, draw_log10_signed
pic_output = _pic_output

def normalize(v):
    return v / norm(v)

# vim: et sw=4 sts=4
