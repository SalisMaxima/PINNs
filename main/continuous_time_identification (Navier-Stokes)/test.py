print('Importing Libraries...')

# Import TensorFlow = 1.15 
import tensorflow.compat.v1 as tf
#print(tf.__version__)
import numpy as np
print(np.__version__)
import matplotlib.pyplot as plt
print("matplotlib imported")
import scipy.io
print("scipy.io imported")
from scipy.interpolate import griddata
print("scipy.interpolate imported")
import time
print("time imported")
from itertools import product, combinations
print("itertools imported")
from mpl_toolkits.mplot3d import Axes3D
print("mpl_toolkits.mplot3d imported")
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
print("mpl_toolkits.mplot3d.art3d imported")
from plotting_navier import newfig, savefig
print("plotting_navier imported")

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
print("All Libraries Imported Successfully!")