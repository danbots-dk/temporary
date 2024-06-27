# NOTE run from cd <pathtodir>/project/

"""
all packages are imported here
"""

#-----------------------------------------------------------------
## debugging
#-----------------------------------------------------------------
import pdb

#-----------------------------------------------------------------
## essentials
#-----------------------------------------------------------------
import numpy as np
from numpy.matlib import repmat
import pandas as pd
import seaborn as sns
import time
import timeit
import datetime
from tqdm import tqdm # progress bar
import string
import re
from random import choices,choice, sample
import os, glob, subprocess, sys, socket, struct, random, math, argparse, logging, time

from math import hypot
import itertools
from itertools import combinations, product
from statistics import mean
import copy
import json
from pathlib import Path # path manipulation
import pickle as pkl # saving 
from packaging import version # TODO what is this for? 
from sys import platform
import inspect
import socket
import collections
from statistics import mean

#-----------------------------------------------------------------
## plotting
#-----------------------------------------------------------------
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import figure
import matplotlib.image as mpimg
import matplotlib
from matplotlib.colors import * #LinearSegmentedColormap,TwoSlopeNorm
import plotly.express as px
from pylab import *


#-----------------------------------------------------------------
## image and pc packages
#-----------------------------------------------------------------
# from skimage import io
# from skimage.transform import resize
# import open3d as o3d
# from scipy import ndimage
import cv2
from PIL import Image
import imageio

import skimage.color
import skimage.filters
import skimage.io
from skimage import util
from skimage import morphology as morph
from skimage import exposure

#-----------------------------------------------------------------
# sklearn
#-----------------------------------------------------------------
# from sklearn.externals._pilutil import bytescale
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV

#-----------------------------------------------------------------
## tensorflow and keras
#-----------------------------------------------------------------
import tensorflow as tf
import numpy as np


# handle GPU resources in a multis-user environment
# https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory
config_resources = tf.compat.v1.ConfigProto()
config_resources.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config_resources)

# TODO: keras and tf.keras are two different bundles.
# https://www.pyimagesearch.com/2019/10/21/keras-vs-tf-keras-whats-the-difference-in-tensorflow-2-0/
from tensorflow import keras 

# NOTE: keep tensorflow.keras.<SUBPACKAGE>
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization,Concatenate, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Add, add, Activation, UpSampling2D,Conv2DTranspose
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# TODO: why here using tensorflow.python.keras instead of tensorflow.keras? 
from tensorflow.python.keras.layers import Layer, InputSpec
from focal_loss import SparseCategoricalFocalLoss
print (f"\nPackages import successful")

