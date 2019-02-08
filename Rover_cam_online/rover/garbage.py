#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 16:44:06 2018

@author: mpcr
"""

import time
import numpy as np
import h5py
import progressbar
import datetime
import tflearn
from tflearn.layers.core import input_data
import torchvision.models as models
# =============================================================================
# from NetworkSwitch import *
# import torch
# import torch.nn as nn
# #from scipy.misc import imresize
# from skimage.transform import resize
# import pdb
# =============================================================================


import sys
sys.path.insert(0,'/home/mpcr/Adam_python/rover')
from Q_Rover import Q_agent
