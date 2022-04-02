
"""
Library imports
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import argparse

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras
from tensorflow.keras import Input

"""
Custom Imports
"""

from data_generator import data_generator







def main(args):
    # data_gen = data_generator(ids=)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=) # Add default directory
    args = parser.parse_args()

    main(args)
