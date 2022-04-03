
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

from constant import TRAIN_IDS_PATH, VALID_IDS_PATH, TRAIN_DIR, VALID_DIR
from data_loader import data_generator, get_ids
from losses import dice_loss
from model_architectures import unet

# TODO: Add file for file paths

def main(args):
    # read in patient ids
    train_ids = get_ids(TRAIN_IDS_PATH)
    valid_ids = get_ids(VALID_IDS_PATH)

    # TODO: model architecture
    # model = unet(input_size=(64,64,64,1))

    # TODO: hyperparameters
    # TODO: compile model
    # TODO: dice loss
    # model.compile(optimizer=Adam(lr=learning_rate), loss=dice_loss)
    # TODO: Callbacks + model.fit

    train_generator = data_generator(ids=train_ids, data_dir=TRAIN_DIR, batch_size=1, patch_size=64)
    valid_generator = data_generator(ids=valid_ids, data_dir=VALID_DIR, batch_size=1, patch_size=64)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_dir', type=str, default=) # TODO: Add default directory
    args = parser.parse_args()

    main(args)
