# Library imports

import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import argparse
import json

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras
from tensorflow.keras import Input

# Custom Imports

from constant import TRAIN_IDS_PATH, VALID_IDS_PATH, TRAIN_DIR, VALID_DIR, MODEL_INPUT_SIZE, PARAMS_PATH, DATA_DIR, CKPT_PATH
from data_loader import data_generator, get_ids
from callbacks import model_checkpoint_callback
from losses import dice_loss

from model_architectures import unet

def main(args):
    # TODO: hyperparameters
    # TODO: using json file
    # if args.use_json_file:
    #     json_file = str(args.use_json_file)
    #     with open(json_file, "r") as read_file:
    #         params = json.load(read_file)
    #         patch_size = params["patch_size"]
    #         batch_size = params["batch_size"]
    #         learning_rate = params["learning_rate"]
    #         num_epochs = params["num_epochs"]
    
    patch_size = args.patch_size
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    print(patch_size, batch_size, learning_rate, num_epochs)

    # Read in patient ids
    train_ids = get_ids(TRAIN_IDS_PATH)
    valid_ids = get_ids(VALID_IDS_PATH)

    # Declare model architecture
    model = unet(input_size=MODEL_INPUT_SIZE)

    # TODO: compile model
    # TODO: Test dice loss
    model.compile(optimizer=Adam(lr=learning_rate), loss=dice_loss)
    # TODO: Callbacks + model.fit

    train_generator = data_generator(ids=train_ids, data_dir=TRAIN_DIR, batch_size=batch_size, patch_size=patch_size)
    valid_generator = data_generator(ids=valid_ids, data_dir=VALID_DIR, batch_size=batch_size, patch_size=patch_size)



    model.fit(x=train_generator, \
      validation_data=valid_generator, \
      validation_steps=400 \
      validation_freq=5, \
      steps_per_epoch = 2000, \
      epochs=num_epochs, \
      callbacks=[model_checkpoint_callback])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # TODO: Configure argument parser
    parser.add_argument("-json", '--use_json_file', type=str, default=PARAMS_PATH)
    parser.add_argument("-p", "--patch_size", type=int, default=64)
    parser.add_argument("-bs", "--batch_size", type=int, default=1)
    parser.add_argument("-lr", "--learning_rate", type=int, default=1e-4)
    parser.add_argument("-d", "--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("-n", "--num_epochs", type=str, default=1000)

    # parser.add_argument('--data_dir', type=str, default=) # TODO: Add default directory
    args = parser.parse_args()

    main(args)