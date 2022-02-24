# environment name: tf_airway

import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

""" DL imports """

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
#from tensorflow.keras 

# TODO read in image and mask

ct_path = "/home/jtabalon/airway_seg/test_data/ct_patients10004O.nii"
aw_path = "/home/jtabalon/airway_seg/test_data/aw_10004O.nii"

img = nib.load(ct_path).get_fdata()
mask = nib.load(aw_path).get_fdata()

print(img.shape)
# Visualize CT

#plt.imshow(img[:, :, 100])
#plt.show()




# %%