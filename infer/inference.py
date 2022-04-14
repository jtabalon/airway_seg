# Library Imports
import argparse
import numpy as np
import tensorflow as tf
import nibabel as nib
from infer_constants import CKPT_PATH,TEST_IMG


# Custom Imports

def main(self):
    # TODO: Configure arguments

    img_dir = args.image_dir
    patch_size = args.patch_size
    weights_path = args.weights_path

    # TODO: Load model + weights

    model = tf.keras.models.load_model(weights_path, compile=False)





    # TODO: Load image
    img = nib.load(img_dir).get_fdata() / 2000.
    img = img[:,:,0:512] # Ask Kyle about how to handle this... original size is (512,512,654)
                         # How do we handle the excess z axis? z coordinate varies from img to img
    row_dim, column_dim, slice_dim  = img.shape[0], img.shape[1], img.shape[2]
    print(img.shape)

    # TODO: Iterate through patches

    

    # TODO: Iterate through image given patch size (start with 64)

    # TODO: Stitch inferred images together


    # TODO: Calculate dice metric.

    print(f"hello world")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # TODO: Configure arguments

    parser.add_argument("-p", "--patch_size", type=int, default=64)
    parser.add_argument("-w", "--weights_path", type=str, default=CKPT_PATH)
    parser.add_argument("-i", "--image_dir", type=str, default=TEST_IMG)

    args = parser.parse_args()

    main(args)