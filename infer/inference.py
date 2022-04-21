# Library Imports
import argparse
import numpy as np
import tensorflow as tf
import nibabel as nib
import os

# Custom Imports
from infer_constants import CKPT_PATH,TEST_IMG

def main(self):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # TODO: Configure arguments
    img_dir = args.image_dir
    patch_size = args.patch_size
    weights_path = args.weights_path

    patch_distance = int(patch_size / 2)

    # TODO: Load image
    img = nib.load(img_dir).get_fdata() / 2000.
    img = img[:,:,0:512] # Ask Kyle about how to handle this... original size is (512,512,654)
                         # How do we handle the excess z axis? z coordinate varies from img to img
    row_dim, column_dim, slice_dim  = img.shape[0], img.shape[1], img.shape[2]

    # TODO: Work with only 1 patch.

    first_patch_midpoint = (patch_distance, patch_distance, patch_distance)

    patch_mid_row, patch_mid_col, patch_mid_slice = patch_distance,patch_distance,patch_distance

    rows_patchs = []
    col_patchs = []
    slice_patches = []
    tot_patchs = []
    # TODO: Iterate through patches

    # # Slices   
    # while patch_mid_slice < slice_dim:
    #     # Columns
    #     while patch_mid_col < column_dim:
    #         # Rows
    #         while patch_mid_row < row_dim:
    #             row_patch = img[(patch_mid_row-patch_distance):(patch_mid_row+patch_distance), \
    #                     (patch_mid_col-patch_distance):(patch_mid_col+patch_distance), \
    #                     (patch_mid_slice-patch_distance):(patch_mid_slice+patch_distance)]
    #             expanded_row_patch = np.expand_dims(np.expand_dims(row_patch, -1), 0)
    #             print(expanded_row_patch.shape)

    #             rows_patchs.append(expanded_row_patch)
    #             tot_patchs.append(expanded_row_patch)
    #             patch_mid_row += patch_size

    #         col_patchs.append(rows_patchs)
    #         patch_mid_col += patch_size

    #     # col_patchs = np.array(col_patchs) Ask Kyle about ways to store this?
    #     slice_patches.append(col_patchs)
    #     patch_mid_slice += patch_size
       
    num_row_patchs, num_col_patchs, num_slice_patchs = int(row_dim / patch_size), \
                                                       int(column_dim / patch_size), \
                                                       int(slice_dim / patch_size)

    print(f"rows: {num_row_patchs} cols: {num_col_patchs} slices: {num_slice_patchs}  ")

    # create extra array size of image
    # and another array np.zeroes

    model = tf.keras.models.load_model(weights_path, compile=False)

    # TODO: Initialize 2 working arrays (per Kyle's guidance)

    predicted_mask = np.zeros(shape=(row_dim,column_dim,slice_dim))
    counts = np.zeros(shape=(row_dim,column_dim,slice_dim))

    # Gonna try Kyle's method now.
    counts = 0
    for slice in range(num_slice_patchs):
        for col in range(num_col_patchs):
            for patch in range(num_row_patchs):
                # Extract patch
                print(f"slice: {slice}")
                print(f"col: {col}")
                print(f"patch: {patch}")
                print(f"patch midpoint location: {patch_mid_row, patch_mid_col, patch_mid_slice}")
                row_patch = img[(patch_mid_row-patch_distance):(patch_mid_row+patch_distance), \
                        (patch_mid_col-patch_distance):(patch_mid_col+patch_distance), \
                        (patch_mid_slice-patch_distance):(patch_mid_slice+patch_distance)]
                # Expand dims (necessary for model prediction)
                print(f"patch shape: {np.shape(row_patch)}")
                expanded_row_patch = np.expand_dims(np.expand_dims(row_patch, -1), 0)
                print(f"expanded patch shape: {np.shape(expanded_row_patch)}")
                # Make prediction
                with tf.device("/device:GPU:0"):
                    inferred_patch = np.squeeze(model.predict(expanded_row_patch))
                    print(f"inferred patch shape: {np.shape(inferred_patch)}")
                    print(f"patch midpoint location: {patch_mid_row, patch_mid_col, patch_mid_slice}")
                    print("\n")
                    predicted_mask[(patch_mid_row-patch_distance):(patch_mid_row+patch_distance), \
                            (patch_mid_col-patch_distance):(patch_mid_col+patch_distance), \
                            (patch_mid_slice-patch_distance):(patch_mid_slice+patch_distance)] = \
                            predicted_mask[(patch_mid_row-patch_distance):(patch_mid_row+patch_distance), \
                            (patch_mid_col-patch_distance):(patch_mid_col+patch_distance), \
                            (patch_mid_slice-patch_distance):(patch_mid_slice+patch_distance)] + \
                                inferred_patch 
                if patch_mid_row < row_dim - patch_distance:
                    print(patch_mid_row)
                    patch_mid_row += patch_size
                counts += 1
            if patch_mid_col < column_dim - patch_distance:
                patch_mid_col += patch_size
        if patch_mid_slice < slice_dim - patch_distance:
            patch_mid_slice += patch_size
    
    print(counts)

# KYLES (below)
# counts and masks are all zeros of size img.

                        # mask = np.zeros(img.shape)

#                         mask[(patch_mid_row-patch_distance):(patch_mid_row+patch_distance), \
#                         (patch_mid_col-patch_distance):(patch_mid_col+patch_distance), \
#                         (patch_mid_slice-patch_distance):(patch_mid_slice+patch_distance)]
#                         = mask[(patch_mid_row-patch_distance):(patch_mid_row+patch_distance), \
#                         (patch_mid_col-patch_distance):(patch_mid_col+patch_distance), \
#                         (patch_mid_slice-patch_distance):(patch_mid_slice+patch_distance)]
#                         + mask_pred[] #from model.predict

                        # counts = np.zeros(img.shape)

#                         counts[(patch_mid_row-patch_distance):(patch_mid_row+patch_distance), \
#                         (patch_mid_col-patch_distance):(patch_mid_col+patch_distance), \
#                         (patch_mid_slice-patch_distance):(patch_mid_slice+patch_distance)]
#                         = counts[(patch_mid_row-patch_distance):(patch_mid_row+patch_distance), \
#                         (patch_mid_col-patch_distance):(patch_mid_col+patch_distance), \
#                         (patch_mid_slice-patch_distance):(patch_mid_slice+patch_distance)]
#                         + np.ones(patch_size)

                        # final mask = mask / counts

                        # Finally mask divided by counts: averaged image

#KYLES (above)


    # for slice in range(num_slice_patchs):
    #     for col in range(num_col_patchs):
    #         for row in range(num_row_patchs):
    #             row_patch = img[(patch_mid_row-patch_distance):(patch_mid_row+patch_distance), \
    #                     (patch_mid_col-patch_distance):(patch_mid_col+patch_distance), \
    #                     (patch_mid_slice-patch_distance):(patch_mid_slice+patch_distance)]
    #             expanded_row_patch = np.expand_dims(np.expand_dims(row_patch, -1), 0)
    #             patch_mid_row += patch_size
    #             rows_patchs.append(expanded_row_patch)
    #         patch_mid_col += patch_size
    #         # col_patchs.append(rows_patchs)
    #     patch_mid_slice += patch_size
        # slice_patches.append(col_patchs)

    # print(expanded_row_patch.shape)
    # rows_patchs.append(expanded_row_patch)
    # tot_patchs.append(expanded_row_patch)

    # print(len(tot_patchs))
    # print(np.shape(col_patchs))
    # print(len(rows_patchs))
    # print(len(col_patchs))
    # print(len(col_patchs[0]))

    # TODO: Iterate through image given patch size (start with 64)

    # TODO: Load model + weights
    # model = tf.keras.models.load_model(weights_path, compile=False)

    # TODO: Stitch inferred images together
    # inferred_img = model.predict(rows_patchs[0])
    # print(inferred_img.shape)

    # TODO: Calculate dice metric.

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # TODO: Configure arguments

    parser.add_argument("-p", "--patch_size", type=int, default=64)
    parser.add_argument("-w", "--weights_path", type=str, default=CKPT_PATH)
    parser.add_argument("-i", "--image_dir", type=str, default=TEST_IMG)

    args = parser.parse_args()

    main(args)