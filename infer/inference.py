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

    patch_distance = int(patch_size / 2)

    # TODO: Load model + weights

    model = tf.keras.models.load_model(weights_path, compile=False)





    # TODO: Load image
    img = nib.load(img_dir).get_fdata() / 2000.
    img = img[:,:,0:512] # Ask Kyle about how to handle this... original size is (512,512,654)
                         # How do we handle the excess z axis? z coordinate varies from img to img
    row_dim, column_dim, slice_dim  = img.shape[0], img.shape[1], img.shape[2]
    # print(img.shape)

    # TODO: Work with only 1 patch.
    # num_row_patchs = int(row_dim / patch_size)
    # num_col_patchs = int(column_dim / patch_size)
    # num_slice_patchs = int(slice_dim / patch_size)

    first_patch_midpoint = (patch_distance, patch_distance, patch_distance)

    patch_mid_row = patch_distance
    patch_mid_col = patch_distance
    patch_mid_slice = patch_distance

    rows_patchs = []
    col_patchs = []
    slice_patches = []
    tot_patchs = []
    # TODO: Iterate through patches

    # Slices   
    while patch_mid_slice < slice_dim:
        # Columns
        while patch_mid_col < column_dim:
            # Rows
            while patch_mid_row < row_dim:
                row_patch = img[(patch_mid_row-patch_distance):(patch_mid_row+patch_distance), \
                        (patch_mid_col-patch_distance):(patch_mid_col+patch_distance), \
                        (patch_mid_slice-patch_distance):(patch_mid_slice+patch_distance)]
                expanded_row_patch = np.expand_dims(np.expand_dims(row_patch, -1), 0)
                print(expanded_row_patch.shape)

                rows_patchs.append(expanded_row_patch)
                tot_patchs.append(expanded_row_patch)
                patch_mid_row += patch_size

            col_patchs.append(rows_patchs)
            patch_mid_col += patch_size

        # col_patchs = np.array(col_patchs) Ask Kyle about ways to store this?
        slice_patches.append(col_patchs)
        patch_mid_slice += patch_size


    print(len(tot_patchs))
    print(np.shape(col_patchs))
    print(len(rows_patchs))
    print(len(col_patchs))
    print(len(col_patchs[0]))



    

    # TODO: Iterate through image given patch size (start with 64)

    # TODO: Stitch inferred images together
    inferred_img = model.predict(rows_patchs[0])
    print(inferred_img.shape)



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