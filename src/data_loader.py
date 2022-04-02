import tensorflow as tf
import os
import numpy as np
import nibabel as nib

def get_ids(path):
    with open(path, "r") as file:
        ids = [line.rstrip() for line in file]
    return ids

def data_generator(ids, data_dir, batch_size=1, patch_size=64):
    while True:
        patch_distance = int(patch_size / 2)
    
        for patient_id in ids:
        # Read in image

            img_dir = data_dir + "/ct/ct_patients" + str(patient_id) + ".nii"
            mask_dir = data_dir + "/labels/aw_" + str(patient_id) + ".nii"

            img = nib.load(img_dir).get_fdata() / 2000.
            mask = nib.load(mask_dir).get_fdata()

            row_dim = img.shape[0]
            column_dim = img.shape[1]
            slice_dim = img.shape[2]

            # Define mask to randomly find patch midpoint within bounds of patch distance
            working_mask = mask[patch_distance:(row_dim-patch_distance), \
                            patch_distance:(column_dim-patch_distance), \
                            patch_distance:(slice_dim-patch_distance)]

            # Iterate through slices
            slices = np.sum(working_mask, axis=(0,1))

            # Find slices which contain mask
            slices_with_airway = [i for i in range(0, len(slices)) if slices[i] > 0] # Potentially tune this 0 hyper parameter

            # Randomly find slice
            random_slice_index = np.random.randint(0, len(slices_with_airway)) # in small 3d... 
            random_slice = slices_with_airway[random_slice_index]

            # Find columns which contain mask
            row_column_locations = np.where(working_mask[:,:,random_slice] > 0)
            num_mask_voxels = len(row_column_locations[0]) 

            # Randomly find row + column
            random_row_column_index = np.random.randint(0, num_mask_voxels)
            random_row = row_column_locations[0][random_row_column_index]
            random_column = row_column_locations[1][random_row_column_index]

            # Adjust voxel coordinates for larger image
            adjusted_row = random_row + patch_distance
            adjusted_column = random_column + patch_distance
            adjusted_slice = random_slice + patch_distance

            # Final voxel coordinates       
            voxel_coordinates = (adjusted_row, adjusted_column, adjusted_slice)

            # Index based on specified patch distance
            indexed_patch_img = img[(adjusted_row-patch_distance):(adjusted_row+patch_distance), \
                    (adjusted_column-patch_distance):(adjusted_column+patch_distance), \
                    (adjusted_slice-patch_distance):(adjusted_slice+patch_distance)]

            indexed_patch_label = mask[(adjusted_row-patch_distance):(adjusted_row+patch_distance), \
                            (adjusted_column-patch_distance):(adjusted_column+patch_distance), \
                            (adjusted_slice-patch_distance):(adjusted_slice+patch_distance)]

            # Add patient and channel
            patch_img = np.expand_dims(np.expand_dims(indexed_patch_img, -1), 0)
            patch_mask = np.expand_dims(np.expand_dims(indexed_patch_label, -1), 0)

            final_patch_img = tf.convert_to_tensor(patch_img)
            final_patch_mask = tf.convert_to_tensor(patch_mask)

        yield (final_patch_img, final_patch_mask)
