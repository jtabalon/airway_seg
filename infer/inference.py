# Library Imports
import argparse
import numpy as np
import tensorflow as tf

# Custom Imports

def main():
    # TODO: Load model + weights


    # TODO: Load image


    # TODO: Iterate through image given patch size (start with 64)

    # TODO: Stitch inferred images together


    # TODO: Calculate dice metric.

    print(f"hello world")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # TODO: Configure arguments

    parser.add_argument("-p", "--patch_size", type=int, default=64)
    # parser.add_argument("-w", "--weights_path", type=str, default=)
    # parser.add_argument("-i", "--image_dir", type=str, default=)

    args = parser.parse_args()

    main(args)