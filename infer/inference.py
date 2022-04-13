# Library Imports
import tensorflow as tf
import argparse

# Custom Imports

def main():
    print(f"hello world")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # TODO: Configure arguments

    parser.add_argument("-bs", "--batch_size", type=int, default=1)

    args = parser.parse_args()

    main(args)


