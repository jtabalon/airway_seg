import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

def model_checkpoint_callback(filepath):
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        save_best_only=True,
        monitor="loss"
    )