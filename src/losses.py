import tensorflow as tf

def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return 1 - numerator / denominator


# TODO: Add Dice-BCE (Binary Cross Entropy)