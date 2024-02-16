import tensorflow as tf


class MinMaxNormalization(tf.keras.layers.Layer):
    def __init__(self, min_value=0.0, max_value=1.0):
        super(MinMaxNormalization, self).__init__()
        self.min_value = min_value
        self.max_value = max_value

    def call(self, inputs):
        min_val = tf.reduce_min(inputs)
        max_val = tf.reduce_max(inputs)
        normalized = (inputs - min_val) / (max_val - min_val)
        normalized = normalized * (self.max_value - self.min_value) + self.min_value
        return normalized
