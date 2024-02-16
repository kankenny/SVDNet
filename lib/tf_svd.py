import tensorflow as tf

from util.constants import HRSTC

# from util.timer_deco import timer


"""
Abadi, M., et al. (2015). TensorFlow:
Large-scale machine learning on heterogeneous systems
[Software]. tensorflow.org.
"""


@tf.function
def compress_image_with_energy(image, energy_factor=HRSTC["MAX_ENERGY_FACTOR"]):
    # Returns a compressed image based on a desired energy factor
    image_rescaled = tf.convert_to_tensor(image)
    image_batched = tf.transpose(image_rescaled, [2, 0, 1])
    s, U, V = tf.linalg.svd(image_batched)

    # Extracting singular values
    props_rgb = tf.map_fn(lambda x: tf.cumsum(x) / tf.reduce_sum(x), s)
    props_rgb_mean = tf.reduce_mean(props_rgb, axis=0)

    # Find closest r that corresponds to the energy factor
    k = tf.argmin(tf.abs(props_rgb_mean - energy_factor)) + 1
    image_k = _tf_compress(image, k)
    return image_k


@tf.function
def _tf_compress(a, k):
    a = tf.convert_to_tensor(a)
    a_batched = tf.transpose(a, [2, 0, 1])  # tf.linallg.svd is channel first (c, h, w)
    s, U, V = tf.linalg.svd(a_batched, compute_uv=True)

    # Compute low-rank approximation of image across each RGB channel
    image_r = _rank_r_approx(s, U, V, k)
    image_r = tf.transpose(image_r, [1, 2, 0])  # (c, h, w) -> (h, w, c)

    return image_r


@tf.function
def _rank_r_approx(s, U, V, k):
    # Compute the matrices necessary for a rank-r approximation
    s_k, U_k, V_k = s[..., :k], U[..., :, :k], V[..., :, :k]

    # Compute the low-rank approximation
    A_k = tf.einsum("...s,...us,...vs->...uv", s_k, U_k, V_k)

    return A_k
