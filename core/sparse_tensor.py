import tensorflow as tf


class GenerateSparseTensor:
    def __init__(self):
        pass

    def __call__(self, indices_list, values_list, dtype, dense_shape, *args, **kwargs):
        indices = tf.convert_to_tensor(value=indices_list, dtype=tf.dtypes.int64)
        values = tf.convert_to_tensor(value=values_list, dtype=dtype)
        dense_shape = tf.convert_to_tensor(value=dense_shape, dtype=tf.dtypes.int64)
        return tf.sparse.SparseTensor(indices, values, dense_shape)