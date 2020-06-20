import tensorflow as tf

from configuration import Config
from core.utils import index_to_char


class Accuracy:
    def __init__(self, blank_index):
        self.idx2char_dict = Config.get_idx2char()
        self.blank_index = blank_index

    def __call__(self, decoded_text, true_label, *args, **kwargs):
        """

        :param decoded_text: tensor, shape: (batch_size, max_decoded_length)
        :param true_label: sparse tensor
        :param args:
        :param kwargs:
        :return: the accuracy of batch prediction
        """
        batch_size = decoded_text.shape[0]
        decoded_text = tf.cast(x=decoded_text, dtype=tf.int32)
        decoded_text = decoded_text.numpy()
        decoded_text = index_to_char(inputs=decoded_text, idx2char_dict=self.idx2char_dict, blank_index=self.blank_index)
        label = tf.sparse.to_dense(sp_input=true_label, default_value=self.blank_index).numpy()
        label = index_to_char(inputs=label, idx2char_dict=self.idx2char_dict, blank_index=self.blank_index)

        correct_num = 0
        for y_pred, y_true in zip(decoded_text, label):
            if y_pred == y_true:
                correct_num += 1

        accuracy = correct_num / batch_size
        return accuracy
