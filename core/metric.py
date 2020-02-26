import tensorflow as tf
from configuration import char2index_map


class Accuracy(object):
    def __init__(self):
        super(Accuracy, self).__init__()
        self.default_value = len(char2index_map)

    def __call__(self, decoded_text, true_label, *args, **kwargs):
        """

        :param decoded_text: tensor, shape: (batch_size, max_decoded_length)
        :param true_label: tensor, shape: (batch_size, max_label_seq_length)
        :param args:
        :param kwargs:
        :return: the accuracy of a batch prediction
        """
        decoded_text = tf.cast(x=decoded_text, dtype=tf.dtypes.int32)
        batch_size = decoded_text.shape[0]
        max_decoded_length = decoded_text.shape[1]
        assert max_decoded_length == true_label.shape[1]
        num_correct_char = 0
        for i in range(batch_size):
            for j in range(max_decoded_length):
                if decoded_text[i, j] == true_label[i, j]:
                    num_correct_char += 1
        total_num_char = batch_size * max_decoded_length
        batch_accuracy = num_correct_char / total_num_char
        return batch_accuracy
