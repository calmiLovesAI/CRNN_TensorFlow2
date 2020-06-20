import tensorflow as tf

from configuration import Config


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
        decoded_text = self.__index_to_char(inputs=decoded_text)
        label = tf.sparse.to_dense(sp_input=true_label, default_value=self.blank_index).numpy()
        label = self.__index_to_char(inputs=label)

        correct_num = 0
        for y_pred, y_true in zip(decoded_text, label):
            if y_pred == y_true:
                correct_num += 1

        accuracy = correct_num / batch_size
        return accuracy

    def __index_to_char(self, inputs, merge_repeated=False):
        chars = []
        for item in inputs:
            text = ""
            pre_char = -1
            for current_char in item:
                if merge_repeated:
                    if current_char == pre_char:
                        continue
                pre_char = current_char
                if current_char == self.blank_index:
                    continue
                text += self.idx2char_dict[current_char]
            chars.append(text)
        return chars

