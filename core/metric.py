import tensorflow as tf


class Accuracy:
    def __init__(self):
        pass

    def __call__(self, decoded_text, true_label, *args, **kwargs):
        """

        :param decoded_text: tensor, shape: (batch_size, max_decoded_length)
        :param true_label: list
        :param args:
        :param kwargs:
        :return: the accuracy of batch prediction
        """
        total_num_char, num_each_dim_list = self.__dim_of_list(x=true_label)
        decoded_text = tf.cast(x=decoded_text, dtype=tf.dtypes.int32)
        num_correct_char = 0
        for i in range(len(num_each_dim_list)):
            for j in range(num_each_dim_list[i]):
                if decoded_text[i, j] == true_label[i][j]:
                    num_correct_char += 1
        batch_accuracy = num_correct_char / total_num_char
        return batch_accuracy

    @staticmethod
    def __dim_of_list(x):
        total_num = 0
        num_each_dim = []
        for item in x:
            count = 0
            for _ in item:
                total_num += 1
                count += 1
            num_each_dim.append(count)
        return total_num, num_each_dim
