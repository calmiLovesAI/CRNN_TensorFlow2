import tensorflow as tf


class CTCLoss(object):
    def __init__(self):
        super(CTCLoss, self).__init__()

    def __call__(self, y_true, y_pred, *args, **kwargs):
        batch_size = y_pred.shape[1]
        logit_length = tf.fill(dims=[batch_size], value=y_pred.shape[0])
        loss_value = tf.nn.ctc_loss(labels=y_true, logits=y_pred, label_length=self.__get_label_length(y_true), logit_length=logit_length)
        loss_value = tf.math.reduce_mean(loss_value)
        return loss_value

    @staticmethod
    def __get_label_length(label):
        batch_size, max_label_seq_length = label.shape
        label_length_list = []
        for i in range(batch_size):
            count = 0
            for j in range(max_label_seq_length):
                if label[i, j] != -1:
                    count += 1
                else:
                    break
            label_length_list.append(count)
        label_length = tf.convert_to_tensor(value=label_length_list, dtype=tf.dtypes.int32)
        return label_length

