import tensorflow as tf

from configuration import padding_value


class CTCLoss(object):
    def __init__(self):
        super(CTCLoss, self).__init__()

    def __call__(self, y_true, y_pred, *args, **kwargs):
        batch_size = y_pred.shape[1]
        logit_length = tf.fill(dims=[batch_size], value=y_pred.shape[0])
        loss_value = tf.nn.ctc_loss(labels=y_true, logits=y_pred, label_length=None, logit_length=logit_length, blank_index=padding_value)
        loss_value = tf.math.reduce_mean(loss_value)
        return loss_value
