import tensorflow as tf



class CTCLoss:
    def __call__(self, y_true, y_pred, *args, **kwargs):
        batch_size = y_pred.shape[1]
        logit_length = tf.fill(dims=[batch_size], value=y_pred.shape[0])
        loss_value = tf.nn.ctc_loss(labels=y_true, logits=y_pred, label_length=None, logit_length=logit_length, blank_index=-1)
        loss_value = tf.math.reduce_mean(loss_value)
        return loss_value

