import tensorflow as tf
from configuration import padding_value


def predict_text(model_output):
    batch_size = model_output.shape[1]
    sequence_length = tf.fill(dims=[batch_size], value=model_output.shape[0])
    decoded, neg_sum_logits = tf.nn.ctc_greedy_decoder(inputs=model_output,
                                                       sequence_length=sequence_length,
                                                       merge_repeated=False)
    dense_decoded = tf.sparse.to_dense(sp_input=decoded[0], default_value=padding_value)
    return dense_decoded