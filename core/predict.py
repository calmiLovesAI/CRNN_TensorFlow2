import tensorflow as tf



def predict_text(model_output, blank_index):
    batch_size = model_output.shape[1]
    sequence_length = tf.fill(dims=[batch_size], value=model_output.shape[0])
    decoded, neg_sum_logits = tf.nn.ctc_greedy_decoder(inputs=model_output,
                                                       sequence_length=sequence_length,
                                                       merge_repeated=True)
    dense_decoded = tf.sparse.to_dense(sp_input=decoded[0], default_value=blank_index)
    return dense_decoded