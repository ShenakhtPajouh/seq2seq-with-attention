import tensorflow as tf
def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob, source_vocab_size, encoding_embedding_size):
    embed = tf.contrib.layers.embed_sequence(rnn_inputs, vocab_size=source_vocab_size,
                                             embed_dim=encoding_embedding_size)
    stacked_cells = tf.contrib.rnn.MultiRNNCell(
        [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(rnn_size), keep_prob) for _ in range(num_layers)])
    outputs, state = tf.nn.dynamic_rnn(stacked_cells, embed, dtype=tf.float32)

    # return: tuple (RNN output, RNN state)
    # RNN State: LSTM Tuple ( c(hidden_state of every layer) and h (output per input) )
    return outputs, state


def process_decoder_input(target_data, target_vocab_to_int, batch_size):
    # get '<GO>' id
    go_id = target_vocab_to_int['<GO>']
    after_slice = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    after_concat = tf.concat([tf.fill([batch_size, 1], go_id), after_slice], 1)

    return after_concat


def decoding_layer_train(encoder_outputs, dec_cell, dec_embed_input, target_sequence_length, max_summary_length,
                         output_layer, keep_prob):
    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, output_keep_prob=keep_prob)

    attn_mech = tf.contrib.seq2seq.BahdanauAttention(num_units=rnn_size, memory=encoder_outputs,
                                                     memory_sequence_length=target_sequence_length)

    attn_cell = tf.contrib.seq2seq.AttentionWrapper(cell=dec_cell, attention_mechanism=attn_mech,
                                                    alignment_history=True)

    helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input, target_sequence_length)

    decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, helper, encoder_outputs, output_layer)

    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True,
                                                      maximum_iterations=max_summary_length)
    return outputs


def decoding_layer_infer(encoder_outputs, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id,
                         max_target_sequence_length, vocab_size, output_layer, batch_size, keep_prob):
    # Creating an inference process in decoding layer
    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, output_keep_prob=keep_prob)

    attn_mech = tf.contrib.seq2seq.BahdanauAttention(num_units=rnn_size, memory=ecnoder_outputs,
                                                     memory_sequence_length=max_target_sequence_length)

    attn_cell = tf.contrib.seq2seq.AttentionWrapper(cell=dec_cell, attention_mechanism=attn_mech,
                                                    alignment_history=True)

    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings, tf.fill([batch_size], start_of_sequence_id),
                                                      end_of_sequence_id)

    decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, helper, encoder_outputs, output_layer)
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True,
                                                      maximum_iterations=max_target_sequence_length)
    return outputs


def decoding_layer(dec_input, encoder_state, target_sequence_length, max_target_sequence_length,
                   rnn_size, num_layers, target_vocab_to_int, target_vocab_size, batch_size, keep_prob,
                   decoding_embedding_size):
    target_vocab_size = len(target_vocab_to_int)
    dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

    cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(num_layers)])

    with tf.variable_scope("decode"):
        output_layer = tf.layers.Dense(target_vocab_size)
        train_output = decoding_layer_train(encoder_state, cells, dec_embed_input, target_sequence_length,
                                            max_target_sequence_length, output_layer, keep_prob)

    with tf.variable_scope("decode", reuse=True):
        infer_output = decoding_layer_infer(encoder_state, cells, dec_embeddings, target_vocab_to_int['<GO>'],
                                            target_vocab_to_int['<EOS>'], max_target_sequence_length, target_vocab_size,
                                            output_layer, batch_size, keep_prob)

    return (train_output, infer_output)


def seq2seq_model(input_data, target_data, keep_prob, batch_size, target_sequence_length, max_target_sentence_length,
                  source_vocab_size, target_vocab_size, enc_embedding_size, dec_embedding_size, rnn_size, num_layers,
                  target_vocab_to_int):
    enc_states, enc_outputs = encoding_layer(input_data, rnn_size, num_layers, keep_prob, source_vocab_size,
                                             enc_embedding_size)

    dec_input = process_decoder_input(target_data, target_vocab_to_int, batch_size)

    train_output, infer_output = decoding_layer(dec_input, enc_states, target_sequence_length,
                                                max_target_sentence_length,
                                                rnn_size, num_layers, target_vocab_to_int, target_vocab_size,
                                                batch_size,
                                                keep_prob, dec_embedding_size)

    return train_output, infer_output, enc_outputs, enc_states


to, ifo, eno, ens = seq2seq_model((inputs), targets, keep_prob, batch_size, target_sequence_length, max_target_length,
                                  len(source_vocab_to_int), len(target_vocab_to_int), embed_size, dec_embed_size,
                                  rnn_size, num_layers,
                                  target_vocab_to_int)


# Attention Wrapper: adds the attention mechanism to the cell
attn_cell = wrapper.AttentionWrapper(
    cell = lstm_cell_decoder,# Instance of RNNCell
    attention_mechanism = attn_mech, # Instance of AttentionMechanism
    attention_size = embedding_dim, # Int, depth of attention (output) tensor
    attention_history=False, # whether to store history in final output
    name="attention_wrapper")


# Decoder setup
decoder = tf.contrib.seq2seq.BasicDecoder(
          cell = lstm_cell_decoder,
          helper = helper, # A Helper instance
          initial_state = encoder_state, # initial state of decoder
          output_layer = None) # instance of tf.layers.Layer, like Dense

# Perform dynamic decoding with decoder object
outputs, final_state = tf.contrib.seq2seq.dynamic_decode(decoder)