import tensorflow as tf


class SimpleLSTMEncoder(object):
    """
    Encodes the inputs using a LSTM
    """

    def __init__(self, num_units, batch_size, depth=1, dropout_probability=0.5):
        """
        not SimpleLSTMEncoder's problem!
        TODO: receive the problem somewhere else.
        """
        # self.dictionary = dictionary
        # self.vocabulary_size = vocabulary_size
        # self.embedding_lookup_table = embedding_lookup_table

        self.depth = depth
        self.dropout_probability = dropout_probability
        self.num_units = num_units
        self.batch_size = batch_size

    def cell(self):
        cell = tf.nn.rnn_cell.LSTMCell(num_units=self.num_units, use_peepholes=True, forget_bias=1.0)
        tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell
                                    .DropoutWrapper(cell, output_keep_prob=self.dropout_probability) for j in
                                     range(self.depth)])
        return cell

    def call(self, input_seq):
        """
        Description:

        Args:
            input_seq: a Tensor of shape [batch_size, seq_length, embedding_dim],
            each batch containing concatenated embeddings of tokens of a sentence.

        Returns:
            outputs: a Tensor shaped: [batch_size, max_time, cell.output_size].
        """
        outputs, state = tf.nn.dynamic_rnn(cell=self.cell(), inputs=input_seq, dtype=tf.float32,
                                           parallel_iterations=1024)
        return outputs, state


class SimpleLSTMDecoder(object):
    """
    Simply uses the last hidden state of encoder to create decoded sequence.
    """

    def __init__(self, num_units, batch_size, depth=1, dropout_probability=0.5):
        # self.dictionary = dictionary
        # self.vocabulary_size = vocabulary_size
        # self.embedding_lookup_table = embedding_lookup_table

        self.depth = depth
        self.dropout_probability = dropout_probability
        self.num_units = num_units
        self.batch_size = batch_size

    def cell(self):
        cell = tf.nn.rnn_cell.LSTMCell(num_units=self.num_units, use_peepholes=True, forget_bias=1.0)
        tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell
                                    .DropoutWrapper(cell, output_keep_prob=self.dropout_probability) for j in
                                     range(self.depth)])
        return cell

    def call(self, mode, initial_state, input, input_lengths, embeddings, special_symbols):
        """
        Description:

        Args:
            mode: A string, can either be "train" or infer
            initial_state: initial state of decoder
            input: (in case of mode=="train") A Tensor of shape [batch_size,
            input_lengths: (in case of mode=="train") A vector. Lengths of each sequence in a batch
            embeddings: (in case of mode=="train") Embedding look-up table of shape [vocabulary_size, embedding_dim]
            special_symbols: (in case of mode=="infer") A tuple of form (start_symbol, end_symbol) with dtype tf.int32,
             place of mentioned symbols in embeddings

        Returns:
            outputs: a Tensor shaped: [batch_size, max_time, cell.output_size].
        """

        cell = self.cell()
        start_symbol, end_symbol = special_symbols[0], special_symbols[1]
        if mode == "train":
            helper = tf.contrib.seq2seq.TrainingHelper(
                input=input,
                sequence_length=input_lengths)
        elif mode == "infer":
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding=embeddings,
                start_tokens=tf.tile([start_symbol], [self.batch_size]),
                end_token=end_symbol)

        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=cell,
            helper=helper,
            initial_state=initial_state)
        outputs, state = tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder,
            impute_finished=True,
            maximum_iterations=40)
        return outputs, state


class SimpleLSTMDecoderWithAttention(object):
    """
    Simply uses the last hidden state of encoder to create decoded
    sequence with attention on its own past hidden states.
    TODO: IMPALEMENT!
    """
