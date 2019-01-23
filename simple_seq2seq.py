import tensorflow as tf
from tensorflow.python.layers.core import Dense


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
        self.cell = self.cell()

    def cell(self):
        cell = tf.nn.rnn_cell.LSTMCell(num_units=self.num_units, use_peepholes=True, forget_bias=1.0)
        tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell
                                    .DropoutWrapper(cell, output_keep_prob=self.dropout_probability) for j in
                                     range(self.depth)])
        return cell

    def get_cell(self):
        return self.cell

    def __call__(self, input_seq):
        """
        Description:

        Args:
            input_seq: a Tensor of shape [batch_size, seq_length, embedding_dim],
            each batch containing concatenated embeddings of tokens of a sentence.

        Returns:
            outputs: a Tensor shaped: [batch_size, max_time, cell.output_size].
        """
        outputs, state = tf.nn.dynamic_rnn(cell=self.cell, inputs=input_seq, dtype=tf.float32,
                                           parallel_iterations=1024)
        return outputs, state


class SimpleLSTMDecoder(object):
    """
    Simply uses the last hidden state of encoder to create decoded sequence.
    """

    def __init__(self, num_units, batch_size, cell=None, depth=1, dropout_probability=0.5):
        self.depth = depth
        self.dropout_probability = dropout_probability
        self.num_units = num_units
        self.batch_size = batch_size
        if cell is None:
            self.cell = self.cell()
        else:
            self.cell = cell

    def cell(self):
        cell = tf.nn.rnn_cell.LSTMCell(num_units=self.num_units, use_peepholes=True, forget_bias=1.0)
        tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell
                                    .DropoutWrapper(cell, output_keep_prob=self.dropout_probability) for j in
                                     range(self.depth)])
        return cell

    def call(self, mode, initial_state, inputs, input_lengths#, embeddings, special_symbols
     ):
        """
        Description:

        Args:
            mode: A string, can either be "train" or infer
            initial_state: initial state of decoder
            input: (in case of mode=="train") A Tensor of shape [batch_size, TODO: ?
            input_lengths: (in case of mode=="train") A vector. Lengths of each sequence in a batch
            embeddings: (in case of mode=="infer") Embedding look-up table of shape [vocabulary_size, embedding_dim]
            special_symbols: (in case of mode=="infer") A tuple of form (start_symbol, end_symbol) with dtype tf.int32,
             place of mentioned symbols in embeddings

        Returns:
            outputs: a Tensor shaped: [batch_size, max_time, cell.output_size].
        """

        output_logits = Dense(30522, use_bias=False)
        global helper
        if mode == "train":
            helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=inputs,
                sequence_length=input_lengths)
        # elif mode == "infer":
            # helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            #     embedding=embeddings,
            #     start_tokens=tf.tile([special_symbols[0]], [self.batch_size]),
            #     end_token=special_symbols[1])

        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=self.cell,
            helper=helper,
            initial_state=initial_state,
            output_layer=output_logits)
        final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder,
            impute_finished=True,
            maximum_iterations=40)
        return final_outputs, final_state, final_sequence_lengths

    def __call__(self, *args, **kwargs):
        self.call(**kwargs)


class LSTMDecoderWithAttention(object):
    """
    Simply uses the last hidden state of encoder to create decoded
    sequence with attention on its own past hidden states.
    """

    def __init__(self, num_units, batch_size, depth=1, dropout_probability=0.5):
        self.depth = depth
        self.dropout_probability = dropout_probability
        self.num_units = num_units
        self.batch_size = batch_size

    def cell(self):
        cell = tf.nn.rnn_cell.LSTMCell(
            num_units=self.num_units, use_peepholes=True, forget_bias=1.0)
        tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.DropoutWrapper(
                cell, output_keep_prob=self.dropout_probability) for j in range(self.depth)])
        return cell

    def call(self, mode, initial_state, encoder_memory, inputs, input_lengths, embeddings, special_symbols):
        """
        Description:

        Args:
            mode: A string, can either be "train" or infer
            initial_state: initial state of decoder
            encoder_memory: Hidden states of encoder of shape [batch_size, max_time, cell.output_size]
            input: (in case of mode=="train") A Tensor of shape [batch_size, TODO: ?
            input_lengths: (in case of mode=="train") A vector. Lengths of each sequence in a batch
            embeddings: (in case of mode=="infer") Embedding look-up table of shape [vocabulary_size, embedding_dim]
            special_symbols: (in case of mode=="infer") A tuple of form (start_symbol, end_symbol) with dtype tf.int32,
             place of mentioned symbols in embeddings

        Returns:
            outputs: a Tensor shaped: [batch_size, max_time, cell.output_size].
        """

        global helper
        cell = self.cell()
        if mode == "train":
            helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=inputs,
                sequence_length=input_lengths)
        elif mode == "infer":
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding=embeddings,
                start_tokens=tf.tile([special_symbols[0]], [self.batch_size]),
                end_token=special_symbols[1])

        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units=self.num_units,
            memory=encoder_memory,
            normalize=True)

        cell = tf.contrib.seq2seq.AttentionWrapper(
            cell=cell,
            attention_mechanism=attention_mechanism,
            alignment_history=False)

        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=cell,
            helper=helper,
            initial_state=initial_state)

        final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder,
            impute_finished=True,
            maximum_iterations=40)
        return final_outputs, final_state, final_sequence_lengths


class TrainingHelperWithMemory(tf.contrib.seq2seq.TrainingHelper):
    """
    states: must be initialized with last state of encoder
    """

    def __init__(self, inputs, sequence_length, states, time_major=False, name=None):
        self.states = states
        super(tf.contrib.seq2seq.TrainingHelper, self).__init__(inputs, sequence_length, time_major, name)

    def next_inputs(self, time, outputs, state, name=None, **unused_kwargs):
        self.states = tf.concat(values=[self.states, state], axis=0)
        return super(tf.contrib.seq2seq.TrainingHelper, self).next_inputs(time, outputs, name, **unused_kwargs)


class LSTMDecoderWithSelfAttention(object):
    """
    Simply uses the last hidden state of encoder to create decoded
    sequence with attention on its own past hidden states.
    """

    def __init__(self, num_units, batch_size, depth=1, dropout_probability=0.5):
        self.depth = depth
        self.dropout_probability = dropout_probability
        self.num_units = num_units
        self.batch_size = batch_size

    def cell(self):
        cell = tf.nn.rnn_cell.LSTMCell(
            num_units=self.num_units, use_peepholes=True, forget_bias=1.0)
        tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.DropoutWrapper(
                cell, output_keep_prob=self.dropout_probability) for j in range(self.depth)])
        return cell

    def call(self, mode, initial_state, encoder_memory, inputs, input_lengths, embeddings, special_symbols,
             self_attention=True):
        """
        Description:

        Args:
            mode: A string, can either be "train" or infer
            initial_state: initial state of decoder
            encoder_memory: Hidden states of encoder of shape [batch_size, max_time, cell.output_size]
            input: (in case of mode=="train") A Tensor of shape [batch_size, TODO: ?
            input_lengths: (in case of mode=="train") A vector. Lengths of each sequence in a batch
            embeddings: (in case of mode=="infer") Embedding look-up table of shape [vocabulary_size, embedding_dim]
            special_symbols: (in case of mode=="infer") A tuple of form (start_symbol, end_symbol) with dtype tf.int32,
             place of mentioned symbols in embeddings
             self_attention: A boolean, if true uses Bahdanau Attention on previous states of decoder


        Returns:
            outputs: a Tensor shaped: [batch_size, max_time, cell.output_size].
        """

        global helper
        cell = self.cell()
        if mode == "train":
            helper = TrainingHelperWithMemory(
                inputs=inputs,
                sequence_length=input_lengths)
        elif mode == "infer":
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding=embeddings,
                start_tokens=tf.tile([special_symbols[0]], [self.batch_size]),
                end_token=special_symbols[1])
        """Attention on encoder states"""
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units=self.num_units,
            memory=encoder_memory,
            normalize=True)

        cell = tf.contrib.seq2seq.AttentionWrapper(
            cell=cell,
            attention_mechanism=attention_mechanism,
            alignment_history=False)

        """Self-attention on previous decoder states"""
        self_attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units=self.num_units,
            memory=helper.states,
            normalize=True)

        cell = tf.contrib.seq2seq.AttentionWrapper(
            cell=cell,
            attention_mechanism=self_attention_mechanism,
            alignment_history=False)

        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=cell,
            helper=helper,
            initial_state=initial_state)

        final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder,
            impute_finished=True,
            maximum_iterations=40)
        return final_outputs, final_state, final_sequence_lengths
