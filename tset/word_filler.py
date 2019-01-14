import tensorflow as tf
import numpy as np


class WordFiller(object):
    def __init__(self, session, data, dictionary, embeddings, embeddings_dim=128, vocabulary_size=50000,
                 hidden_state_dim=128,
                 multiRNNCell_size=100,
                 fully_connected_layer_dim=128,
                 projection_dim=None, drop_out_prob=0.6,
                 device="cpu:0",
                 construct_model=True):

        # data - list of codes (integers from 0 to vocabulary_size-1).
        #   This is the original text but words are replaced by their codes
        self.data = data
        # dictionary - map of words(strings) to their codes(integers)
        self.dictionary = dictionary

        self.embeddings = embeddings
        self.embeddings_dim = embeddings_dim

        self.vocabulary_size = vocabulary_size
        self.multiRNNCell_size = multiRNNCell_size
        self.fully_connected_layer_dim = fully_connected_layer_dim

        self.session = session
        self.device = device
        # batch_genrator variables
        self.index = 0
        self.batch_size = 64

        self.seq_length = 100

        self.batch_len = len(self.data) // self.batch_size

        self.batches = (np.asarray(self.data[:self.batch_len * self.batch_size], dtype=np.int32)) \
            .reshape(self.batch_size, self.batch_len)
        # cell configuration data
        self.hidden_state_dim = hidden_state_dim
        self.projection_dim = projection_dim
        self.drop_out_prob = drop_out_prob
        self.number_of_steps = 100
        # Model
        if construct_model:
            self.construct_model()

    def construct_model(self):
        with tf.device(self.device):
            self.input = tf.placeholder(tf.int32, [self.batch_size, self.seq_length])
            self.target = tf.placeholder(tf.int32, [self.batch_size, ])
            self.embedding_init = tf.placeholder(tf.float32, [self.vocabulary_size, self.embeddings_dim])
            self.embedded_inputs = tf.nn.embedding_lookup(self.embedding_init, self.input)
            self.embedded_target = tf.nn.embedding_lookup(self.embedding_init, self.target)

    def initialize(self):
        self.session.run(tf.global_variables_initializer())

    def process_data(self):
        return self.data, self.dictionary, self.embeddings

    # returns next training batch
    def next_batch(self):
        # mind to don't ask for indexes greater that len(data) - self.batch_size
        data_batch = np.copy(self.batches[:, self.index:self.index + self.seq_length])
        label_batch = np.copy(self.batches[:, self.index + self.seq_length])
        self.index = self.index + 1
        # print(data_batch)
        # print(data_batch.shape)
        # print(label_batch.shape)
        return data_batch, label_batch

    # with each call, a LSTMCell with desired configuration data is returned
    def cell(self, j):
        # todo concatenate with j
        name = "Cell "
        # print(type(name))
        cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_state_dim, num_proj=self.projection_dim, forget_bias=1.0)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.drop_out_prob)
        return cell

    def multiRNNCell(self):
        return tf.nn.rnn_cell.MultiRNNCell([self.cell(j=j) for j in range(self.multiRNNCell_size)])

    # creates a new RNN layer, returns @Param 1 : output @Param 2 : final state
    def RNN(self):
        print(self.embedded_inputs.shape)
        return tf.nn.dynamic_rnn(cell=self.multiRNNCell(), inputs=self.embedded_inputs, dtype=tf.float32)

    # creates and returns a fully-connected layer for classification
    def fully_connected(self, rnn_output):
        return tf.layers.dense(inputs=rnn_output, units=self.fully_connected_layer_dim)

    # computes and returns
    def softmax_loss(self, rnn_output, target):
        return tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.fully_connected(rnn_output=rnn_output),
                                                       labels=target))

    def train_model(self, epochs=200, learning_rate=0.5):
        """code here"""
        """Make an object with proper values and run sess"""
        loss_ = 0
        with tf.device(self.device):
            with tf.Session.as_default(self.session):
                output, final_state = self.RNN()
                soft_max_loss = self.softmax_loss(output[:, self.seq_length - 1, :], target=self.embedded_target)
                self.session.run(tf.global_variables_initializer())
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(soft_max_loss)
                for step in range(epochs):
                    """do some trainish thing"""
                    for j in range(1, self.batch_len - self.seq_length):
                        data_batch, label_batch = self.next_batch()
                        feed_dict = {self.target: label_batch, self.input: data_batch, self.embedding_init: self.embeddings}
                        loss, _ = self.session.run([soft_max_loss, optimizer], feed_dict=feed_dict)
                        print('Average loss at step ', step, ': ', loss)