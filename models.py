import pickle

from simple_seq2seq import *
import tensorflow as tf
import numpy as np
from random import randint


class Model1(object):
    def __init__(self, session, data, dictionary, embeddings, embeddings_dim=768, vocabulary_size=30522,
                 num_units=512,
                 device="cpu:0",
                 construct_model=True):

        self.vocabulary_size = vocabulary_size
        self.dictionary = dictionary

        self.embeddings = embeddings
        self.embeddings_dim = embeddings_dim

        self.session = session
        self.device = device

        self.max_seq_length = 30
        # batch_genrator variables
        self.index = 0
        self.batch_size = 64

        self.num_units = num_units

        # corpus
        self.input_sequences, self.target_sequences, self.corpus_size = self.data_preprocessing(data)

        # Model
        if construct_model:
            self.construct_model()

    def construct_model(self):
        with tf.device(self.device):
            # seq_lenght?
            self.inputs = tf.placeholder(tf.int32, [self.batch_size, self.max_seq_length])
            self.targets = tf.placeholder(tf.int32, [self.batch_size, self.max_seq_length])
            self.target_lengths = tf.placeholder(tf.int32, [self.batch_size])
            self.embedding_init = tf.placeholder(tf.float32, [self.vocabulary_size, self.embeddings_dim])
            self.embedded_inputs = tf.nn.embedding_lookup(self.embedding_init, self.inputs)
            self.embedded_targets = tf.nn.embedding_lookup(self.embedding_init, self.targets)

            self.encoder = SimpleLSTMEncoder(num_units=512, batch_size=64, depth=2)
            # for the purpose of weight sharing
            self.decoder = SimpleLSTMDecoder(num_units=512, cell=self.encoder.get_cell(), batch_size=64, depth=None)

    def initialize(self):
        self.session.run(tf.global_variables_initializer())

    def data_preprocessing(self, data):
        # TODO fix problem here. zero pad sequences with less than max_seq_lenght and concat results of each step of
        # target and input are seperated for the sake of generality. in this case it is more memory efficient to use
        #  from a single list
        # for-loop
        target_sequences, input_sequences = [[1]], [[2]]
        for i in range(len(data)):
            lenght = len(data[i])
            # last sequence of each list has no following sequence
            input_sequences += data[i][:lenght - 1]
            # first element of each list has no previous sequence
            target_sequences += data[i][1:]
        corpus_size = len(input_sequences)
        print(input_sequences[99])
        for i in range(corpus_size):
            if len(input_sequences[i]) > self.max_seq_length:
                input_sequences[i] = np.array(input_sequences[i][:self.max_seq_length])
            # zero padding
            elif len(input_sequences[i]) < self.max_seq_length:
                zeros_ = [0 for i in range(self.max_seq_length - len(input_sequences[i]))]
                input_sequences[i] = np.array(input_sequences[i] + zeros_)
            if len(target_sequences[i]) > self.max_seq_length:
                target_sequences[i] = np.array(target_sequences[i][:self.max_seq_length])
            # zero padding
            elif len(target_sequences[i]) < self.max_seq_length:
                zeros_ = [0 for i in range(self.max_seq_length - len(target_sequences[i]))]
                target_sequences[i] = np.array(target_sequences[i] + zeros_)
        input_sequences, target_sequences = np.array(input_sequences), np.array(target_sequences)
        print(input_sequences.shape)
        print(target_sequences.shape)
        return input_sequences, target_sequences, corpus_size

    def next_batch(self):
        global next_target_batch, next_input_batch, target_batch_lengths
        if self.index < len(self.input_sequences):
            next_input_batch = self.input_sequences[self.index:self.index + self.batch_size]
            next_target_batch = self.target_sequences[self.index:self.index + self.batch_size]
            target_batch_lengths = np.array(
                [next_target_batch[i].shape[0] for i in range(
                    next_target_batch.shape[0]
                )], dtype=np.int32)
            print(target_batch_lengths)
            self.index = self.index + 1
            print(target_batch_lengths)
        return next_input_batch, next_target_batch, target_batch_lengths

    # TODO separate model from train.
    def train(self):
        with tf.device(self.device):
            with tf.Session.as_default(self.session):
                output, final_state = self.encoder(self.embedded_inputs)
                # final_outputs, final_state, final_sequence_lengths = self.decoder({
                #     'mode': 'train',
                #     'initial_state': final_state,
                #     'input': self.embedded_targets,
                #     'input_lengths': self.target_lengths,
                #     'embeddings': None,
                #     'special_symbols': None
                # })
                print(type(self.target_lengths))
                final_outputs, final_state, final_sequence_lengths = self.decoder.call(
                    mode='train',
                    initial_state=final_state,
                    inputs=self.embedded_targets,
                    input_lengths=self.target_lengths
                )

                mask = tf.to_float(tf.sequence_mask(self.target_lengths))
                print(final_outputs.rnn_output)

                _loss = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(logits=final_outputs.rnn_output,
                                                                       targets=self.targets, weights=mask,
                                                                       average_across_timesteps=True,
                                                                       average_across_batch=True))
                # setting the lr
                optimizer = tf.train.AdamOptimizer(learning_rate=0.005, epsilon=0.01).minimize(_loss)
                self.session.run(tf.global_variables_initializer())

                for step in range(self.corpus_size - self.batch_size - 1):
                    next_input_batch, next_target_batch, target_batch_lengths = self.next_batch()
                    print(type(self.targets))
                    feed_dict = {self.targets: next_target_batch, self.inputs: next_input_batch,
                                 self.target_lengths: target_batch_lengths, self.embedding_init: self.embeddings}
                    loss, _ = self.session.run([_loss, optimizer], feed_dict=feed_dict)
                    print('Average loss at step ', step, ': ', loss)


# with open("stories.pkl", 'rb') as file:
#     data = pickle.load(file)
#     file.close()

data = [[[k for k in range(randint(0, j))] for j in range(randint(0, i))] for i in range(100)]
with open('embeddings_table.pkl', 'rb') as file:
    embeddings = pickle.load(file)
    file.close()
with open('dictionary.pkl', 'rb') as file:
    dictionary = pickle.load(file)
    file.close()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
model1 = Model1(session=sess, data=data, dictionary=dictionary, embeddings=embeddings)
model1.train()
