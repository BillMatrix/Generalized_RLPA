import tensorflow as tf
import numpy as np
import tflearn
import argparse
import pprint as pp

from replay_buffer import ReplayBuffer


class CriticNetwork(object):

    def create_critic_network(self, hidden_size):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(inputs, hidden_size)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        t1 = tflearn.fully_connected(net, hidden_size)
        t2 = tflearn.fully_connected(action, hidden_size)

        net = tflearn.activation(
            tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b,
            activation='relu',
        )

        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out
