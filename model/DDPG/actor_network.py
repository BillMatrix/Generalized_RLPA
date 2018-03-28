import tensorflow as tf
import numpy as np
import tflearn
import argparse
import pprint as pp

from replay_buffer import ReplayBuffer


class ActorNetwork(object):
    '''
        Given the state, the ActorNetwork's output is the deterministic policy
    '''

    def __init__(
        self, sess, s_dim, a_dim, action_bound,
        lr, tau, batch_size, hidden_size
    ):
        self.sess = sess
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.action_bound = action_bound
        self.lr = lr
        self.tau = tau
        self.batch_size = batch_size
        self.hidden_size = hidden_size

        self.inputs, self.scaled_out = self.create_actor_network(
            self.hidden_size
        )

        self.params = tf.trainable_variables()

        # By Deep Deterministic Policy Gradient
        self.target_inputs, self.target_scaled_out \
            = self.create_actor_network(self.hidden_size)

        self.target_params = tf.trainable_variables()[len(self.params):]

        self.update_target_params \
            = [self.target_params[]]

    def create_actor_network(self, hidden_size):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, hidden_size)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(inputs, hidden_size)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(
            net,
            self.a_dim,
            activation='tanh',
            weights_init=w_init,
        )
        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, scaled_out
