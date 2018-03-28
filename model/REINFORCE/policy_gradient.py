import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tf_utils


class REINFORCE():

    def __init__(
        self,
        lr=0.5,
        state_size=4,
        action_size=2,
        n_hidden_1=20,
        n_hidden_2=20,
        scope="reinforce",
    ):
        self.lr = lr
        self.state_size = state_size
        self.action_size = action_size
        self.total_steps = 0
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2
        self.scope = scope

        self._build_policy_net()

    def build_policy_net(self):
        with tf.variable_scope(self.scope):
            self.state_input = tf.placeholder(
                tf.float32,
                [None, self.state_size]
            )
