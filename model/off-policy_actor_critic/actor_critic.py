import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tf_utils


class offpolicy_actor_critic():

    def __init__(
        self,
        lr_v=0.01,
        lr_w=0.01,
        lr_u=0.01,
        scope='offpol_a3c',
    ):
        self.lr_v = lr_v
        self.lr_w = lr_w
        self.lr_u = lr_u
        self.scope = scope

    def build_graph(self, data_samples):
