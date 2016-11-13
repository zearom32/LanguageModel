from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import datetime
import numpy as np
import tensorflow as tf

import reader
import codecs
import os
from ptb_word_lm import PTBInput, PTBModel, TestConfig, inference, inference_file



class LanguageModel(object):

    def __init__(self, model_path="model"):
        self._model_path = model_path
        with tf.Graph().as_default():
            config = TestConfig()
            initializer = tf.random_uniform_initializer(-config.init_scale,
                                                        config.init_scale)
            with tf.name_scope("Test"):
                with tf.variable_scope("Model", reuse=None, initializer=initializer):
                    mtest = PTBModel(is_training=False, config=config)
            init_op = tf.initialize_all_variables()
            saver = tf.train.Saver()
            session =tf.Session()
            session.run(init_op)
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(session, ckpt.model_checkpoint_path)
                self._session = session
                self._config = config
                self._model = mtest
            else:
                print("No checkpoint found")
                return


    def test(self, x):
        return inference(self._session, self._model, self._config, x)

    def test_file(self, filepath):
        return inference_file(self._session, self._model, self._config, filename=filepath)


    @property
    def config(self):
        return self._config

    @property
    def model(self):
        return self._model

    @property
    def session(self):
        return self._session

    @property
    def model_path(self):
        return self._model_path
