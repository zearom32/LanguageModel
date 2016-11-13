# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import codecs
import itertools
import random
import numpy as np

import tensorflow as tf
from ch import ch_word_to_id, ch_id_to_word,ch_list


def _get_id(word, word_to_id):
    if word == "\n":
        return word_to_id['<eos>']
    elif word in word_to_id:
        return word_to_id[word]
    else:
        return word_to_id['<unk>']

def _data_to_word_id(data, word_to_id):
    return [_get_id(word, word_to_id) for word in data]

def _file_to_word_ids(filename, word_to_id):
    with codecs.open(filename, "r") as f:
        data = list(f.read().decode("utf-8").replace(" ",""))
        return _data_to_word_id(data, word_to_id)

def inference_raw_data(data_path=None, filename="inference.txt"):
    raw_data_path = os.path.join(data_path, filename)
    raw_data = _file_to_word_ids(raw_data_path, ch_word_to_id)
    return raw_data

def line_to_data(line):
    tmp = list(line.decode('utf-8').replace(' ',""))
    return _data_to_word_id(tmp, ch_word_to_id)

def ch_raw_data(data_path=None):
    train_path = os.path.join(data_path, "ch.train.txt")
    valid_path = os.path.join(data_path, "ch.valid.txt")
    test_path = os.path.join(data_path, "ch.test.txt")

    word_to_id = ch_word_to_id
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    return train_data, valid_data, test_data, vocabulary

def shift(sos, eos, tokens):
    return [sos] + tokens[:-1]


def list_flatten(l):
    return list(itertools.chain(*l))

def ch_producer(raw_data, batch_size, num_steps, name=None):
    '''
    Make an iterator on PTB data for language model

    INPUT:   <sos> the cat  sat  <eos> <eos>  ...
    WRAPPED:  THE  CAT SAT <eos> <eos> <eos>
    TARGET:   the  cat sat <eos> <eos> <eos>  ...
    WEIGHT:    1    1   1    1     1     0   000

    T:       -----------------------------
    W:       ----------------------------- ----- ----- (W = T + 2)

    Note: *weights* input is used in loss=seq2seq. It masks the padding
          part of input.
          len(weights) = len(inputs) + 2 is necessary
          to train rnn stop at the ending.
    '''

    with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
        data_len = len(raw_data)
        eos = ch_word_to_id['<eos>']
        sos = ch_word_to_id['<sos>']

        i = 0

        X = []
        Y = []
        W = []

        while i < data_len:
            t = 0
            while i + t < data_len and raw_data[i + t] != eos \
                    and t < num_steps:
                t += 1
            org = raw_data[i:i + t] + [eos] * (num_steps - t)
            # w = t + 2
            ww = [1.0] * min(t + 2, num_steps) + [0.0] * (num_steps - t - 2)

            X.append(shift(sos, eos, org))
            Y.append(org)
            W.append(ww)

            if i + t < data_len and raw_data[i + t] == eos:
                t += 1
            i += t


        epoch_size = len(W) // batch_size  # the iterations in an epoch

        if epoch_size == 0:
            raise ValueError("epoch_size == 0 "
                             "decrease batch_size or num_steps")

        def iterator():
            for j in range(epoch_size):
                x = np.array(X[j*batch_size:(j+1)*batch_size])
                y = np.array(Y[j*batch_size:(j+1)*batch_size])
                w = np.array(W[j*batch_size:(j+1)*batch_size])
                yield (x, y, w)

        return iterator(), epoch_size


def generate_random(filename, line):
    with codecs.open(filename, "w") as f:
        for i in range(line):
            k = random.randint(8,15)
            for j in range(k):
                f.write(" ")
                f.write(random.choice(ch_word_to_id.keys()).encode('utf-8'))
            f.write(" \n")
