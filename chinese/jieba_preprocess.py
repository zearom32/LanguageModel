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
import ch
import jieba

# input: a line decoded as utf-8
# output: jieba cut result, input is preprocessed
# line is decoded as utf-8
# remove other symbols

ch_list_set = set(ch.ch_list)
def cut_line(line):
    new_line = unicode()
    for k in line:
        if k in ch_list_set:
            new_line += k
    return jieba.cut(new_line)

def build_dict(filename):
    word_set = set(ch.ch_list)
    tmp_set = set()
    count = 0
    with open(filename, 'r') as f:
        for line in f:
            line = line.decode('utf-8')
            line_cut = cut_line(line)
            tmp_set = tmp_set.union(set(line_cut))
            count = count + 1
            if count % 50000 == 0:
                word_set = word_set.union(tmp_set)
                tmp_set = set()
                print(count)
    return word_set