#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import random

import numpy as np


def read_data(filename):
    with open(filename, encoding="utf-8") as f:
        data = f.read()
    data = list(data)
    return data


def index_data(sentences, dictionary):
    shape = sentences.shape
    sentences = sentences.reshape([-1])
    index = np.zeros_like(sentences, dtype=np.int32)
    for i in range(len(sentences)):
        try:
            index[i] = dictionary[sentences[i]]
        except KeyError:
            index[i] = dictionary['UNK']

    return index.reshape(shape)


def get_train_data(vocabulary, batch_size, num_steps):
    ################# My code here ###################
    # 有时不能整除，需要截掉一些字
    data_partition_size = len(vocabulary) // batch_size
    word_valid_count = batch_size * data_partition_size
    vocabulary_valid = np.array(vocabulary[: word_valid_count])
    word_x = vocabulary_valid.reshape([batch_size, data_partition_size])

    # 随机一个起始位置
    start_idx = random.randint(0, 16)
    while True:
        # 因为训练中要使用的是字和label(下一个字)的index，但这里没有dictionary，无法得到index
        # 所以将每个time step返回的训练数据长度是num_steps+1
        #     其中前num_steps个字用于训练(训练时转化为index)
        #     从第2个字起的num_steps个字用于训练的label(训练时转化为index)
        if start_idx + num_steps + 1 > word_valid_count:
            break

        yield word_x[:, start_idx: start_idx + num_steps + 1]
        start_idx += num_steps
    ##################################################


def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary
