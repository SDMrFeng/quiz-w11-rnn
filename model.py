#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


class Model():
    def __init__(self, learning_rate=0.001, batch_size=16, num_steps=32, num_words=5000, dim_embedding=128, rnn_layers=3):
        r"""初始化函数

        Parameters
        ----------
        learning_rate : float
            学习率.
        batch_size : int
            batch_size.
        num_steps : int
            RNN有多少个time step，也就是输入数据的长度是多少.
        num_words : int
            字典里有多少个字，用作embeding变量的第一个维度的确定和onehot编码.
        dim_embedding : int
            embding中，编码后的字向量的维度
        rnn_layers : int
            有多少个RNN层，在这个模型里，一个RNN层就是一个RNN Cell，各个Cell之间通过TensorFlow提供的多层RNNAPI（MultiRNNCell等）组织到一起
            
        """
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_words = num_words
        self.dim_embedding = dim_embedding
        self.rnn_layers = rnn_layers
        self.learning_rate = learning_rate

    def build(self, embedding_file=None):
        # global step
        self.global_step = tf.Variable(
            0, trainable=False, name='self.global_step', dtype=tf.int64)

        self.X = tf.placeholder(
            tf.int32, shape=[None, self.num_steps], name='input')
        self.Y = tf.placeholder(
            tf.int32, shape=[None, self.num_steps], name='label')

        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        with tf.variable_scope('embedding'):
            if embedding_file:
                # if embedding file provided, use it.
                embedding = np.load(embedding_file)
                embed = tf.constant(embedding, name='embedding')
            else:
                # if not, initialize an embedding and train it.
                embed = tf.get_variable(
                    'embedding', [self.num_words, self.dim_embedding])
                tf.summary.histogram('embed', embed)

            data = tf.nn.embedding_lookup(embed, self.X)

        with tf.variable_scope('rnn'):
            ################# My code here ###################
            cells = [tf.nn.rnn_cell.DropoutWrapper(
                                                tf.nn.rnn_cell.BasicLSTMCell(self.dim_embedding),
                                                output_keep_prob=self.keep_prob
                                                ) for i in range(self.rnn_layers)]

            rnn_multi = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

            self.state_tensor = rnn_multi.zero_state(self.batch_size, tf.float32)

            outputs_tensor, self.outputs_state_tensor = tf.nn.dynamic_rnn(
                rnn_multi, data, initial_state = self.state_tensor, dtype=tf.float32)

            tf.summary.histogram('outputs_state_tensor', self.outputs_state_tensor)

        seq_output = tf.concat(outputs_tensor, 1)
            ##################################################

        # flatten it
        seq_output_final = tf.reshape(seq_output, [-1, self.dim_embedding])

        with tf.variable_scope('softmax'):
            ################# My code here ###################
            softmax_w = tf.Variable(tf.truncated_normal([self.dim_embedding, self.num_words], stddev=0.1))
            softmax_b = tf.Variable(tf.zeros(self.num_words))

            tf.summary.histogram('softmax_w', softmax_w)
            tf.summary.histogram('softmax_b', softmax_b)

        logits = tf.reshape(tf.matmul(seq_output_final, softmax_w) + softmax_b, [self.batch_size, -1, self.num_words])
            ##################################################

        tf.summary.histogram('logits', logits)

        self.predictions = tf.nn.softmax(logits, name='predictions')

        #print('logits shape:', logits.get_shape())
        #print('labels shape:', self.Y.get_shape())
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=self.Y)
        self.loss = tf.reduce_mean(loss)
        tf.summary.scalar('logits_loss', self.loss)

        # gradient clip
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(
            zip(grads, tvars), global_step=self.global_step)

        tf.summary.scalar('loss', self.loss)

        self.merged_summary_op = tf.summary.merge_all()
