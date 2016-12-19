import numpy as np
import tensorflow as tf

class TextCNN(object):
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters):
        # input,  dropout
        input = tf.placeholder(tf.int32, [None, sequence_length], name='input')
        label = tf.placeholder(tf.float32, [None, num_classes], name='label')
        dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        with tf.name_scope('embedding'):
            W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name='W')
            # [None, sequence_length, embedding_size]
            embedded_chars = tf.nn.embedding_lookup(W, input)
            # [None, sequence_length, embedding_size, 1]
            embedded_chars = tf.expand_dims(embedded_chars, -1)

        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv-maxpool-%s' % filter_size):
                # convolution
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
                conv = tf.nn.conv2d(
                    embedded_chars,
                    W,
                    strides=[1,1,1,1],
                    padding='VALID',
                    name='conv')
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='pool')
                pooled_outputs.append(pooled)
        # 
        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(3, pooled_outputs)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        # dropout
        with tf.name_scope('dropout'):
            h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)

        # prediction
        with tf.name_scope('output'):
            W = tf.get_variable('W', shape=[num_filters_total, num_classes], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')

            scores = tf.nn.xw_plus_b(h_drop, W, b, name='scores')
            predictions = tf.argmax(scores, 1, name='predictions')

        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(scores, label)
            loss = tf.reduce_mean(losses)

        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(predictions, tf.argmax(label, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')

        # variables
        self.input = input
        self.label = label
        self.dropout_keep_prob = dropout_keep_prob
        self.predictions = predictions
        self.loss = loss
        self.accuracy = accuracy


        
    if __name__ == '__main__':
        TextCNN(59, 2, 100, 128, [3,4,5], 128)