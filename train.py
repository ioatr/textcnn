from data import *
from textcnn import TextCNN
import tensorflow as tf
import random
import numpy as np
import os
import sys

TRAIN_FILENAME = 'ratings_train.txt'
TRAIN_DATA_FILENAME = TRAIN_FILENAME + '.data'
TRAIN_VOCAB_FILENAME = TRAIN_FILENAME + '.vocab'

TEST_FILENAME = 'ratings_test.txt'
TEST_DATA_FILENAME = TRAIN_FILENAME + '.data'
TEST_VOCAB_FILENAME = TRAIN_FILENAME + '.vocab'

def train():

    if (os.path.exists(TRAIN_DATA_FILENAME) and os.path.exists(TRAIN_VOCAB_FILENAME)):
        print('load prebuilt train data & vocab file') 
        input = load_data(TRAIN_DATA_FILENAME)
        vocab =  load_vocab(TRAIN_VOCAB_FILENAME)
    else:
        print('build train data & vocab from raw text')
        data = read_raw_data(TRAIN_FILENAME)
        tokens = [t for d in data for t in d[0]]
        
        vocab = build_vocab(tokens)
        input = build_input(data, vocab)

        print('save train data & vocab file')
        save_data(TRAIN_DATA_FILENAME, input)
        save_vocab(TRAIN_VOCAB_FILENAME, vocab)
    
    if (os.path.exists(TEST_DATA_FILENAME) and os.path.exists(TEST_VOCAB_FILENAME)):
        print('load prebuilt test data & vocab file ')
        test_input = load_data(TEST_DATA_FILENAME)
        test_vocab = load_vocab(TEST_VOCAB_FILENAME)
    else:
        print('build test data & vocab from raw text')
        data = read_raw_data(TEST_FILENAME)
        tokens = [t for d in data for t in d[0]]
        
        test_vocab = build_vocab(tokens)
        test_input = build_input(data, test_vocab)

        print('save test data & vocab file')
        save_data(TEST_DATA_FILENAME, test_input)
        save_vocab(TEST_VOCAB_FILENAME, test_vocab)

    
    

    with tf.Session() as sess:

        seq_length = np.shape(input[0][0])[0]
        num_class = np.shape(input[0][1])[0]

        print('initialize cnn filter')
        print('sequence length %d,  number of class %d, vocab size %d' % (seq_length, num_class, len(vocab)))
        
        cnn = TextCNN(seq_length, num_class, len(vocab), 128, [3,4,5], 128)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        def train_step(x_batch, y_batch):
            feed_dict = {
                cnn.input : x_batch,
                cnn.label : y_batch,
                cnn.dropout_keep_prob : 0.5
            }

            _, step, loss, accuracy = sess.run([train_op, global_step, cnn.loss, cnn.accuracy], feed_dict)

        def evaluate(x_batch, y_batch):
            feed_dict = {
                cnn.input : x_batch,
                cnn.label : y_batch,
                cnn.dropout_keep_prob : 1.0
            }

            step, loss, accuracy = sess.run([global_step, cnn.loss, cnn.accuracy], feed_dict)
            print("step %d, loss %f, acc %f" % (step, loss, accuracy))

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        
        for i in range(10000):
            try:
                batch = random.sample(input, 64) 
            
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % 100 == 0:
                    batch = random.sample(test_input, 64)
                    x_test, y_test = zip(*batch)
                    evaluate(x_test, y_test)
                if current_step % 1000 == 0:
                    save_path = saver.save(sess, './textcnn.ckpt')
                    print('model saved : %s' % save_path)
            except:
                print ("Unexpected error:", sys.exc_info()[0])
                raise

if __name__ == '__main__':
    train()