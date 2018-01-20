import tensorflow as tf
import numpy as np
import os.path
import math
import time

def createHidden(input_layer, units, name):
        input_units = int(input_layer.shape[1])
        with tf.name_scope(name):
            weights = tf.Variable(tf.truncated_normal([input_units, units], stddev=1.0/math.sqrt(float(input_units)), dtype=tf.float32), name='weights')
            biases = tf.Variable(tf.zeros([units], dtype=tf.float32), name='biases')
            result = tf.matmul(input_layer, weights)
            result = tf.nn.relu(result)
            result = tf.add(result, biases)
            if input_units == units:
                result = tf.add(input_layer, result)
            return result

def createOutput(input_layer, units, name):
        input_units = int(input_layer.shape[1])
        with tf.name_scope(name):
            weights = tf.Variable(tf.truncated_normal([input_units, units], stddev=1.0/math.sqrt(float(input_units)), dtype=tf.float32), name='weights')
            biases = tf.Variable(tf.zeros([units], dtype=tf.float32), name='biases')
            result = tf.matmul(input_layer, weights)
            result = tf.add(result, biases)
            if input_units == units:
                result = tf.add(input_layer, result)
            return result

class Model:
    size=256

    def __init__(self):

        size = Model.size

        self.X = tf.placeholder(tf.float32, shape=(None, size, size, 3), name='X')
        self.Y = tf.placeholder(tf.float32, shape=(None, size, size, 3), name='Y')

        input = tf.reshape(self.X, (-1, size * size * 3), name='input')
        hidden1 = createHidden(input, 128, 'hidden1')
        hidden2 = createHidden(hidden1, 32, 'hidden2')
        hidden3 = createHidden(hidden2, 128, 'hidden3')
        output = createOutput(hidden3, size * size * 3, 'output')

        self.prediction = tf.reshape(output, (-1, size, size, 3), name='prediction')

        self.loss = tf.reduce_mean(tf.losses.mean_squared_error(self.Y, self.prediction), name='loss')
        self.run_train = tf.train.AdagradOptimizer(.1).minimize(self.loss)

        self.output = tf.cast(self.prediction, tf.int8, name='output')

        self.sess = tf.Session()

        self.summary_writer = tf.summary.FileWriter('./graph', self.sess.graph)
        loss_summary = tf.summary.scalar('loss', self.loss)
        # model_expected_summary = tf.summary.histogram('value expected', self.values)
        # model_predicted_summary = tf.summary.histogram('value prediction', self.model_prediction)
        # model_hidden0_summary = tf.summary.histogram('model hidden 0', model_hidden0)
        # model_hidden1_summary = tf.summary.histogram('model hidden 1', model_hidden1)
        self.summary = tf.summary.merge([loss_summary])

        self.sess.run(tf.global_variables_initializer())

        if os.path.exists('graph/graph.meta'):
                print("loading training data")
                saver = tf.train.Saver()
                saver.restore(self.sess, 'graph/graph')

    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, 'graph/graph')
        self.summary_writer.flush()

    def run(self, X):
        return self.sess.run(self.output, feed_dict={self.X: X})

    def train(self, X, Y):
        feed_dict = {self.X: X, self.Y: Y}
        # start = time.time()
        loss = math.inf
        while loss > 1:
            loss, _, summary = self.sess.run([self.loss, self.run_train, self.summary], feed_dict=feed_dict)
            # self.summary_writer.add_summary(summary)
            print(loss)
        # print(time.time() - start)
        return loss
