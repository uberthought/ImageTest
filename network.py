import tensorflow as tf
import numpy as np
import os.path
import math
import time

def createHidden(input, shape, name):
        input_units = int(np.prod(input.shape.as_list()[1:]))
        output_units = np.prod(shape)
        with tf.name_scope(name):
            weights = tf.Variable(tf.truncated_normal([input_units, output_units], stddev=1.0/math.sqrt(float(input_units)), dtype=tf.float32), name='weights')
            biases = tf.Variable(tf.zeros([output_units], dtype=tf.float32), name='biases')

            flatten = tf.reshape(input, (-1, input_units))
            output = tf.matmul(flatten, weights)
            output = tf.add(output, biases)
            output = tf.nn.relu(output)
            if input_units == output_units:
                output = tf.add(flatten, output)
            output = tf.reshape(output, (-1,) + shape)

            return output

def createOutput(input, shape, name):
        input_units = int(np.prod(input.shape.as_list()[1:]))
        output_units = int(np.prod(shape))
        with tf.name_scope(name):
            weights = tf.Variable(tf.truncated_normal([input_units, output_units], stddev=1.0/math.sqrt(float(input_units)), dtype=tf.float32), name='weights')
            biases = tf.Variable(tf.zeros([output_units], dtype=tf.float32), name='biases')

            flatten = tf.reshape(input, (-1, input_units))
            output = tf.matmul(flatten, weights)
            output = tf.add(output, biases)
            if input_units == output_units:
                output = tf.add(flatten, output)
            output = tf.reshape(output, (-1,) + shape)

            return output

def createConv(input, filters, name):
        with tf.name_scope(name):
            input_shape = input.shape.as_list()[1:]
            dimensions = input_shape[-1]
            filter = tf.Variable(tf.truncated_normal([3, 3, dimensions, filters], stddev=1.0/math.sqrt(float(3 * 3 * dimensions * filters)), dtype=tf.float32), name='filter')
            biases = tf.Variable(tf.zeros(input_shape[:-1] + [filters], dtype=tf.float32), name='biases')

            output = tf.nn.conv2d(input=input, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
            output = tf.add(output, biases)
            output = tf.nn.tanh(output)

            return output

class Model:
    size=64
    output_size=10

    def __init__(self):

        size = Model.size
        output_size = Model.output_size
        shape = (size, size, 3)

        self.sess = tf.Session()


        self.X = tf.placeholder(tf.int8, shape=(None,) + shape, name='X')
        self.Y = tf.placeholder(tf.float32, shape=(None, output_size), name='Y')

        units = np.prod(shape)

        input = self.X
        input = tf.cast(input, tf.float32)
        input = tf.multiply(input, 1 / 255)

        conv1 = createConv(input, 8, 'conv1')
        conv2 = createConv(conv1, 8, 'conv2')
        hidden = createHidden(conv2, (16,), 'hidden')

        output = createOutput(hidden, (output_size,), 'output')

        self.loss = tf.reduce_mean(tf.losses.mean_squared_error(self.Y, output), name='loss')
        self.run_train = tf.train.AdagradOptimizer(.1).minimize(self.loss)

        self.output = output

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
        i = 0
        while loss > .001 and i < 1000:
            loss, _, summary = self.sess.run([self.loss, self.run_train, self.summary], feed_dict=feed_dict)
            i += 1
            # self.summary_writer.add_summary(summary)
            if i % 100 == 0:
                print(i, loss)
        # print(time.time() - start)
        return loss
