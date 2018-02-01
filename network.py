import tensorflow as tf
import numpy as np
import os.path
import math
import time

def createConv(input, filters, stride, name):
        with tf.name_scope(name):
            size = input.shape.as_list()[1] / stride
            dimensions = input.shape.as_list()[-1]
            filter = tf.Variable(tf.truncated_normal([3, 3, dimensions, filters], stddev=1.0/math.sqrt(float(3 * 3 * dimensions * filters)), dtype=tf.float32), name='filter')
            biases = tf.Variable(tf.zeros([size, size, filters], dtype=tf.float32), name='biases')

            output = tf.nn.conv2d(input=input, filter=filter, strides=[1, stride, stride, 1], padding='SAME')
            output = tf.add(output, biases)

            if dimensions == filters:
                if stride == 1:
                    pool = input
                if stride == 2:
                    pool = tf.nn.pool(input, [stride, stride], "AVG", "VALID", strides=[stride, stride])
                output = tf.add(output, pool)

            # output = tf.maximum(tf.zeros_like(output), output)
            # output = tf.minimum(tf.ones_like(output), output)
            # output = tf.nn.relu(output)
            output = tf.nn.tanh(output)

            return output

def upsample(input, name):
        with tf.name_scope(name):
            shape = input.get_shape().as_list()
            dimensions = len(shape[1:-1])

            output = (tf.reshape(input, [-1] + shape[-dimensions:]))

            for i in range(dimensions, 0, -1):
                output = tf.concat([output, output], i)
                # output = tf.concat([output, tf.zeros_like(output)], i)
            output_size = [-1] + [s * 2 for s in shape[1:-1]] + [shape[-1]]
            output = tf.reshape(output, output_size)

            return output

class Model:
    size=128

    def __init__(self):

        size = Model.size
        shape = (size, size, 3)

        self.sess = tf.Session()


        self.X = tf.placeholder(tf.uint8, shape=(None,) + shape, name='X')
        self.Y = tf.placeholder(tf.uint8, shape=(None,) + shape, name='Y')

        input = tf.multiply(tf.cast(self.X, tf.float32), 1 / 255)

        layer = input
        original_size = np.prod(layer.get_shape().as_list()[1:])

        layer = createConv(layer, 8, 1, 'conv')
        layer = createConv(layer, 8, 2, 'conv')
        layer = createConv(layer, 8, 2, 'conv')
        layer = createConv(layer, 3, 1, 'conv')

        compressed_size = np.prod(layer.get_shape().as_list()[1:])
        print(original_size, compressed_size, 1 - compressed_size / original_size)
        # exit()

        layer = createConv(layer, 8, 1, 'conv')
        layer = upsample(layer, 'up')
        layer = createConv(layer, 8, 1, 'conv')
        layer = upsample(layer, 'up')
        layer = createConv(layer, 8, 1, 'conv')
        layer = createConv(layer, 3, 1, 'conv')

        layer = tf.nn.relu(layer)

        output = layer

        self.output = tf.cast(tf.multiply(output, 255), tf.uint8)

        self.loss = tf.reduce_mean(tf.losses.mean_squared_error(input, output), name='loss')
        self.run_train = tf.train.AdagradOptimizer(.1).minimize(self.loss)

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
        loss = math.inf
        i = 0
        while i < 100:
            loss, _, summary = self.sess.run([self.loss, self.run_train, self.summary], feed_dict=feed_dict)
            i += 1

        self.summary_writer.add_summary(summary)

        return loss
