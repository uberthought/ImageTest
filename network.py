import tensorflow as tf
import numpy as np
import os.path
import math
import random

def createConv(input, filters, stride, name):
    with tf.name_scope(name):
        size = input.shape.as_list()[1] / stride
        dimensions = input.shape.as_list()[-1]

        filter3x3 = tf.Variable(tf.truncated_normal([3, 3, dimensions, filters], stddev=1.0 / math.sqrt(float(3 * 3 * dimensions * filters)), dtype=tf.float32), name='filter3x3')
        # filter5x5 = tf.Variable(tf.truncated_normal([5, 5, dimensions, filters], stddev=1.0 / math.sqrt(float(5 * 5 * dimensions * filters)), dtype=tf.float32), name='filter5x5')
        # filter7x7 = tf.Variable(tf.truncated_normal([7, 7, dimensions, filters], stddev=1.0 / math.sqrt(float(7 * 7 * dimensions * filters)), dtype=tf.float32), name='filter7x7')
        biases = tf.Variable(tf.zeros([size, size, filters], dtype=tf.float32), name='biases')

        output3x3 = tf.nn.conv2d(input=input, filter=filter3x3, strides=[1, stride, stride, 1], padding='SAME')
        # output5x5 = tf.nn.conv2d(input=input, filter=filter5x5, strides=[1, stride, stride, 1], padding='SAME')
        # output7x7 = tf.nn.conv2d(input=input, filter=filter7x7, strides=[1, stride, stride, 1], padding='SAME')

        output = output3x3 + biases

        if dimensions == filters:
            if stride == 1:
                pool = input
            if stride == 2:
                pool = tf.nn.pool(input, [stride, stride], "AVG", "VALID", strides=[stride, stride])
            output = output + pool

        output = tf.maximum(output * 0.01, output)
        # output = tf.nn.relu(output)

        return output


def upsample(input, units, name):
    with tf.name_scope(name):
        shape = input.get_shape().as_list()
        dimensions = len(shape[1:-1])

        output = input
        output = createConv(output, units, 1, 'conv')
        output = (tf.reshape(output, [-1] + shape[-dimensions:]))

        for i in range(dimensions, 0, -1):
            output = tf.concat([output, output], i)
        output_size = [-1] + [s * 2 for s in shape[1:-1]] + [shape[-1]]
        output = tf.reshape(output, output_size)

        return output


class Model:
    size = 32

    def __init__(self):

        size = Model.size
        shape = (size, size, 3)

        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32, name="steps")
        self.step = tf.assign(self.global_step, self.global_step + 1)

        self.global_loss = tf.Variable(0.2, trainable=False, dtype=tf.float32, name="global_loss")

        self.sess = tf.Session()

        self.X = tf.placeholder(tf.uint8, shape=(None,) + shape, name='X')

        input = tf.cast(self.X, tf.float32)
        input = input / 255

        layer = input
        original_size = np.prod(layer.get_shape().as_list()[1:])

        outer_units = 16
        outer_layers = 2

        inner_units = 16
        inner_layers = 2

        for i in range(outer_layers):
            layer = createConv(layer, outer_units, 1, 'conv')

        for i in range(inner_layers):
            layer = createConv(layer, inner_units, 2, 'conv')

        compressed_size = np.prod(layer.get_shape().as_list()[1:])
        print(original_size, compressed_size, 1 - compressed_size / original_size)

        center_layer = layer

        for i in range(inner_layers):
            layer = upsample(layer, inner_units, 'up')

        for i in range(outer_layers):
            layer = createConv(layer, outer_units, 1, 'conv')

        layer = createConv(layer, 3, 1, 'conv')

        # layer = tf.clip_by_value(layer, 0.0, 1.0)
        layer = tf.nn.sigmoid(layer)

        output = layer

        self.output = tf.cast(output * 255, tf.uint8)

        self.train_loss = tf.reduce_mean(tf.losses.mean_squared_error(input, output), name='train_loss')
        self.run_train = tf.train.AdagradOptimizer(.1).minimize(self.train_loss)

        weight = 0.25
        self.eloss = tf.assign(self.global_loss, self.train_loss * weight + self.global_loss * (1.0 - weight))

        self.test_loss = tf.reduce_mean(tf.losses.mean_squared_error(input, output), name='test_loss')

        self.summary_writer = tf.summary.FileWriter('./graph', self.sess.graph)
        self.train_loss_summary = tf.summary.scalar('train_loss', self.train_loss)
        self.test_loss_summary = tf.summary.scalar('test_loss', self.test_loss)
        center_layer_summary = tf.summary.histogram('center_layer', center_layer)
        self.summary = tf.summary.merge([self.test_loss_summary, center_layer_summary])

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

    def train(self, training, testing):
        x = training
        batch = 4
        if len(x) > batch:
            i = np.random.choice(range(len(x)), batch)
            x = x[i]

        i = 0
        train_loss = math.inf
        last_loss = 0
        eloss = self.sess.run(self.global_loss, feed_dict={self.X: x})

        while i < 1000 and train_loss > eloss * 0.95 and train_loss != last_loss:
            last_loss = train_loss
            train_loss, _, step = self.sess.run([self.train_loss, self.run_train, self.step], feed_dict={self.X: x})
            i = i + 1

            if i % 100 == 0:
                print('training', ((eloss - train_loss) / eloss))
                

        if train_loss == last_loss:
            print('hun?')
        
        if train_loss < 0.25 and train_loss != last_loss:
            self.sess.run([self.eloss], feed_dict={self.X: x})
            train_summary = self.sess.run(self.train_loss_summary, feed_dict={self.X: x})
            test_summary = self.sess.run(self.summary, feed_dict={self.X: testing})
            self.summary_writer.add_summary(train_summary, step)
            self.summary_writer.add_summary(test_summary, step)
        self.save()

        print('done', eloss, train_loss, ((eloss - train_loss) / eloss))

        return eloss
