import tensorflow as tf
import numpy as np
import os.path
import math
import random
import time

class Model:
    size = 128

    def __init__(self):

        size = Model.size
        shape = (size, size, 3)

        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32, name="steps")
        self.step = tf.assign(self.global_step, self.global_step + 1)

        self.global_loss = tf.Variable(0.2, trainable=False, dtype=tf.float32, name="global_loss")
        self.last_loss = tf.placeholder(tf.float32, shape=(), name='last_loss')
        weight = 0.5
        self.eloss = tf.assign(self.global_loss, self.last_loss * weight + self.global_loss * (1.0 - weight))

        self.sess = tf.Session()

        self.X = tf.placeholder(tf.uint8, shape=(None,) + shape, name='X')

        input = tf.cast(self.X, tf.float32)
        input = input / 255

        layer = input
        original_size = np.prod(layer.get_shape().as_list()[1:])

        outer_units = 128
        outer_layers = 2

        inner_units = 128
        inner_layers = 4

        run = "run1"

        for i in range(outer_layers):
            layer = self.createConv(layer, outer_units, 1, 'conv')

        for i in range(inner_layers):
            layer = self.createConv(layer, inner_units, 2, 'conv')

        compressed_size = np.prod(layer.get_shape().as_list()[1:])
        print(original_size, compressed_size, 1 - compressed_size / original_size)

        center_layer = layer

        for i in range(inner_layers):
            layer = self.upsample(layer, inner_units, 'up')

        for i in range(outer_layers):
            layer = self.createConv(layer, outer_units, 1, 'conv')

        layer = self.createConv(layer, 3, 1, 'conv')

        layer = tf.clip_by_value(layer, 0.0, 1.0)
        # layer = tf.nn.sigmoid(layer)

        output = layer

        self.output = tf.cast(output * 255, tf.uint8)

        self.train_loss = tf.reduce_mean(tf.losses.mean_squared_error(input, output), name='train_loss')
        self.train_loss2 = tf.sqrt(self.train_loss, name='train_loss2')
        self.run_train = tf.train.AdagradOptimizer(.1).minimize(self.train_loss)

        self.test_loss = tf.reduce_mean(tf.losses.mean_squared_error(input, output), name='test_loss')
        self.test_loss2 = tf.sqrt(self.test_loss, name='test_loss2')

        self.summary_writer = tf.summary.FileWriter('./summary/' + run, self.sess.graph)
        self.train_loss_summary = tf.summary.scalar('train_loss',self.train_loss2)
        self.test_loss_summary = tf.summary.scalar('test_loss', self.test_loss2)
        center_layer_summary = tf.summary.histogram('center_layer', center_layer)
        self.summary = tf.summary.merge([self.test_loss_summary, center_layer_summary])

        self.sess.run(tf.global_variables_initializer())

        if os.path.exists('graph/graph.meta'):
            print("loading training data")
            saver = tf.train.Saver()
            saver.restore(self.sess, 'graph/graph')

    def createConv(self, input, filters, stride, name):
        with tf.name_scope(name):
            size = input.shape.as_list()[1] / stride
            dimensions = input.shape.as_list()[-1]

            filter3x3 = tf.Variable(tf.truncated_normal([3, 3, dimensions, filters], stddev=1.0 / math.sqrt(float(3 * 3 * dimensions * filters)), dtype=tf.float32), name='filter3x3')
            # filter5x5 = tf.Variable(tf.truncated_normal([5, 5, dimensions, filters], stddev=1.0 / math.sqrt(float(5 * 5 * dimensions * filters)), dtype=tf.float32), name='filter5x5')
            # filter7x7 = tf.Variable(tf.truncated_normal([7, 7, dimensions, filters], stddev=1.0 / math.sqrt(float(7 * 7 * dimensions * filters)), dtype=tf.float32), name='filter7x7')
            # filter13x13 = tf.Variable(tf.truncated_normal([13, 13, dimensions, filters], stddev=1.0 / math.sqrt(float(13 * 13 * dimensions * filters)), dtype=tf.float32), name='filter13x13')
            biases = tf.Variable(tf.zeros([size, size, filters], dtype=tf.float32), name='biases')

            output3x3 = tf.nn.conv2d(input=input, filter=filter3x3, strides=[1, stride, stride, 1], padding='SAME')
            # output5x5 = tf.nn.conv2d(input=input, filter=filter5x5, strides=[1, stride, stride, 1], padding='SAME')
            # output7x7 = tf.nn.conv2d(input=input, filter=filter7x7, strides=[1, stride, stride, 1], padding='SAME')
            # output13x13 = tf.nn.conv2d(input=input, filter=filter13x13, strides=[1, stride, stride, 1], padding='SAME')

            output = output3x3 + biases

            output = tf.maximum(output * 0.1, output)

            if dimensions == filters:
                if stride == 1:
                    pool = input
                if stride == 2:
                    pool = tf.nn.pool(input, [stride, stride], "MAX", "VALID", strides=[stride, stride])
                    # pool = tf.nn.pool(input, [stride, stride], "AVG", "VALID", strides=[stride, stride])
                output = output * 0.1 + pool

            return output


    def upsample(self, input, units, name):
        with tf.name_scope(name):
            shape = input.get_shape().as_list()
            dimensions = len(shape[1:-1])

            output = input
            output = self.createConv(output, units, 1, 'conv')
            output = (tf.reshape(output, [-1] + shape[-dimensions:]))

            for i in range(dimensions, 0, -1):
                output = tf.concat([output, output], i)
            output_size = [-1] + [s * 2 for s in shape[1:-1]] + [shape[-1]]
            output = tf.reshape(output, output_size)

            return output


    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, 'graph/graph')
        self.summary_writer.flush()

    def run(self, X):
        return self.sess.run(self.output, feed_dict={self.X: X})

    def train(self, training, testing):
        
        i = 0
        train_loss = math.inf
        eloss = self.sess.run(self.global_loss)
        start = time.time()

        while i < 1024 and train_loss > eloss * 0.90:
            
            i += 1
            train_loss, _, step, train_summary = self.sess.run([self.train_loss2, self.run_train, self.step, self.train_loss_summary], feed_dict={self.X: training})
            self.summary_writer.add_summary(train_summary, step)

        end = time.time()
        iteration_per_second = i / (end - start)


        test_loss, test_summary = self.sess.run([self.test_loss2, self.summary], feed_dict={self.X: testing})
        self.summary_writer.add_summary(test_summary, step)
            
        self.sess.run([self.eloss], feed_dict={self.last_loss: train_loss})

        self.save()

        print('training loss', train_loss, 'iteration/second', iteration_per_second)

        return train_loss
