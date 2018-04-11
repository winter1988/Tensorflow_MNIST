# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('data/', one_hot=True)
trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testimglabel = mnist.test.labels
print(trainimg.shape)
#处理数据
trainimg = trainimg.reshape([-1, 28, 28, 1])
testimg = testimg.reshape([-1, 28, 28, 1])
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])
#定义权重
def init_weight(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))
w = init_weight([3, 3, 1, 32])
w2 = init_weight([3, 3, 32, 64])
w3 = init_weight([3, 3, 64, 128])
w4 = init_weight([128*4*4, 256])
w_o = init_weight([256, 10])

keep_prob = 0.8
l1 = tf.nn.relu(tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME'))
l1 = tf.nn.max_pool(l1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
l1= tf.nn.dropout(l1, keep_prob)

l2 = tf.nn.relu(tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding='SAME'))
l2 = tf.nn.max_pool(l2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
l2 = tf.nn.dropout(l2, keep_prob)

l3 = tf.nn.relu(tf.nn.conv2d(l2, w3, strides=[1, 1, 1, 1], padding='SAME'))
l3 = tf.nn.max_pool(l3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])
l3 = tf.nn.dropout(l3, keep_prob)
#全连接层
l4 = tf.nn.relu(tf.matmul(l3, w4))
l4 = tf.nn.dropout(l4, keep_prob)

#输出层
pyx = tf.matmul(l4, w_o)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pyx, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(pyx, 1)

with tf.Session() as sess:
    batch_size = 128
    test_size = 256
    tf.global_variables_initializer().run()
    for i in range(100):
        training_batch = zip(range(0, len(trainimg), batch_size),
                                   range(batch_size, len(trainimg)+1, batch_size))
        batch = mnist.train.next_batch(50)
        trainInput = batch[0].reshape([50, 28, 28, 1])
        trainLabels = batch[1]
        feed_dict = {X: trainInput, Y: trainLabels}
        for s, e in training_batch:
            sess.run(train_op, feed_dict)

            test_indices = np.arange(len(testimg))
            np.random.shuffle(test_indices)
            test_indices = test_indices[0:test_size]
            print(i, np.mean(np.argmax(testimglabel[test_indices], axis=1) ==
                             sess.run(predict_op, feed_dict={X:testimg[test_indices]})))






