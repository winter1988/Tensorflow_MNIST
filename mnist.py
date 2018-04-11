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
print (trainimg.shape)
#定义模型
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.add(tf.matmul(x, W), b)
#定义损失函数 优化器
y_ = tf.placeholder(tf.float32,[None, 10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
#采用SGD作为优化器
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
mini_batch = 100
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #Trian
    for _ in range(1000):
        batch_xs , batch_ys = mnist.train.next_batch(mini_batch)
        sess.run(train_step, feed_dict={x:batch_xs,y_ : batch_ys})

    #tf.argmax(y, 1) 返回预测的值 tf.argmax(y_, 1) 返回的是真实值 比较两个值得出准确率
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    #tf.case 类型转换
    accurary = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    a = sess.run(accurary, feed_dict={x:testimg, y_:testimglabel})
    print(a)
