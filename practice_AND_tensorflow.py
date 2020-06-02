import tensorflow as tf
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.set_random_seed(777)
print(tf.__version__)

#AND

mylist = [[i,v]for i in range(0,2)for v in range(0,2)]
print(mylist)
x_data = np.array(mylist, dtype = np.float32)
print(x_data)
y_data = np.array([[0],[0],[0],[1]],dtype= np.float32)

X = tf.placeholder(tf.float32, [None,2], name = 'x-input')
Y = tf.placeholder(tf.float32, [None,1], name = 'y-input')

W = tf.Variable(tf.random_normal([2,1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(X,W)+b)

cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(1,10001):
        sess.run(train, feed_dict = { X: x_data, Y: y_data})

        if step%100 ==0:
            print(step,sess.run(cost, feed_dict={X:x_data, Y:y_data}),sess.run(W))

        h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})
print("\nHypothesis:", h,"\nCorrect: ",c, "\nAccuracy: ", a)