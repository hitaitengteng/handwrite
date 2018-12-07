import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

from modle.label_one_hot import *

class Cnn:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 60, 160, 3])
        self.y_ = tf.placeholder(dtype=tf.float32, shape=[None, 20, 27])
        self.conv1_w = tf.Variable(tf.random_normal([3, 3, 3, 10], dtype=tf.float32, stddev=tf.sqrt(1 / 10)))
        self.conv1_b = tf.Variable(tf.zeros([10]))

        self.conv2_w = tf.Variable(tf.random_normal([3, 3, 10, 16], dtype=tf.float32, stddev=tf.sqrt(1 / 16)))
        self.conv2_b = tf.Variable(tf.zeros([16]))

        self.conv3_w = tf.Variable(tf.random_normal([3, 3, 16, 32], dtype=tf.float32, stddev=tf.sqrt(1 / 32)))
        self.conv3_b = tf.Variable(tf.zeros([32]))

        # self.conv4_w = tf.Variable(tf.random_normal([3, 3, 32, 64], dtype=tf.float32, stddev=tf.sqrt(1 / 64)))
        # self.conv4_b = tf.Variable(tf.zeros([64]))
        #
        # self.conv5_w = tf.Variable(tf.random_normal([3, 3, 64, 128], dtype=tf.float32, stddev=tf.sqrt(1 / 128)))
        # self.conv5_b = tf.Variable(tf.zeros([128]))

        self.fc1_w=tf.Variable(tf.random_normal([4*10*32,256],dtype=tf.float32,stddev=tf.sqrt(1/256)))
        self.fc1_b=tf.Variable(tf.zeros([256]))

        self.fc2_w = tf.Variable(tf.random_normal([256, 128], dtype=tf.float32, stddev=tf.sqrt(1 / 128)))
        self.fc2_b = tf.Variable(tf.zeros([128]))

        self.fc3_w = tf.Variable(tf.random_normal([128, 20*27], dtype=tf.float32, stddev=tf.sqrt(1 / 20*27)))
        self.fc3_b = tf.Variable(tf.zeros([20*27]))

    def forward(self):

        self.conv1=tf.nn.leaky_relu(
            tf.nn.conv2d(self.x,self.conv1_w,strides=[1,1,1,1],padding='SAME')+self.conv1_b
        )
        self.pool1=tf.nn.max_pool(self.conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

        self.conv2 = tf.nn.leaky_relu(
            tf.nn.conv2d(self.pool1, self.conv2_w, strides=[1, 1, 1, 1], padding='SAME')+self.conv2_b
        )
        self.pool2 = tf.nn.max_pool(self.conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        self.conv3 = tf.nn.leaky_relu(
            tf.nn.conv2d(self.pool2, self.conv3_w, strides=[1, 2, 2, 1], padding='SAME')+self.conv3_b
        )
        self.pool3 = tf.nn.max_pool(self.conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        print(self.pool3)

        # self.conv4 = tf.nn.leaky_relu(tf.layers.batch_normalization(
        #     tf.nn.conv2d(self.pool3, self.conv4_w, strides=[1, 1, 1, 1], padding='SAME')+self.conv4_b
        # ))
        # self.pool4 = tf.nn.max_pool(self.conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        #
        # self.conv5 = tf.nn.leaky_relu(tf.layers.batch_normalization(
        #     tf.nn.conv2d(self.pool4, self.conv5_w, strides=[1, 1, 1, 1], padding='SAME') + self.conv5_b
        # ))
        # self.pool5 = tf.nn.max_pool(self.conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


        self.fc_in=tf.reshape(self.pool3,[-1,4*10*32])
        self.fc1=tf.nn.leaky_relu(tf.matmul(self.fc_in,self.fc1_w)+self.fc1_b)
        self.fc2=tf.nn.leaky_relu(tf.matmul(self.fc1,self.fc2_w)+self.fc2_b)
        out=tf.matmul(self.fc2,self.fc3_w)+self.fc3_b
        self.out=tf.reshape(out,[-1,20,27])


    def backward(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.out))
        self.opt = tf.train.AdamOptimizer(0.001).minimize(self.loss)
        self.label_argmax = tf.arg_max(self.y_, 2)
        self.out_argmax = tf.arg_max(self.out, 2)
        correct_prediction = tf.equal(self.label_argmax, self.out_argmax)
        rst = tf.cast(correct_prediction, "float")
        self.accuracy = tf.reduce_mean(rst)


if __name__=='__main__':
    net=Cnn()
    net.forward()
    net.backward()
    init=tf.global_variables_initializer()
    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        x=[]
        y=[]
        # saver.restore(sess,'./cnn_save.dpk')
        for i in range(10000):
            train_img, train_labels = infordata.__getitem__(100)
            _, loss, ac, l, o = sess.run([net.opt, net.loss, net.accuracy, net.label_argmax, net.out_argmax],
                                         feed_dict={net.x: train_img, net.y_: train_labels})
            # x.append(i)
            # y.append(loss)

            # plt.plot(x, y, 'red')
            # plt.pause(0.01)
            # plt.clf()
            if i%100==0:
                test_img, test_label = testdata.__getitem__(100)
                loss1, ac1, l1, o1 = sess.run([net.loss, net.accuracy, net.label_argmax, net.out_argmax],
                                              feed_dict={net.x: test_img, net.y_: test_label})
                saver.save(sess, './cnn_save.dpk')
                print('第{}次测试集的误差为;{}，精度为:{}'.format(i, loss1, ac1))
                print('第{}次的误差是:{}，精度是:{}'.format(i, loss, ac))
                print('第几次的样本是:{}'.format(l[0]))
                print('第几次的输出是:{}'.format(o[0]))

