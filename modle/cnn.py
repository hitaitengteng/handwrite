import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

from modle.tf_sample import *
from modle.tf_test_sample import *

class Cnn:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 80, 280, 1])
        self.y_ = tf.placeholder(dtype=tf.float32, shape=[None, 20])
        self.conv1_w = tf.Variable(tf.random_normal([3, 3, 1, 10], dtype=tf.float32, stddev=tf.sqrt(1 / 10)))
        self.conv1_b = tf.Variable(tf.zeros([10]))

        self.conv1_dw = tf.Variable(tf.truncated_normal([3, 3, 10, 1], dtype=tf.float32, stddev=tf.sqrt(1 / 10)))
        self.conv1_db=tf.Variable(tf.zeros([10]))

        self.conv2_w = tf.Variable(tf.random_normal([3, 3, 10, 16], dtype=tf.float32, stddev=tf.sqrt(1 / 16)))
        self.conv2_b = tf.Variable(tf.zeros([16]))

        self.conv2_dw = tf.Variable(tf.truncated_normal([3, 3, 16, 1], dtype=tf.float32, stddev=tf.sqrt(1 / 16)))
        self.conv2_db = tf.Variable(tf.zeros([16]))

        self.conv3_w = tf.Variable(tf.random_normal([3, 3, 16, 32], dtype=tf.float32, stddev=tf.sqrt(1 / 32)))
        self.conv3_b = tf.Variable(tf.zeros([32]))

        self.conv3_dw = tf.Variable(tf.truncated_normal([3, 3, 32, 1], dtype=tf.float32, stddev=tf.sqrt(1 / 32)))
        self.conv3_db = tf.Variable(tf.zeros([32]))

        self.fc1_w=tf.Variable(tf.random_normal([5*18*32,256],dtype=tf.float32,stddev=tf.sqrt(1/256)))
        self.fc1_b=tf.Variable(tf.zeros([256]))

        self.fc2_w = tf.Variable(tf.random_normal([256, 128], dtype=tf.float32, stddev=tf.sqrt(1 / 128)))
        self.fc2_b = tf.Variable(tf.zeros([128]))

        self.fc3_w = tf.Variable(tf.random_normal([128, 20], dtype=tf.float32, stddev=tf.sqrt(1 / 20)))
        self.fc3_b = tf.Variable(tf.zeros([20]))

    def forward(self):

        self.conv1=tf.nn.relu(
            tf.nn.conv2d(self.x,self.conv1_w,strides=[1,1,1,1],padding='SAME')+self.conv1_b
        )
        self.conv1d=tf.nn.relu(tf.nn.depthwise_conv2d(self.conv1,self.conv1_dw,strides=[1,2,2,1],padding='SAME')+self.conv1_db)

        self.conv2 = tf.nn.relu(
            tf.nn.conv2d(self.conv1d, self.conv2_w, strides=[1, 1, 1, 1], padding='SAME')+self.conv2_b
        )
        self.conv2d = tf.nn.relu(
            tf.nn.depthwise_conv2d(self.conv2, self.conv2_dw, strides=[1, 2, 2, 1], padding='SAME') + self.conv2_db)

        self.conv3 = tf.nn.relu(
            tf.nn.conv2d(self.conv2d, self.conv3_w, strides=[1, 2, 2, 1], padding='SAME')+self.conv3_b
        )
        self.conv3d = tf.nn.relu(
            tf.nn.depthwise_conv2d(self.conv3, self.conv3_dw, strides=[1, 2, 2, 1], padding='SAME') + self.conv3_db)

        self.fc_in=tf.reshape(self.conv3d,[-1,5*18*32])
        self.fc1=tf.nn.leaky_relu(tf.matmul(self.fc_in,self.fc1_w)+self.fc1_b)
        self.fc2=tf.nn.leaky_relu(tf.matmul(self.fc1,self.fc2_w)+self.fc2_b)
        out=tf.matmul(self.fc2,self.fc3_w)+self.fc3_b
        self.out=tf.reshape(out,[-1,20])
        print(self.out)


    def backward(self):
        global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(0.001, global_step, 100, 0.96, staircase=True)
        self.loss = tf.reduce_mean((self.y_-self.out)**2)
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,global_step=global_step)
        # self.label_argmax = tf.arg_max(self.y_, 2)
        # self.out_argmax = tf.arg_max(self.out, 2)
        # correct_prediction = tf.equal(self.label_argmax, self.out_argmax)
        # rst = tf.cast(correct_prediction, "float")



if __name__=='__main__':
    net=Cnn()
    net.forward()
    net.backward()

    init=tf.global_variables_initializer()
    saver=tf.train.Saver()

    train_data = train_shuffle_batch(train_filename, [80, 280, 1], 100)


    test_data= test_shuffle_batch(test_filename, [80, 280, 1], 100)
    with tf.Session() as sess:
        sess.run(init)
        x=[]
        y=[]
        coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        saver.restore(sess,'./cnn_save.dpk')
        for i in range(50000):
            train_img,train_label=sess.run(train_data)
            train_img=train_img/255-0.5
            train_label=train_label/26.


            _, loss= sess.run([net.opt, net.loss],
                                         feed_dict={net.x: train_img, net.y_: train_label})
            x.append(i)
            y.append(loss)

            plt.plot(x, y, 'red')
            plt.pause(0.01)
            plt.clf()
            print(loss)
            if i%200==0:
                test_img, test_label = sess.run(train_data)
                test_img = test_img / 255 - 0.5
                test_label=test_label/26.

                loss1,out = sess.run([net.loss,net.out],
                                              feed_dict={net.x: test_img, net.y_: test_label})
                saver.save(sess, './cnn_save.dpk')
                print('第{}次测试集的误差为;{}'.format(i, loss1))
                print('第{}次的误差是:{}'.format(i, loss))
                print('label: ',test_label[0]*26.)
                print('out: ',[int(i) for i in ((out[0]*26.).tolist())])
                # print('第几次的样本是:{}'.format(l[0]))
                # print('第几次的输出是:{}'.format(o[0]))

