import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from modle.tf_sample import *
from modle.tf_test_sample import *
# from modle.cnn_lstm_sample import *

class Cnn2Lstm():
    def __init__(self):
        self.x=tf.placeholder(dtype=tf.float32,shape=[None,80,280,1])
        self.y_=tf.placeholder(dtype=tf.float32,shape=[None,20])
        self.conv1_w=tf.Variable(tf.random_normal([3,3,1,16],dtype=tf.float32,stddev=tf.sqrt(1/16)))
        self.conv1_b=tf.Variable(tf.zeros([16]))
        self.conv1_dw = tf.Variable(tf.truncated_normal([3, 3, 16, 1], dtype=tf.float32, stddev=tf.sqrt(1 / 16)))
        self.conv1_db=tf.Variable(tf.zeros([16]))

        self.conv2_w=tf.Variable(tf.random_normal([3,3,16,32],dtype=tf.float32,stddev=tf.sqrt(1/32)))
        self.conv2_b=tf.Variable(tf.zeros([32]))
        self.conv2_dw = tf.Variable(tf.truncated_normal([3, 3, 32, 1], dtype=tf.float32, stddev=tf.sqrt(1 / 32)))
        self.conv2_db = tf.Variable(tf.zeros([32]))

        self.conv3_w=tf.Variable(tf.random_normal([3,3,32,64],dtype=tf.float32,stddev=tf.sqrt(1/64)))
        self.conv3_b=tf.Variable(tf.zeros([64]))
        self.conv3_dw = tf.Variable(tf.truncated_normal([3, 3, 64, 1], dtype=tf.float32, stddev=tf.sqrt(1 / 64)))
        self.conv3_db = tf.Variable(tf.zeros([64]))

        self.fc_out_w=tf.Variable(tf.random_normal([128,20],dtype=tf.float32,stddev=tf.sqrt(1/20)))
        self.fc_out_b=tf.Variable(tf.zeros([20]))

    def forward(self):
        self.conv1=tf.nn.leaky_relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.x,self.conv1_w,strides=[1,1,1,1],padding='SAME')+self.conv1_b))
        self.conv1d=tf.nn.leaky_relu(tf.nn.depthwise_conv2d(self.conv1,self.conv1_dw,strides=[1,2,2,1],padding='SAME')+self.conv1_db)

        self.conv2=tf.nn.leaky_relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.conv1d,self.conv2_w,strides=[1,1,1,1],padding='SAME')+self.conv2_b
        ))
        self.conv2d = tf.nn.leaky_relu(
            tf.nn.depthwise_conv2d(self.conv2, self.conv2_dw, strides=[1, 2, 2, 1], padding='SAME') + self.conv2_db)

        self.conv3 = tf.nn.leaky_relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.conv2d, self.conv3_w, strides=[1, 1, 1, 1], padding='SAME') + self.conv3_b
        ))
        self.conv3d = tf.nn.leaky_relu(
            tf.nn.depthwise_conv2d(self.conv3, self.conv3_dw, strides=[1, 2, 2, 1], padding='SAME') + self.conv3_db)
        _,h,w,c=self.conv3d.shape.as_list()
        self.lstm_in=tf.transpose(self.conv3d,[0,2,1,3])
        self.lstm_in=tf.reshape(self.lstm_in,[-1,w,h*c])
        cell = tf.nn.rnn_cell.BasicLSTMCell(128, forget_bias=1.0, state_is_tuple=True)
        init_state = cell.zero_state(100, dtype=tf.float32)  #

        outputs, final_state = tf.nn.dynamic_rnn(cell, self.lstm_in, initial_state=init_state, time_major=False)
        self.outputs = tf.transpose(outputs, [1, 0, 2])[-1]   #  [100,128]
        print(self.outputs)

        self.fc_out=tf.matmul(self.outputs,self.fc_out_w)+self.fc_out_b
        self.out=tf.reshape(self.fc_out,[-1,20])
        print(self.out)

    def backward(self):
        global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(0.001, global_step, 100, 0.96, staircase=True)
        self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_,logits=self.out))
        self.opt=tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=global_step)    ###########
        # self.opt = tf.train.MomentumOptimizer(self.learning_rate).minimize(self.loss, global_step=global_step)

        # self.label_argmax = tf.arg_max(self.y_, 2)
        # self.out_argmax = tf.arg_max(self.out, 2)
        # correct_prediction = tf.equal(self.label_argmax, self.out_argmax)
        # rst = tf.cast(correct_prediction, "float")
        # self.accuracy = tf.reduce_mean(rst)


if __name__=='__main__':
    net=Cnn2Lstm()
    net.forward()
    net.backward()
    init=tf.global_variables_initializer()
    saver=tf.train.Saver()
    train_data = train_shuffle_batch(train_filename, [80, 280, 1], 100)

    test_data = test_shuffle_batch(test_filename, [80, 280, 1], 100)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        sess.run(init)
        x=[]
        y=[]

        # saver.restore(sess,'./cnn_rnn_save.dpk')
        for i in range(100000):
            train_img, train_label = sess.run(train_data)
            train_img = train_img / 255 - 0.5
            train_label = train_label / 26.

            train_img=train_img.reshape([-1,80,280,1])
            _, loss= sess.run([net.opt, net.loss],
                                         feed_dict={net.x: train_img, net.y_: train_label})
            x.append(i)
            y.append(loss)
            plt.plot(x, y, 'red')
            plt.pause(0.01)
            plt.clf()
            print(loss)
            # train_count = 0
            # for indexxx in range(len(o.tolist())):
            #     if o[indexxx].tolist() == l[indexxx].tolist():
            #         train_count += 1
            #     else:
            #         pass
            # train_acc = train_count / 100.
            if i%100==0:
                test_img, test_label = sess.run(train_data)
                test_img = test_img / 255 - 0.5
                test_label = test_label / 26.

                test_img=test_img.reshape([-1,80,280,1])
                loss1 = sess.run(net.loss,
                                             feed_dict={net.x: test_img, net.y_: test_label})
                saver.save(sess,'./cnn_rnn_save.dpk')
            # print('第几次的样本是:{}'.format(l[0]))
            # print('第几次的输出是:{}'.format(o[0]))
            #     test_count=0
            #     for indexxx in range(len(o1.tolist())):
            #         if o1[indexxx].tolist()==l1[indexxx].tolist():
            #             test_count+=1
            #         else:
            #             pass
            #     test_acc=test_count/100.

                print('第{}次测试集的误差为;{}'.format(i,loss1))
                print('第{}次的误差是:{}'.format(i, loss))
                # print('测试集样本：{}'.format(l1[0]))
                # print('输出数据:{}'.format(o1[0]))
                #


