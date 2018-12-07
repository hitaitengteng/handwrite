import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.python.ops import ctc_ops as ctc
from modle.label_one_hot import *
from modle.sparse import *


# num_classes = 28
# num_hidden = 64
# num_layers = 1
batch_size=16
class Cnn2Lstm():
    def __init__(self):
        self.x=tf.placeholder(dtype=tf.float32,shape=[None,80,280,1])
        self.y_= tf.sparse_placeholder(tf.int32)    #ctc_loss需要的稀疏矩阵
        # self.dp=tf.placeholder(dtype=tf.float32)
        # self.seq_len = tf.placeholder(tf.int32, [None])   #bach_size

        self.conv1_w=tf.Variable(tf.random_normal([3,3,1,10],dtype=tf.float32,stddev=tf.sqrt(1/10)))
        self.conv1_b=tf.Variable(tf.zeros([10]))
        self.conv1_dw = tf.Variable(tf.random_normal([3, 3, 10, 1], dtype=tf.float32, stddev=tf.sqrt(1 / 10)))
        self.conv1_db = tf.Variable(tf.zeros([10]))

        self.conv2_w=tf.Variable(tf.random_normal([3,3,10,16],dtype=tf.float32,stddev=tf.sqrt(1/16)))
        self.conv2_b=tf.Variable(tf.zeros([16]))
        self.conv2_dw = tf.Variable(tf.random_normal([3, 3, 16, 1], dtype=tf.float32, stddev=tf.sqrt(1 / 16)))
        self.conv2_db = tf.Variable(tf.zeros([16]))

        self.conv3_w=tf.Variable(tf.random_normal([3,3,16,32],dtype=tf.float32,stddev=tf.sqrt(1/32)))
        self.conv3_b=tf.Variable(tf.zeros([32]))
        self.conv3_dw = tf.Variable(tf.random_normal([3, 3, 32, 1], dtype=tf.float32, stddev=tf.sqrt(1 / 32)))
        self.conv3_db = tf.Variable(tf.zeros([32]))

        self.conv4_w = tf.Variable(tf.random_normal([3, 3, 32, 128], dtype=tf.float32, stddev=tf.sqrt(1 / 128)))
        self.conv4_b = tf.Variable(tf.zeros([128]))
        self.conv4_dw = tf.Variable(tf.random_normal([3, 3, 128, 1], dtype=tf.float32, stddev=tf.sqrt(1 / 128)))
        self.conv4_db = tf.Variable(tf.zeros([128]))

        self.fc_out_w=tf.Variable(tf.random_normal([256,28],dtype=tf.float32,stddev=tf.sqrt(1/26)))
        self.fc_out_b=tf.Variable(tf.zeros([28]))   #ru

    def forward(self):

        #卷积部分
        self.conv1=tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.x,self.conv1_w,strides=[1,1,1,1],padding='SAME')+self.conv1_b))
        self.conv1d=tf.nn.relu(tf.nn.depthwise_conv2d(self.conv1,self.conv1_dw,strides=[1,2,2,1],padding='SAME')+self.conv1_db)


        self.conv2=tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.conv1d,self.conv2_w,strides=[1,1,1,1],padding='SAME')+self.conv2_b
        ))
        self.conv2d = tf.nn.relu(
            tf.nn.depthwise_conv2d(self.conv2, self.conv2_dw, strides=[1, 2, 2, 1], padding='SAME')+self.conv2_db)



        self.conv3 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.conv2d, self.conv3_w, strides=[1, 1, 1, 1], padding='SAME') + self.conv3_b
        ))
        self.conv3d=tf.nn.relu(tf.nn.depthwise_conv2d(self.conv3,self.conv3_dw,strides=[1,2,2,1],padding='SAME')+self.conv3_db)

        self.conv4 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.conv3d, self.conv4_w, strides=[1, 1, 1, 1], padding='SAME') + self.conv4_b
        ))
        self.conv4d = tf.nn.relu(
            tf.nn.depthwise_conv2d(self.conv4, self.conv4_dw, strides=[1, 2, 2, 1], padding='SAME') + self.conv4_db)
        print(self.conv4d)

        #LSTM部分
        _,h,w,c=self.conv4d.shape.as_list()
        print(h,w,c)
        lstm_in=tf.transpose(self.conv4d,[0,2,1,3])    #[batch_size, feature_w, feature_h, out_channels]  [100,20,8,32]
        print(lstm_in,'lstm_in1')
        lstm_in=tf.reshape(lstm_in,[-1,w,h*c])
        print(lstm_in,'lstm_in2')
        self.seq_len = tf.fill([batch_size], w)

        cell = tf.contrib.rnn.LSTMCell(256, state_is_tuple=True)   #  128为自己设置的LSTM的隐藏层数量
        cell1 = tf.nn.rnn_cell.LSTMCell(256, state_is_tuple=True)

        stack = tf.contrib.rnn.MultiRNNCell([cell,cell1], state_is_tuple=True)   #合并两个LSTM、

        initial_state = stack.zero_state(batch_size, dtype=tf.float32)   #100批

        outputs, final_state = tf.nn.dynamic_rnn(stack, lstm_in,self.seq_len,dtype=tf.float32,initial_state=initial_state)
        #[batch_size, max_stepsize, num_hidden]  [100,-1,128]
        print(outputs,'outputs')



        self.outputs = tf.reshape(outputs, [-1, 256])  # [batch_size * max_stepsize, FLAGS.num_hidden]


        self.fc_out=tf.matmul(self.outputs,self.fc_out_w)+self.fc_out_b
        print(self.fc_out,'self.fc_out')
        logits = tf.reshape(self.fc_out, [batch_size, -1, 28])

        # Time major
        self.logits = tf.transpose(logits, (1, 0, 2))
        print(self.logits)



    def backward(self):
        self.loss = ctc.ctc_loss(labels=self.y_,
                                   inputs=self.logits,
                                   sequence_length=self.seq_len)
        self.cost = tf.reduce_mean(self.loss)

        self.opt = tf.train.AdamOptimizer(0.05).minimize(self.loss)
        self.decoded, self.log_prob = \
            tf.nn.ctc_beam_search_decoder(self.logits,
                                          self.seq_len,
                                          merge_repeated=False)
        #把稀疏矩阵转换成稠密矩阵，长度不足的序列用-1来填充
        self.dense_decoded = tf.sparse_tensor_to_dense(self.decoded[0], default_value=-1)
        self.acc = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0], tf.int32),self.y_))
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
    with tf.Session() as sess:
        sess.run(init)
        x=[]
        y=[]
        # saver.restore(sess,'./cnn_rnn_save.dpk')
        for i in range(100000):
            train_img, train_labels = infordata.__getitem__(batch_size)
            train_img=train_img.reshape([-1,80,280,1])
            train_label_len = []
            for ii in train_labels:
                train_label_len.append(len(ii))

            train_labels=sparse_tuple_from(train_labels)

            _, loss,out,acc,decoded11,log_prob1 = sess.run([net.opt, net.cost,net.logits,net.acc,net.decoded,net.log_prob],
                                         feed_dict={net.x: train_img, net.y_: train_labels})


            print(loss)
            print('acc: ',acc)
            # print(decode_sparse_tensor(decoded11)[0])
            print(decode_sparse_tensor(train_labels)[0])
            x.append(i)
            y.append(loss)
            plt.plot(x, y, 'red')
            plt.pause(0.01)
            plt.clf()
            if i%100==0:
                test_img,test_label=testdata.__getitem__(batch_size)
                test_img = test_img.reshape([-1, 80, 280, 1])
                test_label_len = []
                for i in train_labels:
                    test_label_len.append(len(i))
                test_label = sparse_tuple_from(test_label)
                loss1,test_acc = sess.run([net.cost,net.acc],feed_dict={net.x: test_img, net.y_: test_label})
                saver.save(sess,'./cnn_rnn_save.dpk')
            # print('第几次的样本是:{}'.format(l[0]))
            # print('第几次的输出是:{}'.format(o[0]))
                print('第{}次测试集的误差为;{},acc:{}'.format(i,loss1,test_acc))
                print('第{}次的误差是:{}'.format(i, loss, ))
                # print('测试集样本：{}'.format(l1[0]))
                # print('输出数据:{}'.format(o1[0]))
