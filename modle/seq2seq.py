import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

from modle.label_one_hot import *

class Seq2Seq:
    def __init__(self):
        self.batch_size=100
        self.inputs=60*3
        self.step=160
        self.encoder_size=256
        self.decoder_size=256
        self.output_dim=27
        self.encoder_w=tf.Variable(tf.random_normal([self.inputs,self.encoder_size]))
        self.encoder_b=tf.Variable(tf.zeros([self.encoder_size]))

        self.en_outputs=None
        self.de_output=None
        self.decoder_w=tf.Variable(tf.random_normal([self.decoder_size,self.output_dim]))
        self.decoder_b=tf.Variable(tf.zeros([self.output_dim]))
        self.x=tf.placeholder(tf.float32,[None,self.step,self.inputs])
        self.y_=tf.placeholder(tf.float32,[None,20,27])

    def encoder(self):
        x_in=tf.reshape(self.x,[-1,self.inputs])    #[100*120,60*3]
        encoder_in=tf.matmul(x_in,self.encoder_w)+self.encoder_b   #[100*120,128]
        encoder_in=tf.reshape(encoder_in,[-1,self.step,self.encoder_size])   #[100,120,128]

        cell=tf.nn.rnn_cell.BasicLSTMCell(self.encoder_size,forget_bias=1.0,state_is_tuple=True)
        init_state = cell.zero_state(self.batch_size, dtype=tf.float32)

        outputs, final_state = tf.nn.dynamic_rnn(cell, encoder_in, initial_state=init_state, time_major=False)
        self.en_outputs = tf.transpose(outputs, [1, 0, 2])[-1]   #[100,128]   #最后一个时刻的输出

    def decoder(self):
        with tf.name_scope("decoder") as scope:
            decoder_input=self.en_outputs   #【100，128】
            decoder_input = tf.expand_dims(decoder_input, 1)
            decoder_input = tf.tile(decoder_input, [1, 20, 1])  # [100,20,128]
            print(decoder_input)

            cell = tf.nn.rnn_cell.BasicLSTMCell(self.decoder_size, forget_bias=1.0, state_is_tuple=True)
            init_state = cell.zero_state(self.batch_size, dtype=tf.float32)  #

            outputs, final_state = tf.nn.dynamic_rnn(cell, decoder_input, initial_state=init_state, time_major=False,
                                                 scope=scope)
            print(outputs)
            final_put = tf.reshape(outputs,[-1, self.encoder_size])
            final_put1 = tf.matmul(final_put, self.decoder_w) + self.decoder_b  # [100*20,27]
            self.de_outputs = tf.reshape(final_put1, [-1, 20, 27])  # [100,4,10]


    def forward(self):
        self.encoder()
        self.decoder()
    def backward(self):
        self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.de_outputs,labels=self.y_))
        self.opt=tf.train.AdamOptimizer(0.0001).minimize(self.loss)
        self.label_argmax=tf.arg_max(self.y_,2)
        self.out_argmax=tf.arg_max(self.de_outputs,2)
        correct_prediction = tf.equal(self.label_argmax, self.out_argmax)
        rst = tf.cast(correct_prediction, "float")
        self.accuracy = tf.reduce_mean(rst)


if __name__=='__main__':
    net=Seq2Seq()
    net.forward()
    net.backward()
    init=tf.global_variables_initializer()
    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        x=[]
        y=[]
        # saver.restore(sess,'./seq2seq.dpk')
        for i in range(100000):
            train_img, train_labels = infordata.__getitem__(100)
            train_img=train_img.reshape([-1,160,180])
            _,loss,ac,l,o=sess.run([net.opt,net.loss,net.accuracy,net.label_argmax,net.out_argmax],feed_dict={net.x:train_img,net.y_:train_labels})
            # x.append(i)
            # y.append(loss)
            # plt.plot(x,y,'red')
            # plt.pause(0.01)
            # plt.clf()
            if i%500==0:
                saver.save(sess,'./seq2seq.dpk')
            # print('第几次的样本是:{}'.format(l[0]))
            # print('第几次的输出是:{}'.format(o[0]))


            # print('第几次的样本是:{}'.format(l[0]))
            # print('第几次的输出是:{}'.format(o[0]))

                print('第{}次的误差是:{}，精度是:{}'.format(i,loss,ac))












