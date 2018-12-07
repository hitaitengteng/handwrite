import tensorflow as tf
import matplotlib.pyplot as plt

from  chinese_handwrite.tf_train_sample import *


batch_size=40
num_hidden=256
number_clas=640+2


class CtcModle:
    def __init__(self):
        self.x=tf.placeholder(dtype=tf.float32,shape=[None,168,448,1],name='input')
        self.y_=tf.sparse_placeholder(dtype=tf.int32)

        self.conv1_w = tf.Variable(tf.random_normal([3, 3, 1, 10], dtype=tf.float32, stddev=tf.sqrt(1 / 10)))
        self.conv1_b = tf.Variable(tf.zeros([10]))
        self.conv1_dw = tf.Variable(tf.random_normal([3, 3, 10, 1], dtype=tf.float32, stddev=tf.sqrt(1 / 10)))
        self.conv1_db = tf.Variable(tf.zeros([10]))

        self.conv2_w = tf.Variable(tf.random_normal([3, 3, 10, 16], dtype=tf.float32, stddev=tf.sqrt(1 / 16)))
        self.conv2_b = tf.Variable(tf.zeros([16]))
        self.conv2_dw = tf.Variable(tf.random_normal([3, 3, 16, 1], dtype=tf.float32, stddev=tf.sqrt(1 / 16)))
        self.conv2_db = tf.Variable(tf.zeros([16]))

        self.conv3_w = tf.Variable(tf.random_normal([3, 3, 16, 32], dtype=tf.float32, stddev=tf.sqrt(1 / 32)))
        self.conv3_b = tf.Variable(tf.zeros([32]))
        self.conv3_dw = tf.Variable(tf.random_normal([3, 3, 32, 1], dtype=tf.float32, stddev=tf.sqrt(1 / 32)))
        self.conv3_db = tf.Variable(tf.zeros([32]))

        self.conv4_w = tf.Variable(tf.random_normal([3, 3, 32, 128], dtype=tf.float32, stddev=tf.sqrt(1 / 128)))
        self.conv4_b = tf.Variable(tf.zeros([128]))
        self.conv4_dw = tf.Variable(tf.random_normal([3, 3, 128, 1], dtype=tf.float32, stddev=tf.sqrt(1 / 128)))
        self.conv4_db = tf.Variable(tf.zeros([128]))

        self.conv5_w = tf.Variable(tf.random_normal([3, 3, 128, 256], dtype=tf.float32, stddev=tf.sqrt(1 / 256)))
        self.conv5_b = tf.Variable(tf.zeros([256]))
        self.conv5_dw = tf.Variable(tf.random_normal([3, 3, 256, 1], dtype=tf.float32, stddev=tf.sqrt(1 / 256)))
        self.conv5_db = tf.Variable(tf.zeros([256]))

        self.fc_w1 = tf.Variable(tf.random_normal([256, 512], dtype=tf.float32, stddev=tf.sqrt(1 / 512)))
        self.fc_b1 = tf.Variable(tf.zeros([512]))

        self.fc_out_w = tf.Variable(tf.random_normal([512, number_clas], dtype=tf.float32, stddev=tf.sqrt(1 / number_clas)))
        self.fc_out_b = tf.Variable(tf.zeros([number_clas]))

    def forward(self):

        #卷积部分，5层卷积
        self.conv1 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.x, self.conv1_w, strides=[1, 1, 1, 1], padding='SAME') + self.conv1_b))
        self.conv1d = tf.nn.relu(
            tf.nn.depthwise_conv2d(self.conv1, self.conv1_dw, strides=[1, 2, 2, 1], padding='SAME') + self.conv1_db)    #84,224

        self.conv2 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.conv1d, self.conv2_w, strides=[1, 1, 1, 1], padding='SAME') + self.conv2_b))
        self.conv2d = tf.nn.relu(
            tf.nn.depthwise_conv2d(self.conv2, self.conv2_dw, strides=[1, 2, 2, 1], padding='SAME') + self.conv2_db)    #42,112

        self.conv3 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.conv2d, self.conv3_w, strides=[1, 1, 1, 1], padding='SAME') + self.conv3_b))
        self.conv3d = tf.nn.relu(
            tf.nn.depthwise_conv2d(self.conv3, self.conv3_dw, strides=[1, 2, 2, 1], padding='SAME') + self.conv3_db)   #  21,56

        self.conv4 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.conv3d, self.conv4_w, strides=[1, 1, 1, 1], padding='SAME') + self.conv4_b))
        self.conv4d = tf.nn.relu(
            tf.nn.depthwise_conv2d(self.conv4, self.conv4_dw, strides=[1, 2, 2, 1], padding='SAME') + self.conv4_db)   #10,   28

        self.conv5 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.conv4d, self.conv5_w, strides=[1, 1, 1, 1], padding='SAME') + self.conv5_b))
        self.conv5d = tf.nn.relu(
            tf.nn.depthwise_conv2d(self.conv5, self.conv5_dw, strides=[1, 2, 2, 1], padding='SAME') + self.conv5_db)   #5,14

        print(self.conv5d)
        #LSTM部分

        _,h,w,c=self.conv5d.shape.as_list()

        lstm_input=tf.transpose(self.conv5d,[0,2,1,3]) #[batch_size,w,h,c]
        lstm_input=tf.reshape(lstm_input,[-1,w,h*c])

        self.seq_len=tf.fill([batch_size],w)
        #细胞1
        cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
        #细胞2
        cell1 = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
        #合并两个细胞
        stack = tf.contrib.rnn.MultiRNNCell([cell, cell1], state_is_tuple=True)

        initial_state = stack.zero_state(batch_size, dtype=tf.float32)  # 100批
        outputs, final_state = tf.nn.dynamic_rnn(stack, lstm_input, self.seq_len, dtype=tf.float32,# [batch_size, max_stepsize, num_hidden]
                                                 initial_state=initial_state)      #[100,-1,128]
        print(outputs,'outputs')

        self.outputs = tf.reshape(outputs, [-1, num_hidden])  # [batch_size * max_stepsize, num_hidden]
        self.fc_1=tf.nn.relu(tf.layers.batch_normalization(tf.matmul(self.outputs,self.fc_w1)+self.fc_b1))
        self.fc_out=tf.matmul(self.fc_1,self.fc_out_w)+self.fc_out_b

        logits = tf.reshape(self.fc_out, [batch_size, -1, number_clas])
        self.logits=tf.transpose(logits,(1,0,2))   #[time ,batch,clas]

    def backward(self):
        self.loss=tf.nn.ctc_loss(labels=self.y_,
                                 inputs=self.logits,
                                 sequence_length=self.seq_len)

        self.cost=tf.reduce_mean(self.loss)
        self.opt=tf.train.AdamOptimizer(0.0001).minimize(self.loss)
        self.decoded, self.log_prob = \
            tf.nn.ctc_beam_search_decoder(self.logits,
                                          self.seq_len,
                                          merge_repeated=False)

        # 把稀疏矩阵转换成稠密矩阵，长度不足的序列用-1来填充
        self.dense_decoded = tf.sparse_tensor_to_dense(self.decoded[0], default_value=-1)
        self.acc = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.y_))




if __name__=='__main__':
    net=CtcModle()
    net.forward()
    net.backward()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        x=[]
        y=[]
        for i in range(100000):
            train_img,train_labels=infordata.__getitem__(batch_size)
            print('label', train_labels[0])
            train_img = train_img.reshape([-1, 168, 448, 1])
            train_labels = sparse_tuple_from(train_labels)

            _, loss, out, decoded11 = sess.run(
                [net.opt, net.cost, net.logits, net.dense_decoded],
                feed_dict={net.x: train_img, net.y_: train_labels})
            x.append(i)
            y.append(loss)
            plt.plot(x,y,'red')
            plt.pause(0.01)
            plt.clf()

            print('out',decoded11[0])













