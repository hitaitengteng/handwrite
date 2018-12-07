import tensorflow as tf
import tensorflow.contrib.slim as slim


class VGG16(object):
  def __init__(self):
    self._scope = 'vgg_16'
    self._variables_to_fix = {}

  def vgg16(self, inputs, istraining):
    with tf.variable_scope(self._scope):
      with slim.arg_scope([slim.conv2d, slim.fully_connected],
                          activation_fn=tf.nn.relu,             #激活函数
                          weights_initializer=tf.truncated_normal_initializer(0.0, 0.001),   #权重初始化
                          weights_regularizer=slim.l2_regularizer(0.0005), trainable=False):     #权重正则化
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], trainable=False, scope='conv1')   #2个相同的卷积操作
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], trainable=False, scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], trainable=False, scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], trainable=False, scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], trainable=istraining, scope='conv5')
        net = tf.reshape(net, [-1, 3*3*512])
        net = slim.fully_connected(net, 4096, trainable=istraining, scope='fc6')
        net = slim.dropout(net, 0.5, scope='dropout6')
        net = slim.fully_connected(net, 7, activation_fn=None, trainable=istraining, scope='fc7')
        # net = tf.nn.softmax(net)
        # print(net)
        # net = slim.dropout(net, 0.5, scope='dropout7')
        # net = slim.fully_connected(net, 7, activation_fn=None, trainable=istraining, scope='fc8')
    return net

  def get_variables_to_restore(self, variables):
    variables_to_restore = []
    # var_keep_dic = []

    for v in variables:
      # exclude the conv weights that are fc weights in vgg16
      if v.name == (self._scope + '/fc6/weights:0') or v.name==(self._scope + '/fc6/biases:0') or \
              v.name == (self._scope + '/fc7/weights:0') or v.name==(self._scope + '/fc7/biases:0'):
        self._variables_to_fix[v.name] = v
        continue
      # exclude the first conv layer to swap RGB to BGR
      # if v.name == (self._scope + '/conv1/conv1_1/weights:0'):
      #   self._variables_to_fix[v.name] = v
      #   continue
      if v.name == (self._scope + '/fc8/weights:0') or v.name == (self._scope + '/fc8/biases:0'):
        continue

      variables_to_restore.append(v)

    return variables_to_restore

  def fix_variables(self, sess, pretrained_model):
    print('Fix VGG16 layers..')
    with tf.variable_scope('Fix_VGG16') as scope:
      with tf.device("/cpu:0"):
        # fix the vgg16 issue from conv weights to fc weights
        # fix RGB to BGR
        fc6_conv = tf.get_variable("fc6_conv", [7, 7, 512, 4096], trainable=False)
        fc7_conv = tf.get_variable("fc7_conv", [1, 1, 4096, 4096], trainable=False)
        conv1_rgb = tf.get_variable("conv1_rgb", [3, 3, 3, 64], trainable=False)
        restorer_fc = tf.train.Saver({self._scope + "/fc6/weights": fc6_conv,
                                      self._scope + "/fc7/weights": fc7_conv,
                                      self._scope + "/conv1/conv1_1/weights": conv1_rgb})
        restorer_fc.restore(sess, pretrained_model)

        sess.run(tf.assign(self._variables_to_fix[self._scope + '/fc6/weights:0'], tf.reshape(fc6_conv,
                                                                                              self._variables_to_fix[
                                                                                                self._scope + '/fc6/weights:0'].get_shape())))
        sess.run(tf.assign(self._variables_to_fix[self._scope + '/fc7/weights:0'], tf.reshape(fc7_conv,
                                                                                              self._variables_to_fix[
                                                                                                self._scope + '/fc7/weights:0'].get_shape())))
        sess.run(tf.assign(self._variables_to_fix[self._scope + '/conv1/conv1_1/weights:0'],
                           tf.reverse(conv1_rgb, [2])))


