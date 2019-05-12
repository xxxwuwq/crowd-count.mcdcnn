# -*-coding:utf-8-*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import slim as slim
import numpy as np


class Model(object):
    """
    multi-column dialted convolutional neural network
    """
    def __index__(self, name='mdcnn'):
        self.name = name

    def mcdcnn(self):
        INPUT = tf.placeholder(tf.float32, shape=(None, None, None, 3))
        GT_DMP = tf.placeholder(tf.float32, shape=(None, None, None, 1))
        GT_CNT = tf.placeholder(tf.float32, shape=(None, 1))

        with tf.name_scope('norm_data'):
            x = (INPUT - 127.5) / 128.0

        with tf.name_scope('network'):
            with tf.name_scope('large'):
                with slim.arg_scope([slim.conv2d], padding='SAME', rate=2):
                    net1 = slim.conv2d(x, 16, [9, 9])
                    net1 = slim.max_pool2d(net1, [2, 2], 2)
                    net1 = slim.conv2d(net1, 32, [7, 7])
                    net1 = slim.max_pool2d(net1, [2, 2], 2)
                    net1 = slim.conv2d(net1, 16, [7, 7])
                    net1 = slim.conv2d(net1, 8, [7, 7])

            with tf.name_scope('medium'):
                with slim.arg_scope([slim.conv2d], padding='SAME', rate=2):
                    net2 = slim.conv2d(x, 20, [7, 7])
                    net2 = slim.max_pool2d(net2, [2, 2], 2)
                    net2 = slim.conv2d(net2, 40, [5, 5])
                    net2 = slim.max_pool2d(net2, [2, 2], 2)
                    net2 = slim.conv2d(net2, 20, [5, 5])
                    net2 = slim.conv2d(net2, 10, [5, 5])

            with tf.name_scope('small'):
                with slim.arg_scope([slim.conv2d], padding='SAME', rate=2):
                    net3 = slim.conv2d(x, 24, [5, 5])
                    net3 = slim.max_pool2d(net3, [2, 2], 2)
                    net3 = slim.conv2d(net3, 48, [3, 3])
                    net3 = slim.max_pool2d(net3, [2, 2], 2)
                    net3 = slim.conv2d(net3, 24, [3, 3])
                    net3 = slim.conv2d(net3, 12, [3, 3])

            with tf.name_scope('merge'):
                with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=None):
                    net = tf.concat([net1, net2, net3], axis=3)
                    dmp = slim.conv2d(net, 1, [1, 1])               # density map #
                    cnt = tf.reduce_sum(dmp, axis=[1, 2, 3])        # crowd count #

            return dict(INPUT=INPUT, GT_DMP=GT_DMP, GT_CNT=GT_CNT, EST_DMP=dmp, EST_CNT=cnt)

    def losses(self, gt_dmp, gt_cnt, est_dmp, est_cnt, dweight=0.5, cweight=0.0):
        """
        损失函数
        :param gt_dmp:
        :param gt_cnt:
        :param est_dmp:
        :param est_cnt:
        :param dweight:
        :param cweight:
        :return:
        """
        dmp_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(gt_dmp, est_dmp)), axis=[1, 2, 3]))
        count_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(gt_cnt, est_cnt))))

        loss = dweight * dmp_loss + cweight * count_loss
        return loss


if __name__ == '__main__':
    print('测试模型推理是否正确:')
    model = Model()
    network = model.mcdcnn()
    INPUT, GT_DMP, GT_CNT, EST_DMP, EST_CNT = network['INPUT'], network['GT_DMP'], network['GT_CNT'], \
                                              network['EST_DMP'], network['EST_CNT']
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        inputs = np.ones((1, 256, 256, 3))
        gt_dmp = np.ones((1, 64, 64, 1))
        gt_cnt = np.sum(gt_dmp, axis=(1, 2))
        print(gt_cnt.shape)
        result = sess.run([EST_DMP, EST_CNT], feed_dict={INPUT: inputs, GT_DMP: gt_dmp, GT_CNT: gt_cnt})
        est_dmp, est_cnt = result[0], result[1]
        print('the inference result', est_dmp.shape, est_cnt)
