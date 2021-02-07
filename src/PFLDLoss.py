# coding:utf-8
# loss function of yolo
import tensorflow as tf


class PFLDLoss():
    def __init__(self):
        pass

    def pfld_loss(self, batch_size, Landmarks_truth, Attributes_truth, Eulers_Truth, Landmarks_Pre, Euler_Pre):

        attributes_w_n = tf.to_float(Attributes_truth[:, 1:6])
        mat_ratio = tf.reduce_mean(attributes_w_n, axis=0)  
        mat_ratio = tf.map_fn(lambda x: (tf.cond(x > 0, lambda: 1/x, lambda: float(batch_size))), mat_ratio)
        attributes_w_n = tf.convert_to_tensor(attributes_w_n * mat_ratio)
        attributes_w_n = tf.reduce_sum(attributes_w_n, axis=1)

        _sum_k = tf.reduce_sum(tf.map_fn(lambda x: 1 - tf.cos(abs(x)), Eulers_Truth - Euler_Pre), axis=1)
        loss_sum = tf.reduce_sum(tf.square(Landmarks_truth - Landmarks_Pre), axis=1)
        loss_sum = tf.reduce_mean(loss_sum * _sum_k * attributes_w_n)

        return loss_sum
