# coding:utf-8
# convert ckpt model to pb model

import tensorflow as tf
import config
from utils import tools
import cv2
import numpy as np
from src import Log
import os
from os import path
import time
# for save pb file
from tensorflow.python.framework import graph_util

from src import PFLDModel

# your pb model name
ckpt_file_dir="/1T/001_AI/003_PFLD/003_Model"
pd_dir = path.join(ckpt_file_dir, "model.pb")

total_points=98

def main():

    inputs = tf.placeholder(tf.float32, [None, 112, 112, 3])
    Landmarks_GTruth = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 2*total_points])
    Attributes_GTruth = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 6])
    Eulers_GTruth = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 3])
    # Forward
    PreLandmarks, PreEulers = PFLDModel.PFLD_Netework(inputs)
    print(PreLandmarks.shape)
    print(PreEulers.shape)



    init = tf.compat.v1.global_variables_initializer()

    saver = tf.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        saver.restore(sess, "/1T/001_AI/003_PFLD/003_Model/PFLDModel-5268750")

        # save  PB model
        out_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['Placeholder', 'Placeholder_1', 'Placeholder_2', 'Placeholder_3', 'PFLD_Netework/MS-FC/landmark_3', 'PFLD_Netework/Fc2/pre_theta'])  # "yolo/Conv_13/BiasAdd"
        saver_path = tf.train.write_graph(out_graph, "", pd_dir, as_text=False)
        print("saver path: ", saver_path)

if __name__ == "__main__":
    Log.add_log("message: convert ckpt model to pb model")
    main()