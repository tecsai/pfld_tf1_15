# coding:utf-8
# result tests

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import tensorflow as tf
import numpy as np
import os
import config
from utils import tools
import cv2

from tensorflow.python.platform import gfile

# read picture
def read_img(img_name, width, height):
    img_ori = tools.read_img(img_name)
    if img_ori is None:
        return None, None
    img = cv2.resize(img_ori, (width, height))
    nw, nh = None, None

    show_img = img
    
    img = img.astype(np.float32)
    img = img/255.0
    # [416, 416, 3] => [1, 416, 416, 3]
    img = np.expand_dims(img, 0)
    return img, nw, nh, img_ori, show_img

sess = tf.Session()

# your pb_model path
pb_dir = "/1T/001_AI/003_PFLD/003_Model/model.pb"
# your class_num
# class_num = 80
# picture folder
test_imgs_folder = "/1T/001_AI/003_PFLD/005_TestImgs"
test_results_folder = "/1T/001_AI/003_PFLD/006_TestResults"
# your names file
# names_file = "./data/coco.names"

with gfile.FastGFile(pb_dir, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name="") 
sess.run(tf.global_variables_initializer())

# inputs
inputs = sess.graph.get_tensor_by_name('Placeholder:0')
# output
# 'concat_9', 'concat_10', 'concat_11'
pre_landmarks = sess.graph.get_tensor_by_name('PFLD_Netework/MS-FC/landmark_3:0')
pre_thetas = sess.graph.get_tensor_by_name('PFLD_Netework/Fc2/pre_theta:0')  # 

print("pre_landmarks.shape: ", pre_landmarks.shape)
print("pre_thetas.shape: ", pre_thetas.shape)

width, height = 112, 112

# word_dict = tools.get_word_dict(config.voc_names)     # for VOC
# word_dict = tools.get_word_dict(names_file)        # for COCO
# color_table = tools.get_color_table(class_num)

for name in os.listdir(test_imgs_folder):
    img_name = os.path.join(test_imgs_folder, name)
    if not os.path.isfile(img_name):
        continue
    img, nw, nh, img_ori, show_img = read_img(img_name, width, height)
    if img is None:
        print("message:'"+str(img)+"' picture read error")
    pre_landmarks_, pre_thetas_ = sess.run([pre_landmarks, pre_thetas], feed_dict={inputs:img})

    print("pre_landmarks_: ", pre_landmarks_.shape)
    print("pre_thetas_: ", pre_thetas_.shape)

    img_ori = cv2.resize(img_ori, (112, 112))
    # img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)  # SAI-DEBUG
    img_ori = tools.point_img(img_ori, pre_landmarks_, pre_thetas_)

    # print(img_ori.shape)

    cv2.imwrite(os.path.join(test_results_folder, name), img_ori)
    # cv2.imshow('img', img_ori)
    # cv2.waitKey(0)
