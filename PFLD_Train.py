# coding:utf-8

import time
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
tf_config = ConfigProto()
tf_config.gpu_options.allow_growth = True
session = InteractiveSession(config=tf_config)

from os import path

from src import PFLDModel
from src import PFLDUtils
from src import PFLDData
from src import PFLDLoss
from src import Log

from src import Learning_rate as Lr

from src import Optimizer


AnnoPath = "/1T/001_AI/003_PFLD/001_AL/003_LandmarkDetect"
total_epoch = 80000
batch_size = 256
total_data_cnt = 0
train_batch_size = 8
test_batch_size = 1
total_points = 98
input_width = 112
input_height = 112
save_per_epoch = 2

lr_type = 'piecewise'
lr_init = 0.0001
lr_lower =1e-6                  # minimum learning rate
piecewise_boundaries = [1, 10, 50, 100]   #  for piecewise
piecewise_values = [2e-4, 2e-4, 2e-4, 1e-4, 1e-5]   # piecewise learning rate

model_path = "/1T/001_AI/003_PFLD/003_Model"
model_name = "PFLDModel"

# configure the optimizer
optimizer_type = 'adam' # type of optimizer
momentum = 0.9          #
weight_decay = 0.000001


log_dir = "/1T/001_AI/003_PFLD/004_Logs"



def Train_PFLD():
    global_step = tf.Variable(0, trainable=False)

    inputs = tf.placeholder(tf.float32, [None, 112, 112, 3])
    Landmarks_GTruth = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 2*total_points])
    Attributes_GTruth = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 6])
    Eulers_GTruth = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 3])

    print("inputs.name", inputs.name)
    print("Landmarks_GTruth.name", Landmarks_GTruth.name)
    print("Attributes_GTruth.name", Attributes_GTruth.name)
    print("Eulers_GTruth.name", Eulers_GTruth.name)

    Data_wflw_train = PFLDData.WFLWData(train_batch_size, AnnoPath, total_points, input_width, input_height, training=True)  # 98point with
    Data_wflw_test = PFLDData.WFLWData(test_batch_size, AnnoPath, total_points, input_width, input_height, training=True)  # 98point with
    total_train_data_cnt = len(Data_wflw_train)
    total_test_data_cnt = len(Data_wflw_test)

    train_steps_per_epoch = total_train_data_cnt / train_batch_size
    test_steps_per_epoch = total_test_data_cnt / test_batch_size


    # Forward
    PreLandmarks, PreEulers = PFLDModel.PFLD_Netework(inputs)
    print(PreLandmarks.shape)
    print(PreEulers.shape)

    loss = PFLDLoss.PFLDLoss().pfld_loss(batch_size, Landmarks_GTruth, Attributes_GTruth, Eulers_GTruth, PreLandmarks, PreEulers)
    tf.summary.scalar("loss", loss)  # tensorboard

    # reg = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # l2_loss = tf.add_n(reg)
    l2_loss = tf.compat.v1.losses.get_regularization_loss()
    tf.summary.scalar("l2_loss", l2_loss)  # tensorboard

    # lr = Lr.config_lr(lr_type, lr_init, lr_lower=lr_lower, \
    #                   piecewise_boundaries=piecewise_boundaries, \
    #                   piecewise_values=piecewise_values, epoch=total_epoch)
    lr = tf.constant(0.0001)
    optimizer = Optimizer.config_optimizer(optimizer_type, lr, momentum)

    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):  # 先执行update_ops，再执行梯度计算、修正和更新
        gvs = optimizer.compute_gradients(loss+l2_loss)  # 计算梯度
        clip_grad_var = [gv if gv[0] is None else [tf.clip_by_norm(gv[0], 100.), gv[1]] for gv in gvs]  # 修正梯度
        train_step = optimizer.apply_gradients(clip_grad_var, global_step=global_step)  # 更新权重


    init = tf.compat.v1.initialize_all_variables()
    saver = tf.compat.v1.train.Saver()

    merged = tf.summary.merge_all()  # tensorboard
    with tf.compat.v1.Session() as sess:
        sess.run(init)

        # tensor_name_ist = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
        # for i in range(len(tensor_name_ist)):
        #     print(tensor_name_ist[i])

        summery_writer = tf.summary.FileWriter(log_dir, sess.graph)  # tensorboard

        curr_epoch = 0
        while curr_epoch < total_epoch:
            for _ in range(int(train_steps_per_epoch)):  #
            # for _ in range(1):  #
                start = time.perf_counter()
                batch_imgs, batch_landmark, batch_attribute, batch_euler_angle = next(Data_wflw_train)
                _, step_, loss_, lr_, merged_loss = sess.run([train_step, global_step, loss, lr, merged],
                                                            feed_dict={inputs: batch_imgs, Landmarks_GTruth: batch_landmark, Attributes_GTruth: batch_attribute, Eulers_GTruth: batch_euler_angle})
                end = time.perf_counter()

                if step_ % 5 == 2:
                    print("step: %6d, epoch: %3d, loss: %.5g\t, width: %3d, height: %3d, lr:%.5g\t, time: %5f s"
                          % (step_, curr_epoch, loss_, input_width, input_height, lr_, end - start))
                    Log.add_loss(str(step_) + "\t" + str(loss_))
                    summery_writer.add_summary(merged_loss, global_step=step_)  # tensorboard

            curr_epoch += 1
            if curr_epoch % save_per_epoch == 0:
                # save ckpt model
                Log.add_log("message: save ckpt model, step=" + str(step_) + ", lr=" + str(lr_))
                saver.save(sess, path.join(model_path, model_name), global_step=step_)

        Log.add_log("message: save final ckpt model, step=" + str(step_))
        saver.save(sess, path.join(model_path, model_name), global_step=step_)


if __name__ == '__main__':
    Train_PFLD()
