from src.PFLDUtils import *
import config

#  主网络 + 副网络
def PFLD_Netework(input): # 112 * 112 * 3
    with tf.name_scope('PFLD_Netework'):
        ##### Part1: Major Network -- 主网络 #####
        #layers1
        #input= [None,112,112,3]
        landmrk_num = config.pfld_landmark_num  #
        with tf.name_scope('layers1'):
            W_conv1 = weight_variable([3, 3, 3, 64],name='W_conv1')
            b_conv1 = bias_variable([64], name='b_conv1')
            x_image = tf.reshape(input, [-1, 112, 112, 3], name='input_X')
            x_image = Batch_Norm(x_image, is_training=True)
            h_conv_1 = conv2d(x_image, W_conv1, strides=[1, 2, 2, 1], padding='SAME') + b_conv1
        # layers2
        with tf.name_scope('layers1'):
            W_conv2 = weight_variable([3, 3, 64, 1], name='W_conv2')
            b_conv2 = bias_variable([64], name='b_conv2')
            h_conv_1 = Batch_Norm(h_conv_1, is_training=True)
            h_conv_2 = DW_Conv2d(h_conv_1, W_conv2) + b_conv2 # 56 * 56 * 64
        # Bottleneck   input = [56*56*64]
        with tf.name_scope('Mobilenet-V2'):
            with tf.name_scope('bottleneck_1'):
                h_conv_b1 = make_bottleneck_block(h_conv_2, 2, 64, stride=[1, 2, 2, 1], kernel=(3, 3)) # 28*28*64
                h_conv_b1 = make_bottleneck_block(h_conv_b1, 2, 64, stride=[1, 1, 1, 1], kernel=(3, 3))  # 28*28*64
                h_conv_b1 = make_bottleneck_block(h_conv_b1, 2, 64, stride=[1, 1, 1, 1], kernel=(3, 3))  # 28*28*64
                h_conv_b1 = make_bottleneck_block(h_conv_b1, 2, 64, stride=[1, 1, 1, 1], kernel=(3, 3))  # 28*28*64
                h_conv_b1 = make_bottleneck_block(h_conv_b1, 2, 64, stride=[1, 1, 1, 1], kernel=(3, 3))  # 28*28*64
            with tf.name_scope('bottleneck_2'):
                h_conv_b2 = make_bottleneck_block(h_conv_b1,2,128,stride=[1,2,2,1],kernel=(3,3)) # 14*14*128
            with tf.name_scope('bottleneck_3'):
                h_conv_b3 = make_bottleneck_block(h_conv_b2, 4, 128, stride=[1, 1, 1, 1], kernel=(3, 3)) # 14*14*128
                h_conv_b3 = make_bottleneck_block(h_conv_b3, 4, 128, stride=[1, 1, 1, 1], kernel=(3, 3))  # 14*14*128
                h_conv_b3 = make_bottleneck_block(h_conv_b3, 4, 128, stride=[1, 1, 1, 1], kernel=(3, 3))  # 14*14*128
                h_conv_b3 = make_bottleneck_block(h_conv_b3, 4, 128, stride=[1, 1, 1, 1], kernel=(3, 3))  # 14*14*128
                h_conv_b3 = make_bottleneck_block(h_conv_b3, 4, 128, stride=[1, 1, 1, 1], kernel=(3, 3))  # 14*14*128
                h_conv_b3 = make_bottleneck_block(h_conv_b3, 4, 128, stride=[1, 1, 1, 1], kernel=(3, 3))  # 14*14*128
            with tf.name_scope('bottleneck_4'):
                h_conv_b4 = make_bottleneck_block(h_conv_b3,2,16,stride=[1,1,1,1],kernel=(3,3))  # 14*14*16
        # S1
        with tf.name_scope('S1'):
            h_conv_s1 = h_conv_b4 # 14 * 14 * 16
        # s2
        with tf.name_scope('S2'):
            W_conv_s2 = weight_variable([3,3,16,32],name='W_conv_s2')
            b_conv_s2 = bias_variable([32],name='b_conv_s2')
            h_conv_s1 = Batch_Norm(h_conv_s1, is_training=True)
            h_conv_s2 = conv2d(h_conv_s1,W_conv_s2,strides=[1,2,2,1],padding='SAME') + b_conv_s2  # 7*7*32
        # S3
        with tf.name_scope('S3'):
            W_conv_s3 = weight_variable([7,7,32,128],name='W_conv_s3')
            b_conv_s3 = bias_variable([128],name='b_conv_s3')
            h_conv_s2 = Batch_Norm(h_conv_s2, is_training=True)
            h_conv_s3 = conv2d(h_conv_s2,W_conv_s3,strides=[1,1,1,1],padding='VALID') + b_conv_s3  # 1 * 1 * 128
 
        # MS-FC(多尺度全连接层)
        with tf.name_scope('MS-FC'):
            ########################111--收敛较快#################################
            W_conv_fc_s1 = weight_variable([14, 14, 16, 64], name='MS_FC_weight_s1')
            b_conv_fc_s1 = bias_variable([64], name='MS_FC_bias_s1')
            h_conv_s1 = Batch_Norm(h_conv_s1,is_training=True)
            h_conv_fc_s1 = conv2d(h_conv_s1,W_conv_fc_s1,strides=[1,1,1,1],padding='VALID') + b_conv_fc_s1 # 1*1*64
 
            W_conv_fc_s2 = weight_variable([7, 7, 32, 64], name='MS_FC_weight_s2')
            b_conv_fc_s2 = bias_variable([64], name='MS_FC_bias_s2')
            h_conv_s2 = Batch_Norm(h_conv_s2, is_training=True)
            h_conv_fc_s2 = conv2d(h_conv_s2, W_conv_fc_s2, strides=[1, 1, 1, 1],padding='VALID') + b_conv_fc_s2  # 1*1*64
 
            h_conv_s3 = Batch_Norm(h_conv_s3, is_training=True)
            h_conv_ms_fc = tf.concat([h_conv_fc_s1,h_conv_fc_s2,h_conv_s3],axis=3) # 1*1*256
            h_conv_ms_fc = tf.reshape(h_conv_ms_fc,(-1,1*1*256))
            W_ms_fc = weight_variable([1*1*256, landmrk_num*2], name='W_fc_s2')  #
            b_ms_fc = bias_variable([landmrk_num*2], name='b_fc_s2')
            pre_landmark = tf.add(tf.matmul(h_conv_ms_fc, W_ms_fc), b_ms_fc, name='landmark_3')
 
            #########################222--效果很差################################
            """
            W_conv_fc_s1 = tf.reshape(h_conv_s1,[-1,14*14*16])
            W_conv_fc_s2 = tf.reshape(h_conv_s2,[-1,7*7*32])
            W_conv_fc_s3 = tf.reshape(h_conv_s3,[-1,1*1*128])
            h_conv_ms_fc = tf.concat([W_conv_fc_s1,W_conv_fc_s2,W_conv_fc_s3],axis=1)
            h_conv_ms_fc1 = tf.reshape(h_conv_ms_fc,(-1,1*1*4832))
            # W_ms_fc0 = weight_variable([1*1*4832, 1024], name='W_fc_s2')
            # b_ms_fc0 = bias_variable([1024], name='b_fc_s2')
            # pre_land_mark0 = tf.add(tf.matmul(h_conv_ms_fc1,W_ms_fc0),b_ms_fc0)
            W_ms_fc = weight_variable([1 * 1 * 4832, 136], name='W_fc_s2')
            b_ms_fc = bias_variable([136], name='b_fc_s2')
            h_conv_ms_fc1 = Batch_Norm(h_conv_ms_fc1,is_training=True)
            pre_landmark = tf.add(tf.matmul(h_conv_ms_fc1, W_ms_fc), b_ms_fc, name='landmark_3')
            """
            ######################### 333最初用的 ################################
            """
            concat1 = crop_and_concat(h_conv_s1,h_conv_s2) # (?,7,7,78)
            concat2 = crop_and_concat(concat1,h_conv_s3) #(?,1,1,176)
            h_conv_ms_fc = tf.reshape(concat2, [-1, 1 * 1 * 176])
            W_ms_fc = weight_variable([1 * 1 * 176, 136], name='W_fc_s2')
            b_ms_fc = bias_variable([136], name='b_fc_s2')
            pre_landmark = tf.add(tf.matmul(h_conv_ms_fc, W_ms_fc), b_ms_fc, name='landmark_3')
            """
            ##### Part2: Auxiliary Network -- 副网络 #####
        # layers1
        # 副网络输入input： h_conv_b1 === [1,28,28,64]
        with tf.name_scope('Funet-layers1'):
            W_convfu_1 = weight_variable([3, 3, 64, 128],name='W_convfu_1')
            b_convfu_1 = bias_variable([128],name='b_convfu_1')
            h_convfu_1 = conv2d(h_conv_b1, W_convfu_1,strides=[1,2,2,1],padding='SAME') + b_convfu_1  # 14 * 14 * 128
        # layers2
        with tf.name_scope('Funet-layers2'):
            W_convfu_2 = weight_variable([3,3,128,128],name='W_convfu_2')
            b_convfu_2 = bias_variable([128],name='b_convfu_2')
            h_convfu_2 = conv2d(h_convfu_1,W_convfu_2,strides=[1,1,1,1],padding='SAME') + b_convfu_2 # 14 * 14 * 128
        # layers3
        with tf.name_scope('Funet-layers3'):
            W_convfu_3 = weight_variable([3,3,128,32],name='W_convfu_3')
            b_convfu_3 = bias_variable([32],name='b_convfu_3')
            h_convfu_3 = conv2d(h_convfu_2,W_convfu_3,strides=[1,2,2,1],padding='SAME') + b_convfu_3 # 7 * 7 * 32
        # layers4
        with tf.name_scope('Funet-layers4'):
            W_convfu_4 = weight_variable([7,7,32,128], name='W_convfu_4')
            b_convfu_4 = bias_variable([128], name='b_convfu_4')
            h_convfu_4 = conv2d(h_convfu_3,W_convfu_4,strides=[1,1,1,1],padding='VALID') + b_convfu_4 # 1 *  1 * 128
        ####### Fc ######
        # Fc1:
        with tf.name_scope('Fc1'):
            W_fu_fc1 = weight_variable([1 * 1 * 128, 32],name='W_fu_fc1')
            b_fc_s1 = bias_variable([32],name='b_fc_s1')
            h_convfu_4_fc = tf.reshape(h_convfu_4, [-1, 1 * 1 * 128])   # 1 * 128
            pre_theat_s1 = tf.matmul(h_convfu_4_fc, W_fu_fc1) + b_fc_s1 # 1 * 32
        # Fc2:
        with tf.name_scope('Fc2'):
            W_fu_fc2 = weight_variable([1 * 1 * 32, 3], name='W_fu_fc2')
            b_fc_s2 = bias_variable([3],name='b_fc2_s2')
            h_convfu_5_fc = tf.reshape(pre_theat_s1, [-1, 1 * 1 * 32])  # 1 * 32
            pre_euler = tf.add(tf.matmul(h_convfu_5_fc, W_fu_fc2), b_fc_s2, name='pre_theta')   # 1 * 3

        print("pre_landmark.name", pre_landmark.name)
        print("pre_euler.name", pre_euler.name)
 
    return pre_landmark, pre_euler