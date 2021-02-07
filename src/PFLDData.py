# coding:utf-8
# load voc dataset
import numpy as np
from src import Log
from utils import tools
import random
import cv2
import os
from os import path


class WFLWData():
    def __init__(self, batch_size, anno_dir, pt_num, width=112, height=112, training=True):
        self.anno_dirs = anno_dir  #
        self.batch_size = batch_size
        self.point_num = pt_num
        self.train_imgs_paths = []
        self.train_land_marks = []
        self.train_attributes = []  # 6 elems
        self.train_euler_angles = []  # lt, rb

        self.test_imgs_paths = []
        self.test_land_marks = []
        self.test_attributes = []  # 6 elems
        self.test_euler_angles = []  # lt, rb
        #
        self.width = width
        self.height = height
        self.training = training
        self.train_num_imgs = 0
        self.test_num_imgs = 0

        self.__init_args()

    # initial all parameters
    def __init_args(self):
        Log.add_log("message: begin to initial images path")

        if self.training == True:
            train_file = os.path.join(self.anno_dirs, "train_data", "list.txt")  # !!!
            print("TRAIN_DATA: ", train_file)
            with open(train_file) as train:
                for train_line in train.readlines():
                    gts = train_line.split()
                    # print(len(gts))
                    self.train_imgs_paths.append(os.path.join(gts[0]))
                    self.train_land_marks.append(gts[1:197])
                    self.train_attributes.append(gts[197:203])  # 6 elems
                    self.train_euler_angles.append(gts[203:206])  # lt, rb
                    self.train_num_imgs += 1
            """ File exists check """
            for i in range(len(self.train_imgs_paths)):
                if os.path.exists(self.train_imgs_paths[i]) == False:
                    raise ValueError("File %s not exists" % self.train_imgs_paths[i])
            print("Total train image num: ", self.train_num_imgs)

        if self.training == False:
            test_file = os.path.join(self.anno_dirs, "test_data", "list.txt")
            with open(test_file) as test:
                for test_line in test.readlines():
                    gts = test_line.split()
                    self.test_imgs_paths.append(gts[0])
                    self.test_land_marks.append(gts[1:197])
                    self.test_attributes.append(gts[197:203])  # 6 elems
                    self.test_euler_angles.append(gts[203:206])  # lt, rb
                    self.test_num_imgs += 1
            """ File exists check """
            for i in range(len(self.test_imgs_paths)):
                if os.path.exists(self.test_imgs_paths[i]) == False:
                    raise ValueError("File %s not exists" % self.test_imgs_paths[i])
            print("Total test image num: ", self.test_num_imgs)

        return

    # read image
    def read_img(self, img_file):
        '''
        read img_file, and resize it
        return:img, RGB & float
        '''
        img = tools.read_img(img_file)
        if img is None:
            return None, None, None, None
        ori_h, ori_w, _ = img.shape
        img = cv2.resize(img, (self.width, self.height))

        # flip image
        # if np.random.random() < self.flip_img:
        #     self.is_flip = True
        #     img = cv2.flip(img, 1)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # SAI-DEBUG

        # gray
        # if np.random.random() < self.gray_img:
        #     tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #     # convert to 3 channel
        #     img = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)

        # random erase
        # if np.random.random() < self.erase_img:
        #     erase_w = random.randint(20, 100)
        #     erase_h = random.randint(20, 100)
        #     x = random.randint(0, self.width - erase_w)
        #     y = random.randint(0, self.height - erase_h)
        #     value = random.randint(0, 255)
        #     img[y:y + erase_h, x:x + erase_w, :] = value

        test_img = img

        img = img.astype(np.float32)
        img = img / 255.0

        # gasuss noise
        # if np.random.random() < self.gasuss:
        #     noise = np.random.normal(0, 0.01, img.shape)
        #     img = img + noise
        #     img = np.clip(img, 0, 1.0)
        return img, ori_w, ori_h, test_img

    # read label file
    def read_label(self, curr_index):
        '''
        '''
        if self.training == True:
            land_mark = self.train_land_marks[curr_index]
            attribute = self.train_attributes[curr_index]
            euler_angle = self.train_euler_angles[curr_index]

        if self.training == False:
            land_mark = self.test_land_marks[curr_index]
            attribute = self.test_attributes[curr_index]
            euler_angle = self.test_euler_angles[curr_index]

        return land_mark, attribute, euler_angle

    # remove broken file
    def __remove(self, img_file, xml_file):
        self.imgs_path.remove(img_file)
        self.labels_path.remove(xml_file)
        self.num_imgs -= 1
        if not len(self.imgs_path) == len(self.labels_path):
            print("after delete file: %s，the number of label and picture is not equal" % (img_file))
            assert (0)
        return

        # load batch_size images

    def __get_data(self):
        '''
        load  batch_size labels and images
        return:imgs, label_y1, label_y2, label_y3
        '''
        imgs = []
        land_marks = []
        attributes = []  # 6 elems
        euler_angles = []  # lt, rb

        count = 0
        while count < self.batch_size:
            if self.training == True:
                curr_index = random.randint(0, self.train_num_imgs - 1)
            if self.training == False:
                curr_index = random.randint(0, self.test_num_imgs - 1)

            if self.training == True:
                img_name = self.train_imgs_paths[curr_index]

            img, ori_w, ori_h, test_img = self.read_img(img_name)
            if img is None:
                Log.add_log("img file'" + img_name + "'reading exception, will be deleted")
                # self.__remove(img_name, label_name)
                continue

            land_mark, attribute, euler_angle = self.read_label(curr_index)

            imgs.append(img)
            land_marks.append(land_mark)
            attributes.append(attribute)
            euler_angles.append(euler_angle)
            count += 1

        imgs_array = np.asarray(imgs)
        land_marks_array = np.asarray(land_marks)
        attributes_array = np.asarray(attributes)
        euler_angles_array = np.asarray(euler_angles)

        return imgs_array, land_marks_array, attributes_array, euler_angles_array

    # Iterator
    def __next__(self):
        '''    get batch images    '''
        return self.__get_data()

    def __len__(self):
        if(self.training == True):
            return self.train_num_imgs
        if(self.training == False):
            return self.test_num_imgs



if __name__ == '__main__':
    AnnoPath = "/1T/001_AI/003_PFLD/001_AL/003_LandmarkDetect"
    Data_wflw = WFLWData(1, AnnoPath, 98, 112, 112, training=True)  # 98point with
    image, land_mark, attribute, euler_angle = next(Data_wflw)
    cv2.imwrite("/share/PFLD_WFLW.jpg", image[0]*255.0)
    print(land_mark.shape)
    # print(land_mark)
    print(attribute.shape)
    print(euler_angle.shape)
    print(type(land_mark))

    image_t = image[0]*255.0
    for i in range(int(land_mark.shape[1]/2)):
        cv2.circle(image_t, (int(float(land_mark[0][i*2+0])*image_t.shape[1]), int(float(land_mark[0][i*2+1])*image_t.shape[0])), 2, (0, 0, 255), -1)

    cv2.imwrite("/1T/001_AI/003_PFLD/001_AL/003_LandmarkDetect/ttt.jpg", image_t)