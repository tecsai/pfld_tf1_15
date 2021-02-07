# coding:utf-8
# configuration file

# ############# Basic configuration. #############
width = 416                     # image width in net
height = 416                    # image height in net
batch_size = 2
total_epoch = 450       # total epoch
save_per_epoch = 5        # per save_step save one model
data_debug = False       # load data in debug model (show pictures when loading images)
cls_normalizer = 1.0    # Loss coefficient of confidence
iou_normalizer = 0.07   # loss coefficient of ciou
iou_thresh = 0.5     # 
prob_thresh = 0.25      # 
score_thresh = 0.25     # 
val_score_thresh = 0.5      # 
val_iou_thresh = 0.213            # 
max_box = 50                # 
save_img = False             # save the result image when test the net

# ############# log #############
log_dir = '/1T/001_AI/logs'
log_name = 'log.txt'
loss_name = 'loss.txt'

# configure the leanring rate
lr_init = 2e-4                      # initial learning rate	# 0.00261
lr_lower =1e-6                  # minimum learning rate    
lr_type = 'piecewise'   # type of learning rate( 'exponential', 'piecewise', 'constant')
piecewise_boundaries = [1, 10, 50, 100]   #  for piecewise
piecewise_values = [2e-4, 2e-4, 2e-4, 1e-4, 1e-5]   # piecewise learning rate

# configure the optimizer
optimizer_type = 'momentum' # type of optimizer
momentum = 0.949          # 
weight_decay = 0.0005

# ############## training on own dataset ##############
class_num = 25
anchors = 12,19, 14,36, 20,26, 20,37, 23,38, 27,39, 32,35, 39,44, 67,96
model_path = "./checkpoint/"
model_name = "model"
name_file = './data/train.names'                # dataset's classfic names
train_file = './data/train.txt'
val_dir = "./test_pic"  # Test folder directory, which stores test pictures
save_dir = "./save"         # the folder to save result image

# ############## train on VOC ##############
voc_root_dir = "/2T/001_AI/VOC/VOC0712"  # root directory of voc dataset
voc_dir_ls = ['VOC2007', 'VOC2012']                # the version of voc dataset
voc_test_dir = "/2T/001_AI/VOC/VOC0712/voc_test_pic"                                                 # test pictures directory for VOC dataset
voc_save_dir = "/2T/001_AI/VOC/VOC0712/voc_save"                                                     # the folder to save result image for VOC dataset
voc_model_path = "/2T/001_AI/VOC/VOC0712/voc_models"                                                        # the folder to save model for VOC dataset
voc_model_name = "voc"                                          # the model name for VOC dataset
voc_names = "/2T/001_AI/VOC/VOC0712/voc.names"                             # the names of voc dataset
voc_class_num = 20
voc_anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326

# ############## PFLD ##############
pfld_dir = "/media/hyperfd/AI/001_FaceDetection/001_01_Face_DataSet/WFLW"
pfld_annotation_dir = "WFLW_annotations"
pfld_image_dir = "WFLW_images"
pfld_model_name = "pfld"
pfld_landmark_num = 98
