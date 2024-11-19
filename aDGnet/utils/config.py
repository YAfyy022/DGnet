
CUDA_VISIBLE_DEVICES = '4'  # The current version only supports one GPU training


model_name = ''

batch_size =8
save_interval = 1
eval_trainset = 100    #evaluate trainset
end_epoch = 300
max_checkpoint_num = 200
init_lr = 0.001
lr_milestones = [60, 100]
lr_decay_rate = 0.1
weight_decay = 1e-4
stride = 32
channels = 2048
input_size = 384
nums = 4


pretrain_path = 'C:/Users/yingqi/Desktop/毕业论文1/IGCNN/code/DG-Net/pretrained/resnet50-19c8e357.pth'


model_path = 'C:/Users/yingqi/Desktop/毕业论文1/IGCNN/code/DG-Net/model_path'  # pth save path
root = 'C:/Users/yingqi/Desktop/毕业论文1/IGCNN/data/CUB_200_2011/dataset'  # dataset path
num_classes = 200
# model_path = '/data0/hwl_data/pth/21-Plant'  # pth save path
# root = '/data0/hwl_data/FGVC/iNat_2021_MINI'  # dataset path
# num_classes = 4721
# model_path = '/data0/hwl_data/pth/hwl/18-Plant'  # pth save path
# root = '/data0/hwl_data/FGVC/inat-2018'  # dataset path
# num_classes = 2971



