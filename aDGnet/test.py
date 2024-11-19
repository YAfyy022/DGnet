#coding=utf-8
import torch
import torch.nn as nn
import sys
from tqdm import tqdm
from aDGnet.utils.config import input_size, channels, nums
from utils.read_dataset import read_dataset
from utils.auto_laod_resume import auto_load_resume
from networks.model import MainNet

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")

# dataset

# root = '/data0/hwl_data/FGVC/inat-2018/18-test'  # dataset path
# pth_path = '/data0/hwl_data/pth/hwl/18-74.1/epoch119.pth' # model path
# num_classes = 2971
# root = '/data0/hwl_data/FGVC/iNat_2021_MINI'  # dataset path
# pth_path = "/data0/hwl_data/pth/1/21-77.49/epoch163.pth"     # model path
# num_classes = 4721
root = '/data0/hwl_data/FGVC/Aircraft'  # dataset path
pth_path = "/data0/hwl_data/pth/hwl/air-94.4/epoch177.pth" # model path
num_classes = 100



batch_size = 12

#load dataset
_, testloader = read_dataset(input_size, batch_size, root)

# 定义模型
model = MainNet(nums = nums, num_classes=num_classes, channels=channels)

model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()


#加载checkpoint
if os.path.exists(pth_path):
    epoch = auto_load_resume(model, pth_path, status='test')
else:
    sys.exit('There is not a pth exist.')



print('Testing')
raw_correct = 0
object_correct = 0
model.eval()
with torch.no_grad():
    for i, data in enumerate(tqdm(testloader)):

        x, y = data
        x = x.to(DEVICE)
        y = y.to(DEVICE)


        raw_logits, object_logits, _ = model(x, 'test', DEVICE)


        # local
        pred = object_logits.max(1, keepdim=True)[1]
        object_correct += pred.eq(y.view_as(pred)).sum().item()

    print('\nObject branch accuracy: {}/{} ({:.2f}%)\n'.format(
            object_correct, len(testloader.dataset), 100. * object_correct / len(testloader.dataset)))