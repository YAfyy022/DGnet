import torch
from torch import nn
import torch.nn.functional as F
from networks import resnet
from config import pretrain_path

from aDGnet.mine_model.kvasir_model_2 import Guide_CNN
from aDGnet.utils.DSAS import DSAS
from aDGnet.utils.GRQL import GRQL







class MainNet(nn.Module):
    def __init__(self, nums, num_classes, channels):
        super(MainNet, self).__init__()
        self.num_classes = num_classes
        self.pretrained_model = resnet.resnet50(pretrained=True, pth_path=pretrain_path)
        self.model_crop = Guide_CNN(grad_layer='layer4', num_classes=200).to('cuda')
        self.rawcls_net = nn.Linear(channels, num_classes)
        # self.localcls_net = nn.Linear(channels, num_classes)   #no use
        self.nums = nums
        self.parts_att = GRQL(nums)

    def forward(self, x, status='test', DEVICE='cuda'):
        fm, embedding = self.pretrained_model(x)
        batch_size, channel_size, _, _ = fm.shape
        assert channel_size == 2048


        # raw image
        raw_logits = self.rawcls_net(embedding)


        result_img = DSAS(fm.detach())

        #object branch
        coordinates = torch.tensor(result_img)

        object_imgs = torch.zeros([batch_size, 3, 448, 448]).to(DEVICE) 
        for i in range(batch_size):
            [x0, y0, x1, y1] = coordinates[i]
            object_imgs[i:i + 1] = F.interpolate(x[i:i + 1, :, x0:(x1+1), y0:(y1+1)], size=(448, 448),
                                                mode='bilinear', align_corners=True) 


        object_fm, object_embeddings = self.pretrained_model(object_imgs.detach())
        crop_logits, crop_box = self.model_crop(object_imgs.detach())
        object_logits = self.rawcls_net(object_embeddings)



        #parts branch
        if status == "train":
            parts_coordinates = self.parts_att(object_fm)  

            parts_imgs = torch.zeros([batch_size, self.nums, 3, 224, 224]).to(DEVICE)  
            for i in range(batch_size):
                for j in range(self.nums):
                    [x0, y0, x1, y1] = parts_coordinates[i][j]
                    parts_imgs[i:i + 1, j] = F.interpolate(object_imgs[i:i + 1, :, x0:(x1 + 1), y0:(y1 + 1)], size=(224, 224),
                                                                mode='bilinear',
                                                                align_corners=True) 


            parts_imgs = parts_imgs.reshape(batch_size * self.nums, 3, 224, 224) 
            _, parts_embeddings = self.pretrained_model(parts_imgs.detach())  
            parts_logits = self.rawcls_net(parts_embeddings)  
        else:
            parts_logits = torch.zeros([batch_size * self.nums, self.num_classes]).to(DEVICE)





        return   result_img, raw_logits, object_logits,  parts_logits,crop_logits,crop_box


