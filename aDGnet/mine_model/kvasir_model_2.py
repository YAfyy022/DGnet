import os
from base_models import FcaNet
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms.functional as TF
from base_models import FocalLoss, Multiple_GIOU_loss_2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 加边框
class Guide_CNN(nn.Module):
    def __init__(self, grad_layer, num_classes):
        super(Guide_CNN, self).__init__()

        # 修改模型时注意修改全连接层的接口
        self.model = models.resnet50(pretrained=True)
        # self.model=FcaNet.fcanet34(num_classes=1_000,pretrained=True)
        self.model.fc = nn.Linear(2048, num_classes)
        # self.model = models.resnet50(pretrained=True)
        # self.model.fc = nn.Linear(2048, num_classes)

        # print(self.model)
        self.grad_layer = grad_layer
        self.num_classes = num_classes
        # Feed-forward features
        self.feed_forward_features = None
        # Backward features
        self.backward_features = None
        # Register hooks
        self._register_hooks(grad_layer)

    def _register_hooks(self, grad_layer):
        def forward_hook(module, grad_input, grad_output):
            self.feed_forward_features = grad_output

        def backward_hook(module, grad_input, grad_output):
            self.backward_features = grad_output[0]

        gradient_layer_found = False
        for idx, m in self.model.named_modules():
            if idx == grad_layer:
                m.register_forward_hook(forward_hook)
                m.register_backward_hook(backward_hook)
                print("Register forward hook !")
                print("Register backward hook !")
                gradient_layer_found = True
                break

        # for our own sanity, confirm its existence
        if not gradient_layer_found:
            raise AttributeError('Gradient layer %s not found in the internal model' % grad_layer)

    def _to_ohe(self, labels):
        ohe = torch.zeros((labels.size(0), self.num_classes), requires_grad=True)
        for i, label in enumerate(labels):
            label_value = label.item() if isinstance(label, torch.Tensor) else label
            ohe.data[i, label_value] = 1

        ohe = torch.autograd.Variable(ohe)

        return ohe



    def get_obj_box(self, scaled_ac, img_size, threshold=0.8):
        # 由重要性图生成边界框, 重要性从，图是标准化到[0, 1]的
        # 阈值的选取很重要，这里先选取0.7
        # mask转化为np，找目标框
        mask = torch.where(scaled_ac > threshold, torch.ones_like(scaled_ac), scaled_ac)
        mask = torch.where(mask < threshold, torch.zeros_like(scaled_ac), mask)
        # RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
        score_map = mask[0][0].cpu().data.numpy()
        score_map = score_map.astype(np.uint8)
        contours, _ = cv2.findContours(score_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        max_contour = None
        max_area = 0

        for con in contours:
            area = cv2.contourArea(con)
            if area > max_area:
                max_area = area
                max_contour = con

        if max_contour is not None:
            (x, y, w, h) = cv2.boundingRect(max_contour)
            x0=min((x + w),img_size[0,0])
            y0=min((y + h),img_size[0,1])
            boxes.append([x, y, x0, y0])
        boxes = torch.as_tensor(boxes, dtype=torch.float32).to(device)
        # contours, _ = cv2.findContours(score_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # boxes = []
        # for con in contours:
        #     # 最大的面积为原图像的h*w，注意用的时候传入原图像的h*w
        #     max_area = img_size[0][0] * img_size[0][1]
        #     # 这里的阈值也重要，需要修改
        #     if max_area * 0.6 > cv2.contourArea(con):  #> max_area * 0.0005:
        #         (x, y, w, h) = cv2.boundingRect(con)
        #         boxes.append([x, y,  (x+w),  (y+h)])
        # boxes = torch.as_tensor(boxes, dtype=torch.float32).to(device)
        # 为了将梯度链接到box
        # 将mask下采样到boxes的大小，再将boxes的值赋给mask.data
        # (h, w) = boxes.shape
        # mask_boxes = F.interpolate(mask, size=(h, w), mode='bilinear')[0][0]
        # mask_boxes.data = boxes
        # return mask_boxes
        return boxes

    # def map_box_to_new_size(box, original_size, new_size):
    #     # 归一化坐标
    #     x_min = box[0] / original_size[0]
    #     y_min = box[1] / original_size[1]
    #     x_max = box[2] / original_size[0]
    #     y_max = box[3] / original_size[1]
    #
    #     # 映射到新尺寸
    #     new_x_min = int(x_min * new_size[0])
    #     new_y_min = int(y_min * new_size[1])
    #     new_x_max = int(x_max * new_size[0])
    #     new_y_max = int(y_max * new_size[1])
    #
    #     # 确保新的坐标不超出新图像的尺寸
    #     new_x_min = max(0, min(new_x_min, new_size[0]))
    #     new_y_min = max(0, min(new_y_min, new_size[1]))
    #     new_x_max = max(0, min(new_x_max, new_size[0]))
    #     new_y_max = max(0, min(new_y_max, new_size[1]))
    #
    #     # 确保坐标顺序正确
    #     if new_x_min > new_x_max:
    #         new_x_min, new_x_max = new_x_max, new_x_min
    #     if new_y_min > new_y_max:
    #         new_y_min, new_y_max = new_y_max, new_y_min
    #
    #     return [new_x_min, new_y_min, new_x_max, new_y_max]
    #
    # def crop_image_by_box(self, image, boxes, original_size, new_size=(400, 400)):
    #     # 将原始图像的边界框坐标映射到新图像尺寸
    #     mapped_boxes = [map_box_to_new_size(box, original_size, new_size) for box in boxes]
    #
    #     # 截取对应的图像区域
    #     cropped_images = []
    #     for box in mapped_boxes:
    #         x_min, y_min, x_max, y_max = box
    #         cropped_image = image.crop((x_min, y_min, x_max, y_max))
    #         cropped_images.append(cropped_image)
    #
    #     return cropped_images

    def forward(self, images, labels=0, images_size=(0, 0)):
        '''
        '''
        is_train = self.model.training

        if not is_train:
            #with torch.no_grad():
            logits = self.model(images)
            return logits
        else:
            with torch.enable_grad():
                # labels_ohe = self._to_ohe(labels).cuda()
                # labels_ohe.requires_grad = True
                # _, _, img_h, img_w = images.size()

                self.model.train(True)
                logits = self.model(images)  # BS x num_classes
                self.model.zero_grad()

                labels_ohe = self._to_ohe(labels).cuda()
                # class_loss = torch.nn.CrossEntropyLoss()
                # regression_loss = Multiple_GIOU_loss_2()

                gradient = logits * labels_ohe
                grad_logits = (logits * labels_ohe).sum()  # BS x num_classes
                # #grad_logits.backward(gradient=gradient, retain_graph=True)
                grad_logits.backward(retain_graph=True)
                self.model.zero_grad()

            if is_train:
                self.model.train(True)
            else:
                self.model.train(False)
                self.model.eval()
                logits = self.model(images)

            backward_features = self.backward_features  # BS x C x H x W
            # bs, c, h, w = backward_features.size()
            # wc = F.avg_pool2d(backward_features, (h, w), 1)  # BS x C x 1 x 1
            """
            The wc shows how important of the features map
            """
            # Eq 2
            fl = self.feed_forward_features  # BS x C x H x W
            weights = F.adaptive_avg_pool2d(backward_features, 1)
            Ac = torch.mul(fl, weights).sum(dim=1, keepdim=True)
            Ac = F.relu(Ac)
            # Ac = F.interpolate(Ac, size=images.size()[2:], mode='bilinear', align_corners=False)
            BOX = []
            for i in range(labels.shape[0]):
                # ac = Ac[i].unsqueeze(0)
                # size = (images_size[i, 0], images_size[i, 1])
                width = images_size[i, 0, 0].item()  # 获取高度
                height = images_size[i, 0, 1].item()
                score_map = F.upsample_bilinear(Ac[i].unsqueeze(0), size=(width, height))
                score_map_min = score_map.min()
                score_map_max = score_map.max()
                score_map = (score_map - score_map_min) / (score_map_max - score_map_min)
                box = self.get_obj_box(score_map, images_size[i], threshold=0.8)
                BOX.append(box)
            return logits, BOX




# model = Guide_CNN(num_classes=4)
# print(model)