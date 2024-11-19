import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Multiple_GIOU_loss(nn.Module):

    def __init__(self):
        super(Multiple_GIOU_loss, self).__init__()

    def giou(self, pred, target, eps=1e-6):
        """
        preds:[x1,y1,x2,y2]
        bbox:[x1,y1,x2,y2]
        return: giou
        """
        # 求pred, target面积
        pred_widths = (pred[2] - pred[0] + 1.).clamp(0)
        pred_heights = (pred[3] - pred[1] + 1.).clamp(0)
        target_widths = (target[2] - target[0] + 1.).clamp(0)
        target_heights = (target[3] - target[1] + 1.).clamp(0)
        pred_areas = pred_widths * pred_heights
        target_areas = target_widths * target_heights

        # 求pred, target相交面积
        inter_xmins = torch.maximum(pred[0], target[0])
        inter_ymins = torch.maximum(pred[1], target[1])
        inter_xmaxs = torch.minimum(pred[2], target[2])
        inter_ymaxs = torch.minimum(pred[3], target[3])
        inter_widths = torch.clamp(inter_xmaxs - inter_xmins + 1.0, min=0.)
        inter_heights = torch.clamp(inter_ymaxs - inter_ymins + 1.0, min=0.)
        inter_areas = inter_widths * inter_heights

        # 求iou
        unions = pred_areas + target_areas - inter_areas
        # ious = torch.clamp(inter_areas / unions, min=eps)
        # 将iou处的分母从并变为pre的框
        ious = torch.clamp(inter_areas / pred_areas, min=eps)

        # 求最小外接矩形
        outer_xmins = torch.minimum(pred[0], target[0])
        outer_ymins = torch.minimum(pred[1], target[1])
        outer_xmaxs = torch.maximum(pred[2], target[2])
        outer_ymaxs = torch.maximum(pred[3], target[3])
        outer_widths = (outer_xmaxs - outer_xmins + 1).clamp(0.)
        outer_heights = (outer_ymaxs - outer_ymins + 1).clamp(0.)
        outer_areas = outer_heights * outer_widths

        gious = ious - (outer_areas - unions) / outer_areas
        gious = gious.clamp(min=-1.0, max=1.0)
        # if reduction == 'mean':
        #     loss = torch.mean(1 - gious)
        # elif reduction == 'sum':
        #     loss = torch.sum(1 - gious)
        # else:
        #     raise NotImplementedError
        return gious


    def forward(self, imgs_box, img_labels, pre_BOX, pre_label):

        batch_loss = 0
        for i in range(len(img_labels)):
            img_lab = img_labels[i].to(device)
            img_box = imgs_box[i].to(device)

            pre_box = pre_BOX[i]
            pre_lab = pre_label[i][0].to(device)   # pre_label[i]是tuple类型，取出元素是tensor类型

            for c in pre_lab:
                # 在pre_label中定位c，这是因为pre_label中的c不重复
                c_pre_index = (pre_lab == c).nonzero()
                # 取出c类的预测框,是一个tensor类型
                c_pre_box = pre_box[c_pre_index].to(device)

                # 在img_lab中定位c
                c_lab_index = (img_lab == c).nonzero()
                # 取出img_box中对应c类别的框，是一个列表，列表中每个都是tensor类型
                c_lab_box = []
                for idex in c_lab_index:
                    c_box = img_box[idex, :]
                    c_lab_box.append(c_box)

                # 得到c类的c_pre_box与c_lab_box后，求该类的GIOU损失
                for i in range(c_pre_box.shape[0]):
                    pr_box = c_pre_box[i, :]
                    max_giou = 0
                    loss = 0
                    for gt_box in c_lab_box:
                        g_iou = self.giou(pr_box, gt_box[0], eps=1e-6)
                        if g_iou > max_giou:
                            loss = 1 - g_iou

                    batch_loss = batch_loss + loss


                # # 得到c类的c_pre_box与c_lab_box后，求该类的GIOU损失
                # c_loss =
                # for gt_box in c_lab_box:
                #     for i in range(c_pre_box.shape[0]):
                #         pr_box = c_pre_box[i, :]
                #         g_iou = self.giou(pr_box, gt_box[0], eps=1e-6)
                #         loss = 1 - g_iou
                #         batch_loss = batch_loss + loss

        LOSS = batch_loss/len(img_labels)

        return LOSS