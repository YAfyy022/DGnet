import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Multiple_GIOU_loss_2(nn.Module):

    def __init__(self):
        super(Multiple_GIOU_loss_2, self).__init__()

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
        ious = torch.clamp(inter_areas / unions, min=eps)
        # # 将iou处的分母从并变为pre的框
        # ious = torch.clamp(inter_areas / pred_areas, min=eps)

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
        return gious


    def forward(self, imgs_box, pre_BOX, labels):

        batch_loss = 0
        batch_size = labels.shape[0]
        for i in range(batch_size):
            img_box = imgs_box[i].to(device)
            pre_box = pre_BOX[i].to(device)
            for pr_box in pre_box:
                # pr_box = pr_box# .to(device)
                max_giou = -2   # giou的值区间是[-1，1]
                for gt_box in img_box:
                    g_iou = self.giou(pr_box, gt_box, eps=1e-6)
                    if g_iou >= max_giou:
                        max_giou = g_iou
                batch_loss += 1 - max_giou
        LOSS = batch_loss/batch_size
        return LOSS