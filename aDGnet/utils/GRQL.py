import torch
from torch import nn
from skimage import measure







class GRQL(nn.Module):
  def __init__(self, nums):
    super(GRQL, self).__init__()
    self.feature_norm = nn.Softmax(dim=2)
    self.bilinear_norm = nn.Softmax(dim=2)
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.dropout = nn.Dropout(0.5)
    self.nums = nums

  def forward(self, x):

    b = x.shape[0]
    c = x.shape[1]
    h = x.shape[2]
    w = x.shape[3]
    f = x.reshape(b, c, -1)       #(B,C,HW)


    f_norm = self.feature_norm(f * 2)      

    #获得通道间关系矩阵
    bilinear = f_norm.bmm(f.transpose(1, 2))       #(B,C,C)
    attn = self.bilinear_norm(bilinear)


    prob = -torch.log(attn)       
    prob = torch.where(torch.isinf(prob), torch.full_like(prob, 0), prob)     
    entropy = torch.sum(torch.mul(attn, prob), dim=2)      
 
    _, index = torch.sort(entropy)
   
    parts_id = index[:, :self.nums]       
    attn = attn[torch.arange(b)[:, None], parts_id, :]  


    
    x = torch.bmm(attn, f)     
    x = x.view(b,self.nums,h,w)   



    #定位
    y = torch.mean(x, dim=[2, 3], keepdim=True)
    M = (x > y).float()    

    coordinates = []
    for i in range(b):     
        xy = []
        for j in M[i]:     
            mask_np = j.cpu().numpy().reshape(14, 14)    
            component_labels = measure.label(mask_np)


            properties = measure.regionprops(component_labels)

            areas = []
            for prop in properties:
                areas.append(prop.area)
            max_idx = areas.index(max(areas))
            intersection = (component_labels == (max_idx + 1)).astype(int)
            prop = measure.regionprops(intersection.astype(int))

            bbox = prop[0].bbox

            x_lefttop = bbox[0] * 32 - 1
            y_lefttop = bbox[1] * 32 - 1
            x_rightlow = bbox[2] * 32 - 1
            y_rightlow = bbox[3] * 32 - 1
            # for image
            if x_lefttop < 0:
                x_lefttop = 0
            if y_lefttop < 0:
                y_lefttop = 0
            coordinate = [x_lefttop, y_lefttop, x_rightlow, y_rightlow]

            xy.append(coordinate)       

        coordinates.append(xy)         
    return coordinates