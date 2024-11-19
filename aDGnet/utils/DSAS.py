import torch
from skimage import measure
from torch import nn





class MinPool(nn.Module):
    def __init__(self, kernel_size, ndim=2, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super(MinPool, self).__init__()
        self.pool = getattr(nn, f'MaxPool{ndim}d')(kernel_size=kernel_size, stride=stride, padding=padding,
                                                   dilation=dilation,
                                                   return_indices=return_indices, ceil_mode=ceil_mode)

    def forward(self, x):
        x = self.pool(-x)
        return -x


@torch.no_grad()
class ApprConSolution(nn.Module):
    # Approximate connected solution
    def __init__(self, scale=2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale)
        self.thre = 0.3  # [0 - 0.2 ,0.2 - 0.3 , 0.3 - 1] small middle big
        self.magic_number = 169  # 13 * 13
        self.few_connect_col = nn.MaxPool2d(kernel_size=(2, 1), stride=1, padding=0)
        self.few_ablate_col = MinPool(kernel_size=(2, 1), stride=1, padding=0)
        self.few_connect_raw = nn.MaxPool2d(kernel_size=(1, 2), stride=1, padding=0)
        self.few_ablate_raw = MinPool(kernel_size=(1, 2), stride=1, padding=0)
        self.dense_connect = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dense_ablate = MinPool(kernel_size=2, stride=1, padding=0)

    def forward_dense(self, x):
        # print("call forward dense")
        # pad first

        x_pad = torch.nn.functional.pad(x, (0, 1, 0, 1))
       
        ans = self.dense_ablate(x_pad)
        # up scale
        ans = self.up(ans)

        # enchance the max
        ans = self.dense_connect(ans)
        ans = self.dense_connect(ans)
        ans = self.up(ans)
        return ans

    def forward_few(self, x):
        # print("call forward few")
        # pad in different direction

        a_pad = torch.nn.functional.pad(x, (0, 0, 1, 1))
        ans = self.few_ablate_col(a_pad)
        ans_col = self.few_connect_col(ans)

        a_pad = torch.nn.functional.pad(x, (1, 1, 0, 0))
        ans = self.few_ablate_raw(a_pad)
        ans_raw = self.few_connect_raw(ans)
        return ans_col + ans_raw

    def forward(self, x):
        # B,C,W,H = x.shape
        density_rate_list = torch.sum(x, dim=[2, 3]) / self.magic_number  # better more then 14 * 14

        out = []
        for index, density_rate in enumerate(density_rate_list):
            activate_mask = self.forward_few(x[index].unsqueeze(0))


            if torch.sum(activate_mask) == 0:
                activate_mask = x[index].unsqueeze(0)

            out.append(activate_mask)
        out = torch.cat(out)
        return out


acs_filter = ApprConSolution()




def DSAS(fms):

    # maxp = nn.MaxPool2d(2, stride=1, padding=0).cuda()


    A = torch.sum(fms, dim=1, keepdim=True)    #b,1,h,w
    a = torch.mean(A, dim=[2, 3], keepdim=True)
    M = (A > a).float()    #b,1,h,w
    M = acs_filter(M)



    coordinates = []

    for i, m in enumerate(M):
        mask_np = m.cpu().numpy().reshape(14, 14)
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
        coordinates.append(coordinate)
    return coordinates






