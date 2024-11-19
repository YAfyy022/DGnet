from torch.utils.data import Dataset
import os
import torch
from PIL import Image

class CUBDataSet(Dataset):
    """读取解析CUB200-2011数据集"""

    def __init__(self, cub_root, annotations_root, split_file, images_file, labels_file, box_file, train=True, transform=None):
        self.cub_root = cub_root
        self.annotations_root = annotations_root
        self.transform = transform
        self.image_ids = []
        self.image_paths = {}
        self.labels = {}
        self.boxes = {}

        # 读取split文件，获取训练集和测试集的image-id
        with open(os.path.join(annotations_root, split_file), 'r') as f:
            for line in f:
                image_id, is_train = line.strip().split()
                if (is_train == '1' and train) or (is_train == '0' and not train):
                    self.image_ids.append(image_id)

        # 读取images文件，获取image-id对应的图像路径
        with open(os.path.join(annotations_root, images_file), 'r') as f:
            for line in f:
                image_id, image_path = line.strip().split()
                if image_id in self.image_ids:
                    self.image_paths[image_id] = os.path.join(self.cub_root, image_path)

        # 读取labels文件，获取image-id对应的标签
        with open(os.path.join(annotations_root, labels_file), 'r') as f:
            for line in f:
                image_id, label = line.strip().split()
                if image_id in self.image_ids:
                    self.labels[image_id] = int(label)

        # 读取box文件，获取image-id对应的边界框信息
        with open(os.path.join(annotations_root, box_file), 'r') as f:
            for line in f:
                image_id, x, y, w, h = line.strip().split()
                if image_id in self.image_ids:
                    self.boxes[image_id] = [float(x), float(y), float(w), float(h)]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = self.image_paths[image_id]
        label = self.labels[image_id]
        image = Image.open(image_path).convert('RGB')
        width, height = image.size

        # if image.format != "JPEG":
        #     raise ValueError(f"Image '{image_path}' format not JPEG")

        box = self.boxes.get(image_id, None)
        if box:
            x, y, w, h = map(float, box)
            boxes = [[max(0, x), max(0, y), min(x + w, width), min(y + h, height)]]  # 标准边界框格式[x_min, y_min, x_max, y_max]
        else:
            print(f"Warning: No bounding box information for image {image_id}. Skipping...")
            return None

        # convert everything into a torch.Tensor
        img_size = torch.as_tensor([image.size], dtype=torch.int64)
        img_boxes = torch.as_tensor(boxes, dtype=torch.float32)
        img_label = torch.as_tensor(label, dtype=torch.int64)
        image_id = torch.tensor(int(image_id), dtype=torch.int64)
        area = (img_boxes[:, 3] - img_boxes[:, 1]) * (img_boxes[:, 2] - img_boxes[:, 0])

        # 将labels转化为onehot编码
        if 1 <= img_label <= 200:
            img_label_idx = img_label - 1  # 将标签从1-200转换为0-199
            onehot = torch.zeros(200).float()
            onehot[img_label_idx] = 1
            onehot = onehot.float()
        else:
            raise ValueError(f"Label index {img_label} is out of bounds. It should be between 1 and 200.")

        target = {
            "img_size": img_size,
            "boxes": img_boxes,
            "labels": img_label_idx,
            "image_id": image_id,
            "area": area,
            "onehot": onehot
        }

        if self.transform:
            image, target = self.transform(image, target)

        return image, target

    @staticmethod
    def collate_fn(batch):
        batch = [item for item in batch if item is not None]

        # 如果过滤后没有数据，返回空的批次
        if not batch:
            return [], {}
        images, targets = tuple(zip(*batch))
        return images, targets
# 使用示例
# if __name__ == "__main__":
#     dataset = CUBDataSet(
#         cub_root='C:/Users/yingqi/Desktop/毕业论文1/IGCNN/data/CUB_200_2011/images',
#         annotations_root='C:/Users/yingqi/Desktop/毕业论文1/IGCNN/data/CUB_200_2011',
#         split_file='train_test_split.txt',
#         images_file='images.txt',
#         labels_file='image_class_labels.txt',
#         box_file='bounding_boxes.txt',
#         train=True,
#         transform=None  # 如果有图像转换操作，则传入相应的transform
#     )
#
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=CUBDataSet.collate_fn)
#     for images, targets in dataloader:
#         # 处理图像和目标
#         pass