import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
from torch.utils.data import DataLoader
from base_models import FocalLoss, Multiple_GIOU_loss_2
from cub_dataset import CUBDataSet
from tqdm import tqdm
import ziji_transforms
from torchvision import transforms, datasets
from sklearn.metrics import confusion_matrix
from kvasir_model_2 import Guide_CNN
import collections
import numpy as np
import random
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

if __name__ == '__main__':

    epochs = 100
    for alpha in [0.05]:
        # # 保存输出结果到txt中
        # save_dir_txt = './add_pre/IOU_{}_alpha.txt'.format(alpha)
        # f = open(save_dir_txt, 'w', encoding='utf-8')

        def setup_seed(seed):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True
        # 设置随机数种子，40
        setup_seed(100)

        # 1.定义数据预处理格式
        data_transform = {
            "train": ziji_transforms.Compose([ziji_transforms.Resize((400, 400)),
                                              ziji_transforms.ToTensor(),
                                              ziji_transforms.Normalization((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),

            "val": transforms.Compose([transforms.Resize((400, 400)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        }


        # 2.加载训练集
        # 要修改的第一个地方
        # cub_root = 'C:\Users\yingqi\Desktop\毕业论文1\IGCNN\data\CUB200-2011\Image'
        # annotations_root='C:\Users\yingqi\Desktop\毕业论文1\IGCNN\data\CUB200-2011'
        cub_root1 = 'C:/Users/yingqi/Desktop/毕业论文1/IGCNN/data/CUB_200_2011/images'
        annotations_root1 = 'C:/Users/yingqi/Desktop/毕业论文1/IGCNN/data/CUB_200_2011'
        assert os.path.exists(cub_root1), f'{cub_root1} not exists'
        train_dataset = CUBDataSet(cub_root1, annotations_root1 ,split_file='train_test_split.txt',images_file='images.txt',labels_file='image_class_labels.txt',box_file='bounding_boxes.txt',train=True,transform=data_transform["train"])
        train_dataloader = DataLoader(train_dataset,
                                batch_size=8,
                                shuffle=True,
                                num_workers=0,
                                collate_fn=CUBDataSet.collate_fn)

        # 加载验证集
        # 要修改的第二个地方
        # val_dataset = datasets.ImageFolder(root='D:/Code/DATA/kvasir_voc/2_lesion_data/test/',transform=data_transform["val"])
        val_dataset = datasets.ImageFolder(root=os.path.abspath(annotations_root1+'/dataset/val'),transform=data_transform["val"])
        val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=8,
                                                    shuffle=False,
                                                    num_workers=0)
        # print(len(train_dataloader))

        # 3.加载模型
        model = Guide_CNN(grad_layer='layer4', num_classes=200).to(device)

        # 4.定义损失函数
        class_loss = torch.nn.CrossEntropyLoss()
        regression_loss = Multiple_GIOU_loss_2()

        # 5.定义优化方法
        optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0001)
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        logging.basicConfig(filename='logs/training_adam0.1.log', level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

        max_accuracy1 = 0.0
        max_accuracy2 = 0.0
        # 6.开始训练
        for epoch in range(1, epochs+1):
            print('epoch: {}'.format(epoch))
            logging.info('Training started')
            # f.write('epoch: {}'.format(epoch))
            # f.write('\n')
            tra_y = []
            tra_true_y = []
            tra_acc = 0.0
            train_loss = 0.0
            # 开始训练
            model.train()
            for i, (images, zong_targets) in enumerate(train_dataloader):
                images = torch.stack(images, dim=0)
                img_size = []
                imgs_box = []
                img_labels = []
                # print("train_labels:",zong_targets)
                for t in zong_targets:
                    img_size.append(t["img_size"])
                    imgs_box.append(t["boxes"])
                    img_labels.append(t['labels'])

                targets = { #"onehot": torch.stack(onehot, dim=0),
                           "img_size": torch.stack(img_size, dim=0),
                           "img_labels": torch.stack(img_labels, dim=0)
                           }

                images = images.to(device)
                targets = {k: v.to(device) for k, v in targets.items()}
                logits, pre_BOX = model(images, targets["img_labels"], targets["img_size"])

                labels = targets["img_labels"].squeeze(-1)
                loss_1 = class_loss(logits, labels)
                # imgs_box,img_labels:原图像的框和框的标签；pre_BOX, pre_label:预测出来的框以及框对应标签
                loss_2 = regression_loss(imgs_box, pre_BOX, labels)
                # alpha = 0.2
                loss = (1 - alpha)*loss_1 + alpha*loss_2
                # loss = loss_1
                train_loss += loss.item()
                trian_y = torch.max(logits, dim=1)[1]
                # print("train_pre_labels:",trian_y)
                tra_y += trian_y.cpu().numpy().tolist()
                tra_true_y += labels.cpu().numpy().tolist()
                tra_acc = torch.eq(trian_y, labels.to(device)).sum().item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # 输出损失
                # print_freq = 100
                # if (i + 1) % print_freq == 0:
                #     print('iter: {} | loss: {}'.format(i + 1, loss.item()))
                    # f.write('iter: {} | loss: {}'.format(i + 1, loss.item()))
                    # f.write('\n')
            scheduler.step()
            tra_num = len(tra_true_y)
            # print(tra_acc)
            tra_accurate = tra_acc / tra_num
            # tra_accurate_percent = tra_accurate * 100
            print('Train ACC: {} | Train Loss: {}'.format(tra_accurate, train_loss/len(train_dataloader)))
            logging.info(f'Epoch {epoch}, Loss: {train_loss/len(train_dataloader):.4f}, Accuracy: {tra_accurate:.4f}')
            if tra_accurate > max_accuracy1:
                max_accuracy1 = tra_accurate
            # 7.开始验证
            pre_y = []
            ture_y = []
            acc = 0.0  # accumulate accurate number / epoch
            val_loss = 0.0
            model.eval()
            with torch.no_grad():
                for i, (val_images, val_labels) in enumerate(val_dataloader):
                    val_images = val_images.to(device)
                    outputs = model(val_images)
                    val_loss += class_loss(outputs, val_labels.to(device)).item()
                    predict_y = torch.max(outputs, dim=1)[1]
                    # print(predict_y,val_labels)
                    pre_y += predict_y.cpu().numpy().tolist()
                    ture_y += val_labels.cpu().numpy().tolist()
                    acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
            # val_num = 5749
            val_num = len(ture_y)
            val_accurate = acc / val_num
            # val_accurate_percent = val_accurate * 100
            # print(f'Validation Accuracy: {val_accurate_percent:.2f}%')
            print('Val ACC: {} | Val loss: {}'.format(val_accurate, val_loss/len(val_dataloader)))
            logging.info(f'Epoch {epoch}, Val Loss: {val_loss/len(val_dataloader):.4f},Val Accuracy: {val_accurate:.4f}')
            cm = confusion_matrix(ture_y, pre_y)
            if val_accurate > max_accuracy2:
                max_accuracy2 = val_accurate

        print('Max Train ACC: {} | Max Val ACC: {}'.format(max_accuracy1, max_accuracy2))
        logging.info(f'Max Train ACC: {max_accuracy1:.4f} | Max Val ACC: {max_accuracy2:.4f}')

            # 保存模型
            # dirs = './IOU_lesion_save_weights_{}'.format(alpha)
            # if not os.path.exists(dirs):
            #     os.makedirs(dirs)
            # torch.save(model.state_dict(), os.path.join(dirs, 'pre_2cla_epoch_{}.pth').format(epoch))
            # # torch.save(model, os.path.join('./kvasir_save_weights_2', '2cla_epoch_{}.pth').format(epoch))
            # print('save pre_2cla_epoch_{}.pth '.format(epoch))
            # f.write('save pre_2cla_epoch_{}.pth '.format(epoch))
            # f.write('\n')
        # f.close()
