import torch.utils.data as data
from PIL import Image
import torch
import numpy as np
import random
class Mnist_Data(data.Dataset):
    def __init__(self,x,y,transform, cfg):
        self.data = x
        self.label = y
        self.transform = transform
        self.n_data = y.shape[0]
        if not self._check_exists():
            raise RuntimeError('Dataset not found')
        # self.class_list = range(cfg.CLASSES)
        self.split_data = []
        self.split_label = []
        self.split_num = []
        self.classes = cfg.CLASSES
        self.split_data_by_class()

        self.query_classes = cfg.QUERY_CLASSES
        self.pos_num = cfg.POS_NUM

    def split_data_by_class(self):
        for i in range(self.classes):
            tmp_idx = (self.label == i)
            tmp_label = self.label[tmp_idx]
            tmp_data = self.data[tmp_idx]
            self.split_data.append(tmp_data)
            self.split_label.append(tmp_label)
            self.split_num.append(tmp_label.shape[0])

    def __getitem__(self, index):
        img_list = []
        label_list = []
        img_1, label_1 = self.data[index], int(self.label[index])
        img_list.append(img_1)
        label_list.append(label_1)

        #rand_num = torch.from_numpy(np.random.permutation(neg_inds.size(0))).long()
        label_1_idx = np.random.permutation(self.split_num[label_1])
        label_1_idx = label_1_idx[:self.pos_num-1]
        img_list.extend(self.split_data[label_1][label_1_idx])
        label_list.extend(self.split_label[label_1][label_1_idx])

        all_label = range(self.classes)
        all_label.remove(label_1)
        rest_label = np.array(all_label)
        np.random.shuffle(rest_label)

        rest_label = rest_label[:self.query_classes-1]
        for i_rest_label in rest_label:
            i_label_idx = np.random.permutation(self.split_num[i_rest_label])
            i_label_idx = i_label_idx[:self.pos_num]
            img_list.extend(self.split_data[i_rest_label][i_label_idx])
            label_list.extend(self.split_label[i_rest_label][i_label_idx])

        group_imgs = []
        group_labels = []
        for i_tmp, tmp_img_name in enumerate(img_list):
            tmp_img = Image.open(tmp_img_name).convert('RGB')
            # tmp_img = Image.fromarray(tmp_img.astype(np.uint8))
            if self.transform is not None:
                tmp_img = self.transform(tmp_img)
            tmp_img = tmp_img.expand(3, tmp_img.size(1), tmp_img.size(2)).unsqueeze(0)
            group_imgs.append(tmp_img)
            group_labels.append(float(label_list[i_tmp]))

        return torch.cat(group_imgs,0), torch.FloatTensor(group_labels)

    def __len__(self):
        return self.n_data

    def _check_exists(self):
        return self.n_data > 0