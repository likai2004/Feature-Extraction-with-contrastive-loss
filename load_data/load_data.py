import numpy as np
from torchvision import transforms
from imgloader import UKB_Data
import os
import gzip
import cPickle as pickle
from urllib import urlretrieve

class loadUKB():
    def __init__(self,data_root):
        self.x_train = []
        self.y_train = []
        self.data_root = data_root

        each_num = 4
        for root, dirs,files in os.walk(data_root):
            for file in files:
                num_i = file.replace('ukbench','')
                num_i = num_i.replace('.jpg','')
                num_i = int(num_i)//4
                if os.path.splitext(file)[1] == '.jpg':
                    self.x_train.append(os.path.join(root, file))
                    self.y_train.append(num_i)
        self.y_train = np.array(self.y_train)
        self.x_train = np.array(self.x_train)

    def UKBToTensor(self,x,y,transform,cfg):
        # transform = transforms.Compose([
        #     transforms.Resize((64,64)),
        #     transforms.ToTensor(),s
        #     transforms.Normalize((0.1307,), (0.3081,))
        #     ])
        dataset = UKB_Data(x=x, y=y, transform=transform,cfg=cfg)
        return dataset

    def UKB_imgs_list(self):
        imgs_list = []
        num_zero = 5
        total_num = 10200
        for i in range(10200):
            num_i = str(i)
            num_i = num_i.zfill(num_zero)
            img_name = 'ukbench{0}.jpg'.format(num_i)
            imgs_list.append(os.path.join(self.data_root, img_name))
        return imgs_list


