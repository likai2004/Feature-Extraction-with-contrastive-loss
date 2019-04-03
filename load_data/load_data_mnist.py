import numpy as np
from torchvision import transforms
from imgloader import Mnist_Data
import os
import gzip
import cPickle as pickle
from urllib import urlretrieve

class MNIST():
    def __init__(self):
        url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
        filename = 'mnist.pkl.gz'
        if not os.path.exists(filename):
            print("Downloading MNIST dataset...")
            urlretrieve(url, filename)
        with gzip.open(filename, 'rb') as f:
            data = pickle.load(f)

        [X_train, y_train], [X_val, y_val], [X_test, y_test] = data[0], data[1], data[2]
        X_train = np.row_stack([X_train,X_val])
        y_train = np.append(y_train,y_val,axis=0)
        X_train, X_test = X_train.reshape((-1, 28, 28)), X_test.reshape((-1, 28, 28))
        y_train, y_test = y_train.astype(np.uint8), y_test.astype(np.uint8)

        self.x_train, self.y_train = X_train*255, y_train
        self.x_test, self.y_test = X_test*255, y_test

    def MNISTToTensor(self,x,y,transform,cfg):
        # transform = transforms.Compose([
        #     transforms.Resize((64,64)),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.1307,), (0.3081,))
        #     ])
        dataset = Mnist_Data(x=x, y=y, transform=transform,cfg=cfg)
        return dataset

