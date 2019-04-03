import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision.transforms as transforms
from tensorboardX import  SummaryWriter
import numpy as np
from config import para_cfg
from load_data.load_data import MNIST
from load_data.imgloader import Mnist_Data
from model.AlexNet import AlexNet
from model.utils.loss import ContrastiveLoss
import cPickle as pickle
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

################## 0-Parameters Setting ##############
cfg = para_cfg()
cfg.print_para()
using_gpu = torch.cuda.is_available() & cfg.CUDA

################## 1-Load Data ##############
mnist_dat = MNIST()
x_train, y_train = mnist_dat.x_train, mnist_dat.y_train
x_test, y_test = mnist_dat.x_test, mnist_dat.y_test
train_dataset = mnist_dat.MNISTToTensor(x_train, y_train, cfg.TRAIN_transforms,cfg)
test_dataset = mnist_dat.MNISTToTensor(x_test, y_test, cfg.TRAIN_transforms,cfg)

################## 2-Load Net and Loss ##############
net = AlexNet()
criterion = ContrastiveLoss(margin=cfg.LOSS_MARGIN,pos_num=cfg.POS_NUM,
                            neg_num_for_loss=cfg.NEG_NUM_FOR_LOSS)

if cfg.OPTIM == 'Adam':
    optimizer = optim.Adam(net.parameters(),lr=cfg.LR,
                           weight_decay=cfg.WEIGHT_DECAY)


########### resume model ############
if cfg.RESUME:
    print('==========>Resuming from checkpoint...........')
    load_name = cfg.RESUME_PATH.split('/')[-1]
    print("loading checkpoint %s" % (load_name), )
    checkpoint = torch.load(cfg.RESUME_PATH)
    optimizer.load_state_dict(checkpoint['optimizer'])
    net.load_state_dict(checkpoint['net'])
    cfg.LR = optimizer.param_groups[0]['lr']
    # best_loss = checkpoint['loss']
    cfg.START_EPOCH = checkpoint['epoch'] + 1
    print("loading checkpoint %s" % (load_name), )

######### DataParallel and cuda ########
if (torch.cuda.device_count() > 1) & (cfg.mGPUs):
    net = torch.nn.DataParallel(net)
if using_gpu:
    net.cuda()
    criterion.cuda()

################## 3-Training Phase ##############
def test():
    net.eval()
    data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE,
                                              shuffle=True,num_workers=cfg.NUM_WORKERS)
    data = iter(data_loader)
    len_iter = len(data_loader)
    # print len_iter

    for batch_idx in range(1):
    # for batch_idx in range(len_iter):
        inputs, target_label = data.next()
        bs,nq_si,chn,inp_h,inp_w = inputs.size() # bs*(classes*pos_num)*chn*h*w
        inputs = inputs.view(-1,chn,inp_h,inp_w)
        # target_label = target_label.view(-1)

        if using_gpu:
            inputs = Variable(inputs.cuda())
            target_label = Variable(target_label.cuda())
        else:
            inputs = Variable(inputs)
            target_label = Variable(target_label)
        print list(target_label.cpu().data.view(-1))
        outputs = net(inputs) #(bs*nq_si)*Dim
        print outputs.size()
        res_label = list(target_label.cpu().data.int().view(-1))
        res_name = './res/num-{0}.pkl'.format(0)
        fr = open(res_name, 'w')
        for i_la, la in enumerate(res_label):
            tmp = np.array(outputs[i_la].cpu().data)
            print tmp
            fr.write(tmp)
            # pickle.dump(tmp,fr, pickle.HIGHEST_PROTOCOL)
        fr.close()
            # fid = open(res_name)
            # inf = pickle.load(fid)
            # fid.close()
            # print inf

test()