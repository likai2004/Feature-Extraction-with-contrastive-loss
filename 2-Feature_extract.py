import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision.transforms as transforms
from tensorboardX import  SummaryWriter
import numpy as np
from config import para_cfg
from load_data.load_data import loadUKB
from load_data.imgloader import UKB_Data
from model.AlexNet import AlexNet
from model.utils.loss import ContrastiveLoss
import cPickle as pickle
import os
import Image
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

################## 0-Parameters Setting ##############
cfg = para_cfg()
cfg.print_para()
using_gpu = torch.cuda.is_available() & cfg.CUDA

################## 1-Load Data ##############
ukb_dat = loadUKB(cfg.DATA_ROOT)
x_train, y_train = ukb_dat.x_train, ukb_dat.y_train
# train_dataset = ukb_dat.UKBToTensor(x_train, y_train, cfg.TRAIN_transforms,cfg)
test_imgs = ukb_dat.UKB_imgs_list()
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
    test_transform = cfg.TRAIN_transforms
    res_name = './res/num-{0}.pkl'.format(1)
    fr = open(res_name, 'w')
    for img_name in test_imgs:
        img_tmp = Image.open(img_name).convert('RGB')
        tmp_img = test_transform(img_tmp)
        inputs = tmp_img.expand(3, tmp_img.size(1), tmp_img.size(2)).unsqueeze(0)

        if using_gpu:
            inputs = Variable(inputs.cuda(),volatile=True)
        else:
            inputs = Variable(inputs,volatile=True)
        outputs = net(inputs) #(bs*nFalseq_si)*Dim
        # print outputs.size()
        tmp = np.array(outputs.cpu().data)
        # print tmpFalseFalse
        fr.write(tmp)
    fr.close()

test()