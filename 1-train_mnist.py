import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision.transforms as transforms
from tensorboardX import  SummaryWriter

from config import para_cfg
from load_data.load_data import MNIST
from load_data.imgloader import Mnist_Data
from model.AlexNet import AlexNet
from model.utils.loss import ContrastiveLoss

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

if cfg.USE_TFBOARD:
    writer = SummaryWriter(comment=cfg.LOG_NAME)

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
def train(epoch):
    net.train()
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE,
                                              shuffle=True,num_workers=cfg.NUM_WORKERS)
    data = iter(data_loader)
    len_iter = len(data_loader)
    # print len_iter

    len_iter_path = 0
    total_loss = 0.
    for batch_idx in range(len_iter):
        len_iter_path = batch_idx
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
        optimizer.zero_grad()
        outputs = net(inputs) #(bs*nq_si)*Dim

        loss = criterion(outputs.view(bs,nq_si,-1),target_label,reduce=True)
        loss = loss.mean()
        loss.backward()
        optimizer.step()

        total_loss += loss.data[0]
        loss_iter = epoch * len_iter + batch_idx
        if (loss_iter % 300) == 0:
            print ('epoch: % 3d | loss: %.4f' % (epoch, loss.data[0]))
            if cfg.USE_TFBOARD:
                writer.add_scalar('loss_iter', loss.data[0], loss_iter)

    if (epoch % 1) == 0:
        print ('epoch: % 3d | train_loss: %.4f'%(epoch,total_loss/(len_iter)))
        if cfg.USE_TFBOARD:
            writer.add_scalar('train_loss',total_loss/(len_iter), epoch*len_iter+batch_idx)

        save_model(epoch, len_iter_path)

############# save net ################
def save_model(epoch,len_iter):
    print ('Saving...')
    if cfg.mGPUs:
        state = {
            'net':net.module.state_dict(),
            'optimizer':optimizer.state_dict(),
            'lr':cfg.LR,
            'epoch':epoch,
            'op':cfg.OPTIM
        }
    else:
        state = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr': cfg.LR,
            'epoch': epoch,
            'op': cfg.OPTIM
        }
    if not os.path.isdir(cfg.CHECKPOINT_PATH):
        os.mkdir(cfg.CHECKPOINT_PATH)
    torch.save(state, cfg.CHECKPOINT_PATH+'/ckpt-'+str(epoch)+'-'+str(len_iter)+'.pth')

for epoch in range(cfg.START_EPOCH, cfg.MAX_EPOCH):
    train(epoch)