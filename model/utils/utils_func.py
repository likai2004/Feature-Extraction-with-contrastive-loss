import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import combinations
# --------------------------------------
# pooling
# --------------------------------------

def mac(x):
    return F.max_pool2d(x, (x.size(-2), x.size(-1)))
    # return F.adaptive_max_pool2d(x, (1,1)) # alternative


def spoc(x):
    return F.avg_pool2d(x, (x.size(-2), x.size(-1)))
    # return F.adaptive_avg_pool2d(x, (1,1)) # alternative


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
    # return F.lp_pool2d(F.threshold(x, eps, eps), p, (x.size(-2), x.size(-1))) # alternative


def rmac(x, L=3, eps=1e-6):
    ovr = 0.4 # desired overlap of neighboring regions
    steps = torch.Tensor([2, 3, 4, 5, 6, 7]) # possible regions for the long dimension

    W = x.size(3)
    H = x.size(2)

    w = min(W, H)
    w2 = math.floor(w/2.0 - 1)

    b = (max(H, W)-w)/(steps-1)
    (tmp, idx) = torch.min(torch.abs(((w**2 - w*b)/w**2)-ovr), 0) # steps(idx) regions for long dimension

    # region overplus per dimension
    Wd = 0;
    Hd = 0;
    if H < W:  
        Wd = idx.item() + 1
    elif H > W:
        Hd = idx.item() + 1

    v = F.max_pool2d(x, (x.size(-2), x.size(-1)))
    v = v / (torch.norm(v, p=2, dim=1, keepdim=True) + eps).expand_as(v)

    for l in range(1, L+1):
        wl = math.floor(2*w/(l+1))
        wl2 = math.floor(wl/2 - 1)

        if l+Wd == 1:
            b = 0
        else:
            b = (W-wl)/(l+Wd-1)
        cenW = torch.floor(wl2 + torch.Tensor(range(l-1+Wd+1))*b) - wl2 # center coordinates
        if l+Hd == 1:
            b = 0
        else:
            b = (H-wl)/(l+Hd-1)
        cenH = torch.floor(wl2 + torch.Tensor(range(l-1+Hd+1))*b) - wl2 # center coordinates
            
        for i_ in cenH.tolist():
            for j_ in cenW.tolist():
                if wl == 0:
                    continue
                R = x[:,:,(int(i_)+torch.Tensor(range(wl)).long()).tolist(),:]
                R = R[:,:,:,(int(j_)+torch.Tensor(range(wl)).long()).tolist()]
                vt = F.max_pool2d(R, (R.size(-2), R.size(-1)))
                vt = vt / (torch.norm(vt, p=2, dim=1, keepdim=True) + eps).expand_as(vt)
                v += vt

    return v


def roipool(x, rpool, L=3, eps=1e-6):
    ovr = 0.4 # desired overlap of neighboring regions
    steps = torch.Tensor([2, 3, 4, 5, 6, 7]) # possible regions for the long dimension

    W = x.size(3)
    H = x.size(2)

    w = min(W, H)
    w2 = math.floor(w/2.0 - 1)

    b = (max(H, W)-w)/(steps-1)
    _, idx = torch.min(torch.abs(((w**2 - w*b)/w**2)-ovr), 0) # steps(idx) regions for long dimension

    # region overplus per dimension
    Wd = 0;
    Hd = 0;
    if H < W:  
        Wd = idx.item() + 1
    elif H > W:
        Hd = idx.item() + 1

    vecs = []
    vecs.append(rpool(x).unsqueeze(1))

    for l in range(1, L+1):
        wl = math.floor(2*w/(l+1))
        wl2 = math.floor(wl/2 - 1)

        if l+Wd == 1:
            b = 0
        else:
            b = (W-wl)/(l+Wd-1)
        cenW = torch.floor(wl2 + torch.Tensor(range(l-1+Wd+1))*b).int() - wl2 # center coordinates
        if l+Hd == 1:
            b = 0
        else:
            b = (H-wl)/(l+Hd-1)
        cenH = torch.floor(wl2 + torch.Tensor(range(l-1+Hd+1))*b).int() - wl2 # center coordinates
            
        for i_ in cenH.tolist():
            for j_ in cenW.tolist():
                if wl == 0:
                    continue
                vecs.append(rpool(x.narrow(2,i_,wl).narrow(3,j_,wl)).unsqueeze(1))

    return torch.cat(vecs, dim=1)


# --------------------------------------
# normalization
# --------------------------------------

def l2n(x, eps=1e-6):
    return x / (torch.norm(x, p=2, dim=1, keepdim=True) + eps).expand_as(x)

def powerlaw(x, eps=1e-6):
    x = x + eps
    return x.abs().sqrt().mul(x.sign())

# --------------------------------------
# loss
# --------------------------------------

# def contrastive_loss(x, x2, label, margin=0.7, eps=1e-6):
#     # x is D x N
#     dim = x.size(0) # D
#     nq = torch.sum(label.data==-1) # number of tuples
#     S = x.size(1) // nq # number of images per tuple including query: 1+(1+n)
#
#     x1 = x[:, ::S].permute(1,0).repeat(1,S-1).view((S-1)*nq,dim).permute(1,0)
#     idx = [i for i in range(len(label)) if label.data[i] != -1]
#     x2 = x[:, idx]
#     lbl = label[label!=-1]
#
#     dif = x1 - x2
#     D = torch.pow(dif+eps, 2).sum(dim=0).sqrt()
#
#     y = 0.5*lbl*torch.pow(D,2) + 0.5*(1-lbl)*torch.pow(torch.clamp(margin-D, min=0),2)
#     y = torch.sum(y)
#     return y

# def contrastive_loss(x1, x2, label,
#                      margin=0.7, eps=1e-6, reduce=False):
#     # x is N*D
#     # lbl = label[label!=-1]
#     neg_lbl = (label== -1).float()
#     # pos_lbl = (label== 1).float()
#
#     dif = x1 - x2 # N*D
#     D = torch.pow(dif+eps, 2).sum(dim=1).sqrt()
#
#     y = 0.5*neg_lbl*torch.pow(D,2) + 0.5*(1-neg_lbl)*torch.pow(torch.clamp(margin-D, min=0),2)
#     # y = torch.sum(y)
#     if reduce:
#         return torch.mean(y)
#     else:
#         return torch.sum(y)
#     # return y

def contrastive_loss(total_x, total_label, pos_num, neg_num_for_loss,
                     margin=0.7, eps=1e-6, reduce=False):

    # total_x (b,classes*pos_num, Dim)
    # total_x is N*D => (b,10,pos_num,Dim),
    # total_label:batch_size*[0,0,0,1,1,1,...,9,9,9]
    # total_label, (b*10)*pos_num*1

    bs,nq_si,Dim = total_x.size()
    bs,nq_si = total_label.size()

    idx_combins =  [c for c in combinations(range(nq_si),2)]
    idx_combins = torch.FloatTensor(idx_combins)  # bs * num_comb * 2

    idx1 = idx_combins[:,0].long() #num_comb
    idx2 = idx_combins[:,1].long()

    idx1 = Variable(idx1.cuda())
    idx2 = Variable(idx2.cuda())

    pos_pair = (total_label[:,idx1[:]] == total_label[:,idx2[:]]) #>0
    num_pos = torch.sum(pos_pair.data) // bs
    # print (idx1[pos_pair[:]].view(bs,-1).unsqueeze(-1)).size()
    # print bs,num_pos
    pos_idx1 = idx1[pos_pair[:]].view(bs,-1).unsqueeze(-1).expand(bs,num_pos,Dim)
    pos_idx2 = idx2[pos_pair[:]].view(bs,-1).unsqueeze(-1).expand(bs,num_pos,Dim)
    pos_x1 = torch.gather(total_x, 1, pos_idx1) #total_x[:,idx1[pos_pair],:]
    pos_x2 = torch.gather(total_x, 1, pos_idx2) #total_x[:,idx2[pos_pair],:]

    pos_diff = pos_x1 - pos_x2
    pos_D = torch.pow(pos_diff + eps, 2).sum(dim=1).sqrt()
    pos_loss = 0.5 * torch.pow(pos_D, 2)

    # negtive-sample wise
    neg_pair = 1 - pos_pair.data  # <0
    num_neg = torch.sum(neg_pair) // bs
    neg_idx1 = idx1[neg_pair[:]].view(bs, -1).unsqueeze(-1).expand(bs, num_neg, Dim)
    neg_idx2 = idx2[neg_pair[:]].view(bs, -1).unsqueeze(-1).expand(bs, num_neg, Dim)
    neg_x1 = torch.gather(total_x, 1, neg_idx1)  # total_x[:,idx1[pos_pair],:]
    neg_x2 = torch.gather(total_x, 1, neg_idx2)  # total_x[:,idx2[pos_pair],:]
    neg_x1 = neg_x1.view(-1,Dim)
    neg_x2 = neg_x2.view(-1,Dim)

    rand_num = torch.from_numpy(np.random.permutation(torch.sum(neg_pair))).long()
    rand_num = Variable(rand_num.cuda())
    dis_neg_x1 = neg_x1[rand_num[:neg_num_for_loss*bs]]
    dis_neg_x2 = neg_x2[rand_num[:neg_num_for_loss*bs]]

    neg_diff = dis_neg_x1 - dis_neg_x2
    neg_D = torch.pow(neg_diff + eps, 2).sum(dim=1).sqrt()
    neg_loss = 0.5 * torch.pow(torch.clamp(margin - neg_D, min=0), 2)

    y = torch.sum(pos_loss) + torch.sum(neg_loss)

    return y

#
# bs, _, _ = total_x.size()
# for i_x, x in enumerate(total_x):
#     label = total_label[i_x]
#     num, dim = x.size()
#
#     label = label.view(-1)
#     num_label = label.size(0)
#
#     idx_combins = [c for c in combinations(range(num_label), 2)]
#     idx_combins = torch.FloatTensor(idx_combins).squeeze(0)  # num_comb * 2 [(0,1),(0,2),(1,2)]
#     idx1 = idx_combins[:, 0].long()
#     idx2 = idx_combins[:, 1].long()
#
#     idx1 = Variable(idx1.cuda())
#     idx2 = Variable(idx2.cuda())
#
#     # positive-sample wise
#     pos_pair = (label[idx1] == label[idx2])  # >0
#     pos_x1 = x[idx1[pos_pair], :]
#     pos_x2 = x[idx2[pos_pair], :]
#
#     pos_diff = pos_x1 - pos_x2
#     pos_D = torch.pow(pos_diff + eps, 2).sum(dim=1).sqrt()
#     pos_loss = 0.5 * torch.pow(pos_D, 2)
#
#     # negtive-sample wise
#     neg_inds = torch.nonzero(1 - pos_pair.cpu().data).squeeze(1)
#     rand_num = torch.from_numpy(np.random.permutation(neg_inds.size(0))).long()
#     neg_inds = Variable(neg_inds.cuda())
#     rand_num = Variable(rand_num.cuda())
#     dis_inds = neg_inds[rand_num[:neg_num_for_loss]]  # .squeeze(1)
#
#     neg_x1 = x[idx1[dis_inds], :]
#     neg_x2 = x[idx2[dis_inds], :]
#     neg_diff = neg_x1 - neg_x2
#     neg_D = torch.pow(neg_diff + eps, 2).sum(dim=1).sqrt()
#     neg_loss = 0.5 * torch.pow(torch.clamp(margin - neg_D, min=0), 2)
#
#     y = torch.sum(pos_loss) + torch.sum(neg_loss)
#
# return y