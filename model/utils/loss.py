import torch
import torch.nn as nn

import utils_func as LF

# --------------------------------------
# Loss/Error layers
# --------------------------------------

class ContrastiveLoss(nn.Module):
    r"""CONTRASTIVELOSS layer that computes contrastive loss for a batch of images:
        Q query tuples, each packed in the form of (q,p,n1,..nN)
    Args:
        x: tuples arranges in columns as [q,p,n1,nN, ... ]
        label: -1 for query, 1 for corresponding positive, 0 for corresponding negative
        margin: contrastive loss margin. Default: 0.7
    >>> contrastive_loss = ContrastiveLoss(margin=0.7)
    >>> input = torch.randn(128, 35, requires_grad=True)
    >>> label = torch.Tensor([-1, 1, 0, 0, 0, 0, 0] * 5)
    >>> output = contrastive_loss(input, label)
    >>> output.backward()
    """

    def __init__(self, margin=0.7,pos_num=3, neg_num_for_loss=180, eps=1e-6):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.pos_num = pos_num
        self.neg_num_for_loss = neg_num_for_loss
        self.eps = eps

    #total_x, total_label, pos_num, neg_num_for_loss,
                     # margin=0.7, eps=1e-6, reduce=False):
    def forward(self, x1, label, reduce):
        return LF.contrastive_loss(x1, label, pos_num=self.pos_num,
                                   neg_num_for_loss=self.neg_num_for_loss,
                                   margin=self.margin, eps=self.eps,reduce=reduce)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'margin=' + '{:.4f}'.format(self.margin) + ')'