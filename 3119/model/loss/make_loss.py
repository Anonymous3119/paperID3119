# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

# import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
from .mmd_loss import MMD_loss

def make_loss(cfg, num_classes):   
    sampler = 'softmax'
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    cfg.IF_LABELSMOOTH = 'on'
    if cfg.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)    
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, target):
            if cfg.IF_LABELSMOOTH == 'on':
                ID_LOSS = xent(score, target)
            else:
                ID_LOSS = F.cross_entropy(score, target)  
            return ID_LOSS     
    elif cfg.SAMPLER == 'softmax_center':
        def loss_func(score, feat, target):
            if cfg.IF_LABELSMOOTH == 'on':
                return xent(score, target) + \
                        cfg.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
            else:
                return F.cross_entropy(score, target) + \
                        cfg.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
    else:
        print('expected sampler should be softmax, or softmax_center'
              'but got {}'.format(cfg.SAMPLER))

    return loss_func, center_criterion
