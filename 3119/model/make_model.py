# import torch
import torch.nn as nn
from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .loss.arcface import ArcFace
from .backbones.resnet_ibn_a import resnet50_ibn_a,resnet101_ibn_a
from .backbones.se_resnet_ibn_a import se_resnet101_ibn_a
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID
from .backbones.vit_pytorch_uda import uda_vit_base_patch16_224_TransReID, uda_vit_small_patch16_224_TransReID
import torch.nn.functional as F
from .loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.task_type = cfg.MODEL.TASK_TYPE

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, frozen_stages=cfg.MODEL.FROZEN,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        elif model_name == 'resnet101':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, frozen_stages=cfg.MODEL.FROZEN,
                               layers=[3, 4, 23, 3])
            print('using resnet101 as a backbone')
        elif model_name == 'resnet50_ibn_a':
            self.in_planes = 2048
            self.base = resnet50_ibn_a(last_stride)
            print('using resnet50_ibn_a as a backbone')
        elif model_name == 'resnet101_ibn_a':
            self.in_planes = 2048
            self.base = resnet101_ibn_a(last_stride, frozen_stages=cfg.MODEL.FROZEN)
            print('using resnet101_ibn_a as a backbone')
        elif model_name == 'se_resnet101_ibn_a':
            self.in_planes = 2048
            self.base = se_resnet101_ibn_a(last_stride,frozen_stages=cfg.MODEL.FROZEN)
            print('using se_resnet101_ibn_a as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))
        elif pretrain_choice == 'un_pretrain':
            self.base.load_un_param(model_path)
            print('Loading un_pretrain model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes

        if self.cos_layer:
            print('using cosine layer')
            self.arcface = ArcFace(self.in_planes, self.num_classes, s=30.0, m=0.50)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.bottleneck_2 = nn.LayerNorm(self.in_planes)
        
    def forward(self, x, label=None, cam_label=None, view_label=None, return_logits=False):  # label is unused if self.cos_layer == 'no'
        
        x = self.base(x, cam_label=cam_label)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if return_logits:
            cls_score = self.classifier(feat)
            return cls_score
        
        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)

            return cls_score, global_feat  # global feature for triplet loss
        elif self.task_type == 'classify_DA': # test for classify domain adapatation
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score
        
        else:

            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            # if 'classifier' in i or 'arcface' in i:
            #     continue
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from revise {}'.format(trained_path))

    def load_un_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in self.state_dict():
            if 'classifier' in i or 'arcface' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer(nn.Module):
    def __init__(self, num_classes, cfg, factory):
        super(build_transformer, self).__init__()
        cfg.TASK_TYPE ='classify_DA'

        model_path = './data/pretrainModel/image_vit_16.pth'  ##预训练文件位置
        cfg.Transformer_TYPE = 'vit_base_patch16_224_TransReID'
        pretrain_choice = 'imagenet'
        cfg.SIZE_CROP =[224,224]
        cfg.AIE_COE =1.5
        cfg.SIZE_TRAIN=[256, 256]
        cfg.STRIDE_SIZE = [16,16]
        cfg.LOCAL_F = False
        cfg.DROP_PATH =0.1
        cfg.COSINE_SCALE =30
        cfg.COSINE_MARGIN = 0.5
        self.cos_layer = False#false
        self.neck = "bnneck"  #bnneck
        self.neck_feat =  'after'  #after
        self.task_type = 'classify_DA' #classify_DA
        if '384' in cfg.Transformer_TYPE or 'small' in cfg.Transformer_TYPE:
            self.in_planes = 384 
        else:
            self.in_planes = 768
        self.bottleneck_dim = 256
        print('using Transformer_type: {} as a backbone'.format(cfg.Transformer_TYPE))
        if cfg.TASK_TYPE == 'classify_DA':
            self.base = factory[cfg.Transformer_TYPE](img_size=cfg.SIZE_CROP, aie_xishu=cfg.AIE_COE,local_feature=cfg.LOCAL_F, stride_size=cfg.STRIDE_SIZE, drop_path_rate=cfg.DROP_PATH)
        else:
            self.base = factory[cfg.Transformer_TYPE](img_size=cfg.SIZE_TRAIN, aie_xishu=cfg.AIE_COE,local_feature=cfg.LOCAL_F, stride_size=cfg.STRIDE_SIZE, drop_path_rate=cfg.DROP_PATH)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = "softmax" #softmax

        self.SFDA = "pretrain"
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.COSINE_SCALE,cfg.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                    s=cfg.COSINE_SCALE, m=cfg.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.COSINE_SCALE,cfg.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                    s=cfg.COSINE_SCALE, m=cfg.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.COSINE_SCALE,cfg.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.COSINE_SCALE, m=cfg.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.COSINE_SCALE, cfg.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.COSINE_SCALE, m=cfg.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        if cfg.train_source == True:
            self._load_parameter(pretrain_choice, model_path)

    def _load_parameter(self, pretrain_choice, model_path):
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))
        elif pretrain_choice == 'un_pretrain':
            self.base.load_un_param(model_path)
            print('Loading trans_tune model......from {}'.format(model_path))
        elif pretrain_choice == 'pretrain':
            self.load_param_finetune(model_path)
            print('Loading pretrained model......from {}'.format(model_path))

    def forward(self, x, mask_matrix=None,trainable_noise=None):  # label is unused if self.cos_layer == 'no'

        # last_self_attention_layer = self.base.transformer[-1].attn
        global_feat,att = self.base(x, mask_matrix, trainable_noise)

        feat = self.bottleneck(global_feat)

        # if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
        #     cls_score = self.classifier(feat, label)
        # else:
        cls_score = self.classifier(feat)
        return cls_score, global_feat
            # else:
            #     if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
            #         cls_score = self.classifier(feat, label)
            #     else:
            #         cls_score = self.classifier(feat)
            #     return cls_score

        # if return_logits:
        #     if self.cos_layer:
        #         cls_score = self.arcface(feat, label)
        #     else:
        #         cls_score = self.classifier(feat)
        #     return cls_score
        # elif self.training:
        #     if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
        #         cls_score = self.classifier(feat, label)
        #     else:
        #         cls_score = self.classifier(feat)
        #
        #     return cls_score, global_feat  # global feature for triplet loss
        # elif self.is_obtain_label:
        #     if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
        #         cls_score = self.classifier(feat, label)
        #     else:
        #         cls_score = self.classifier(feat)
        #
        #     return cls_score, global_feat
        # elif self.SFDA == "SFDA":
        #     cls_score = self.classifier(feat)
        #     return cls_score, global_feat
        # else:
        #     if self.neck_feat == 'after':
        #         # print("Test with feature after BN")
        #         return feat
        #     else:
        #         # print("Test with feature before BN")
        #         return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i or 'arcface' in i or 'bottleneck' in i or 'gap' in i:
                continue
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))


    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'module.' in i: new_i = i.replace('module.','') 
            else: new_i = i 
            if new_i not in self.state_dict().keys():
                print('model parameter: {} not match'.format(new_i))
                continue
            self.state_dict()[new_i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))







__factory_hh = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID, 
    'uda_vit_small_patch16_224_TransReID': uda_vit_small_patch16_224_TransReID, 
    'uda_vit_base_patch16_224_TransReID': uda_vit_base_patch16_224_TransReID,
    # 'resnet101': resnet101,
}

def make_model(cfg, num_class):

    model = build_transformer(num_class,  cfg, __factory_hh)
    print('===========building transformer===========')

    return model
