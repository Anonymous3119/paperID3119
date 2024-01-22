import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import torch.nn.utils.weight_norm as weightNorm
from collections import OrderedDict
from utils import to_numpy,to_tensor,BlockwiseMaskGenerator
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from typing import List

import torchvision
def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

vgg_dict = {"vgg11":models.vgg11, "vgg13":models.vgg13, "vgg16":models.vgg16, "vgg19":models.vgg19,
"vgg11bn":models.vgg11_bn, "vgg13bn":models.vgg13_bn, "vgg16bn":models.vgg16_bn, "vgg19bn":models.vgg19_bn}
class VGGBase(nn.Module):
  def __init__(self, vgg_name):
    super(VGGBase, self).__init__()
    model_vgg = vgg_dict[vgg_name](pretrained=True)
    self.features = model_vgg.features
    self.classifier = nn.Sequential()
    for i in range(6):
        self.classifier.add_module("classifier"+str(i), model_vgg.classifier[i])
    self.in_features = model_vgg.classifier[6].in_features

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x

res_dict = {"resnet18":models.resnet18, "resnet34":models.resnet34, "resnet50":models.resnet50,
"resnet101":models.resnet101, "resnet152":models.resnet152, "resnext50":models.resnext50_32x4d, "resnext101":models.resnext101_32x8d}

class ResBase(nn.Module):
    def __init__(self, res_name):
        super(ResBase, self).__init__()
        model_resnet = res_dict[res_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

    # def get_downsample_ratio(self: ResNet) -> int:
    #     return 32
    #
    # def get_feature_map_channels(self: ResBase) -> List[int]:
    #     # `self.feature_info` is maintained by `timm`
    #     return [info['num_chs'] for info in self.feature_info[1:]]
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.avgpool(x4)
        x = x.view(x.size(0), -1)
        return x

import torch.nn as nn

class Mask(nn.Module):
    def __init__(self):
        super(Mask, self).__init__()
        self.patch_size = 16
        self.mask_generator = BlockwiseMaskGenerator(input_size=224, mask_patch_size=16, model_patch_size=4, mask_ratio=0.1,
                                                     mask_only=False, mask_color='learnable')

        self.fill_value = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        # x是一个批量的图像数据，x的形状应该是（batch_size, channels, height, width）
        batch_size, _, height, width = x.size()

        # 可视化原始图像（只可视化批量中的第一张图像）
        # original_img = x[0].cpu().numpy().transpose(1, 2, 0)
        # plt.imshow(original_img)
        # plt.axis('off')
        # plt.show()

        # 对批量中的每张图像生成掩码
        masked_images = []
        masks = []
        for i in range(batch_size):  # 逐张处理图像
            masked_image, mask = self.mask_generator(x[i])

            masked_images.append(masked_image)
            masks.append(mask)

        #
        # 可视化处理后的图像（只可视化批量中的第一张处理后的图像）
        # masked_img_numpy = masked_images[0].cpu().numpy().transpose(1, 2, 0)
        # plt.imshow(masked_img_numpy)
        # plt.axis('off')
        # plt.show()

        masked_images_tensor = torch.stack([torch.tensor(img) for img in masked_images])
        masks_tensor= torch.stack([torch.tensor(img) for img in masks])


        return masked_images_tensor, masks_tensor




        #
        #
        #
        # if self.cfg.TRAINER.COOP.WAY == 1:
        #     mask = self.trainble_matrix
        # else:
        #     mask = torch.ones(14, 14).to('cuda')
        #
        # trainble_noise = self.trainble_noise
        #
        # mask_matrix = torch.clamp(mask, 0, 1)
        #
        # print(mask_matrix)
        # mask_matrix = mask_matrix + (torch.bernoulli(mask_matrix) - mask_matrix).detach()
        #
        # return mask_matrix, trainble_noise

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x




class feat_bottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(feat_bottleneck, self).__init__()

        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):

        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x


class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x

class feat_classifier_two(nn.Module):
    def __init__(self, class_num, input_dim, bottleneck_dim=256):
        super(feat_classifier_two, self).__init__()
        self.type = type
        self.fc0 = nn.Linear(input_dim, bottleneck_dim)
        self.fc0.apply(init_weights)
        self.fc1 = nn.Linear(bottleneck_dim, class_num)
        self.fc1.apply(init_weights)

    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        return x

class Res50(nn.Module):
    def __init__(self):
        super(Res50, self).__init__()
        model_resnet = models.resnet50(pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features
        self.fc = model_resnet.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        return x, y

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)
class MetaLinear(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MetaLinear, self).__init__()
        self.ignore = nn.Linear(*args, **kwargs)

        self.register_buffer('weight', to_var(self.ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(self.ignore.bias.data, requires_grad=True))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class CMVNet(nn.Module):
    def __init__(self, input, hidden1, hidden2, output, num_classes):
        super(CMVNet, self).__init__()
        self.feature = nn.Sequential(MetaLinear(input, hidden1), nn.ReLU(inplace=True))
        self.layers = nn.ModuleList()
        for i in range(num_classes):
            self.layers.append(nn.Sequential( MetaLinear(hidden2, output), nn.Sigmoid() ))

    def forward(self, x, num, c):
        # num = torch.argmax(num, -1)
        x = self.feature(x)
        si = x.shape[0]
        output = torch.tensor([]).cuda()
        for i in range(si):

            output = torch.cat((output, self.layers[c[num[i]]](x[i].unsqueeze(0))), 0)


        #print('output:', output.shape)
        return output


class VNet(nn.Module):
    def __init__(self, input, hidden1, output):
        super(VNet, self).__init__()
        self.linear1 = MetaLinear(input, hidden1)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = MetaLinear(hidden1, output)
        # self.linear3 = MetaLinear(hidden2, output)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        # x = self.linear2(x)
        # x = self.relu1(x)
        out = self.linear2(x)
        return F.sigmoid(out)