import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import logging
import torch
import torch.nn as nn
from torch.cuda import  amp
import cv2
from visualizer import *
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList
import random

from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from transformers import ViTModel
from model import make_model
from pytorch_grad_cam import (
    GradCAM,
)
from model.loss import make_loss
from data_list import ImageList, ImageList_idx
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

""" Scheduler Factory
Hacked together by / Copyright 2020 Ross Wightman
"""
from model.cosine_lr import CosineLRScheduler

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def reshape_transform(tensor, height=14, width=14):
    # 去掉cls token
    result = tensor[:, 1:, :].reshape(tensor.size(0),
    height, width, tensor.size(2))

    # 将通道维度放到第一个位置
    result = result.transpose(2, 3).transpose(1, 2)
    return result
def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_src = []
        for i in range(len(txt_src)):
            rec = txt_src[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.src_classes:
                line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
                new_src.append(line)
        txt_src = new_src.copy()

        new_tar = []
        for i in range(len(txt_test)):
            rec = txt_test[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'
                    new_tar.append(line)
        txt_test = new_tar.copy()

    if args.trte == "val":
        dsize = len(txt_src)
        tr_size = int(0.9 * dsize)
        # print(dsize, tr_size, dsize - tr_size)
        tr_txt, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
    else:
        dsize = len(txt_src)
        tr_size = int(0.9 * dsize)
        _, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
        tr_txt = txt_src

    dsets["source_tr"] = ImageList_idx(tr_txt, transform=image_train())
    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True,
                                           num_workers=args.worker, drop_last=False)
    dsets["source_te"] = ImageList_idx(te_txt, transform=image_test())
    dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs, shuffle=True,
                                           num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 2, shuffle=True, num_workers=args.worker,
                                      drop_last=False)

    return dset_loaders


def cal_acc(loader,custom_vit_model, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs,_ = custom_vit_model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(all_output)).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy * 100, mean_ent


def cal_acc_oda(loader, netF, netB, netC):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1) / np.log(args.class_num)
    ent = ent.float().cpu()
    initc = np.array([[0], [1]])
    kmeans = KMeans(n_clusters=2, random_state=0, init=initc, n_init=1).fit(ent.reshape(-1, 1))
    threshold = (kmeans.cluster_centers_).mean()

    predict[ent > threshold] = args.class_num
    matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
    matrix = matrix[np.unique(all_label).astype(int), :]

    acc = matrix.diagonal() / matrix.sum(axis=1) * 100
    unknown_acc = acc[-1:].item()

    return np.mean(acc[:-1]), np.mean(acc), unknown_acc
    # return np.mean(acc), np.mean(acc[:-1])


def train_source(args):
    dset_loaders = data_load(args)

    scaler = amp.GradScaler()
    custom_vit_model = make_model(args, num_class=args.class_num).cuda()
    loss_fn, center_criterion = make_loss(args, num_classes=args.class_num)
    optimizer = make_optimizer(custom_vit_model, args)
    scheduler = create_scheduler(args, optimizer)
    logger = logging.getLogger("reid_baseline.train")
    logger.info('start training')



    acc_init = 0
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])

    iter_num = 0

    custom_vit_model.train()
    epochs=15
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for epoch in range(1, epochs + 1):


        loss_meter.reset()
        acc_meter.reset()
        scheduler.step(epoch)
        iter_num +=1
        for n_iter, (inputs_source, labels_source, target_cam, target_view) in enumerate(dset_loaders["source_tr"]):
            if (len(inputs_source) == 1): continue
            optimizer.zero_grad()

        # lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

            inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()

            with amp.autocast(enabled=True):
                score, feat = custom_vit_model(inputs_source)
                loss = loss_fn(score, labels_source)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(custom_vit_model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            if isinstance(score, list):
                acc = (score[0].max(1)[1] == labels_source).float().mean()
            else:
                acc = (score.max(1)[1] == labels_source).float().mean()

            loss_meter.update(loss.item(), inputs_source.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            # if (n_iter + 1) % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(epoch, (n_iter + 1), len(dset_loaders["source_tr"]),
                                loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))


        print("epoch:",epoch,"Loss:",loss_meter.avg,"Acc:",acc_meter.avg)

        if epoch % 1 == 0:
            custom_vit_model.eval()

            if args.dset == 'VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['source_te'], custom_vit_model, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, iter_num, max_iter,
                                                                            acc_s_te) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(dset_loaders['source_te'], custom_vit_model, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, iter_num, max_iter, acc_s_te)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')

            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                best_custom_vit_model = custom_vit_model.state_dict()
                torch.save(best_custom_vit_model, osp.join(args.output_dir_src, "best_custom_vit_model.pt"))
            file_name = f"custom_vit_model_epoch{epoch}.pt"
            torch.save(custom_vit_model.state_dict(), osp.join(args.output_dir_src, file_name))
            custom_vit_model.train()





    # return best_custom_vit_model


def test_target(args):
    dset_loaders = data_load(args)
    ## set base network
    ##
    args.train_source = True   ##如果预训练vit，请下载image_vit_16预训练参数
    custom_vit_model = make_model(args, num_class=args.class_num).cuda()
    args.modelpath = args.output_dir_src + '/best_custom_vit_model.pt'
    custom_vit_model.load_state_dict(torch.load(args.modelpath))


    custom_vit_model.eval()
    if args.da == 'oda':
        acc_os1, acc_os2, acc_unknown = cal_acc_oda(dset_loaders['test'], custom_vit_model)
        log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}% / {:.2f}% / {:.2f}%'.format(args.trte, args.name,
                                                                                            acc_os2, acc_os1,
                                                                                            acc_unknown)
    else:
        if args.dset == 'VISDA-C':
            acc, acc_list = cal_acc(dset_loaders['test'], custom_vit_model, True)
            log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(args.trte, args.name, acc) + '\n' + acc_list
        else:
            acc, _ = cal_acc(dset_loaders['test'], custom_vit_model, False)
            log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(args.trte, args.name, acc)

    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s
def make_optimizer( model,cfg):
    params = []
    cfg.OPTIMIZER_NAME ="SGD"
    cfg.WEIGHT_DECAY_BIAS = 0.0001
    cfg.WEIGHT_DECAY =1e-4
    cfg.LARGE_FC_LR = False
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = 0.008
        cfg.BASE_LR = 0.008
        weight_decay = cfg.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.BASE_LR * 2
            weight_decay = cfg.WEIGHT_DECAY_BIAS
        if cfg.LARGE_FC_LR:
            if "classifier" in key or "arcface" in key:
                lr = cfg.BASE_LR * 2
                print('Using two times learning rate for fc ')
        # if 'pos_embed' in key or 'cls_token' in key or 'cam_embed' in key:
        #     print(key, 'key with no weight decay')
        #     weight_decay = 0.
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.OPTIMIZER_NAME)(params, momentum=0.9)
    elif cfg.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.OPTIMIZER_NAME)(params)
    # optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.CENTER_LR)

    return optimizer



def create_scheduler(cfg, optimizer):
    num_epochs = 15

    lr_min = 0.002 *  0.008
    warmup_lr_init = 0.01 * 0.008


    warmup_t =10
    noise_range = None

    lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs,
            lr_min=lr_min,
            t_mul= 1.,
            decay_rate=0.1,
            warmup_lr_init=warmup_lr_init,
            warmup_t=warmup_t,
            cycle_limit=1,
            t_in_epochs=True,
            noise_range_t=noise_range,
            noise_pct= 0.67,
            noise_std= 1.,
            noise_seed=42,
        )

    return lr_scheduler





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TDPC')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='2', help="device id to run")
    parser.add_argument('--s', type=int, default=3, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=15, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=0, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home',
                        choices=['VISDA-C', 'office31', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="vgg16, resnet50, resnet101")
    parser.add_argument('--seed', type=int, default=2021, help="random seed")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda', 'oda'])
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65
    if args.dset == 'office13':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    folder = './data/'
    args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
    args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

    if args.dset == 'office-home':
        if args.da == 'pda':
            args.class_num = 65
            args.src_classes = [i for i in range(65)]
            args.tar_classes = [i for i in range(25)]
        if args.da == 'oda':
            args.class_num = 25
            args.src_classes = [i for i in range(25)]
            args.tar_classes = [i for i in range(65)]

    args.output_dir_src = osp.join(args.output, args.da, args.dset, names[args.s][0].upper())
    args.name_src = names[args.s][0].upper()
    if not osp.exists(args.output_dir_src):
        os.system('mkdir -p ' + args.output_dir_src)
    if not osp.exists(args.output_dir_src):
        os.mkdir(args.output_dir_src)

    args.out_file = open(osp.join(args.output_dir_src, 'log.txt'), 'w')
    args.out_file.write(print_args(args) + '\n')
    args.out_file.flush()
    train_source(args)

    args.out_file = open(osp.join(args.output_dir_src, 'log_test.txt'), 'w')
    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i
        args.name = names[args.s][0].upper() + names[args.t][0].upper()

        folder = './data/'
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

        if args.dset == 'office-home':
            if args.da == 'pda':
                args.class_num = 65
                args.src_classes = [i for i in range(65)]
                args.tar_classes = [i for i in range(25)]
            if args.da == 'oda':
                args.class_num = 25
                args.src_classes = [i for i in range(25)]
                args.tar_classes = [i for i in range(65)]

        test_target(args)