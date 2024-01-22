import argparse
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch
import time
import os
import sys
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torchvision import transforms
import loss
from itertools import chain
from torch.utils.data import DataLoader
import torch.optim as optim

from torch.cuda import amp
from tqdm import tqdm

from pytorch_grad_cam import (
    GradCAM,
)

from trainers.cooooop import *
from visualizer import *
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix

from sklearn.mixture import GaussianMixture
from model import make_model

from data_list import ImageList_idx
from model.cosine_lr import CosineLRScheduler
import random




def reshape_transform(tensor, height=14, width=14):
    # 去掉cls token
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # 将通道维度放到第一个位置
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def obtain_label(val_loader, custom_vit_model, model, epoch, distance='cosine', threshold=0):
    start_test = True
    print('obtain label')
    custom_vit_model.eval()
    model.eval()
    with torch.no_grad():
        iter_test = iter(val_loader)
        for _ in range(len(val_loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.to(device)
            outputs, feas = custom_vit_model(inputs)
            VLM_outputs, _, _ = model(inputs)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                VLM_all_output = VLM_outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                VLM_all_output = torch.cat((VLM_all_output, VLM_outputs.float().cpu()), 0)

    VLM_all_output = nn.Softmax(dim=1)(VLM_all_output)
    all_output = nn.Softmax(dim=1)(all_output)
    mean_all_out = (VLM_all_output + all_output) / 2
    _, predict = torch.max(all_output, 1)
    _, VLM_predict = torch.max(VLM_all_output, 1)
    max_probabilities, Mean_predict = torch.max(mean_all_out, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    VLM_accuracy = torch.sum(torch.squeeze(VLM_predict).float() == all_label).item() / float(all_label.size()[0])
    Mean_accuracy = torch.sum(torch.squeeze(Mean_predict).float() == all_label).item() / float(all_label.size()[0])
    print("VIT,acc", accuracy, "VLM acc:", VLM_accuracy, "Mean_ acc:", Mean_accuracy)

    # #
    if epoch < args.warm_up:
        predict = Mean_predict
        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
        if distance == 'cosine':
            all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
            all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
        ### all_fea: extractor feature [bs,N]

        all_fea = all_fea.float().cpu().numpy()
        K = all_output.size(1)
        aff = all_output.float().cpu().numpy()
        ### aff: softmax output [bs,c]

        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        cls_count = np.eye(K)[predict].sum(axis=0)
        labelset = np.where(cls_count > threshold)
        labelset = labelset[0]
        distance = "cosine"
        dd = cdist(all_fea, initc[labelset], distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]
        acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
        log_str = 'Fisrt Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

        print(log_str)


        print(log_str)
        predict = torch.from_numpy(pred_label)
    else:
        predict = Mean_predict
    predict = predict.to(device)
    return predict


def make_optimizer(model, cfg):
    params = []
    cfg.OPTIMIZER_NAME = "SGD"
    cfg.WEIGHT_DECAY_BIAS = 0.0001
    cfg.WEIGHT_DECAY = 1e-4
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

    lr_min = 0.002 * 0.008
    warmup_lr_init = 0.01 * 0.008

    warmup_t = 10
    noise_range = None

    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_epochs,
        lr_min=lr_min,
        t_mul=1.,
        decay_rate=0.1,
        warmup_lr_init=warmup_lr_init,
        warmup_t=warmup_t,
        cycle_limit=1,
        t_in_epochs=True,
        noise_range_t=noise_range,
        noise_pct=0.67,
        noise_std=1.,
        noise_seed=42,
    )

    return lr_scheduler


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


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()

    dsize = len(txt_tar)
    tr_size = int(0.9 * dsize)

    # print(dsize, tr_size, dsize - tr_size)
    tr_txt, te_txt = torch.utils.data.random_split(txt_tar, [tr_size, dsize - tr_size])
    # tr_txt =te_txt
    dsets["source_tr"] = ImageList_idx(tr_txt, transform=image_train())
    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True,
                                           num_workers=args.worker, drop_last=False)
    dsets["source_te"] = ImageList_idx(te_txt, transform=image_test())
    dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs, shuffle=True,
                                           num_workers=args.worker, drop_last=False)

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                        drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs, shuffle=False, num_workers=args.worker,
                                      drop_last=False)

    return dset_loaders


def cal_acc(loader, custom_vit_model, model, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.to(device)
            outputs, _ = custom_vit_model(inputs)
            VLM_outputs, _, _ = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                VLM_all_output = VLM_outputs.float().cpu()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                VLM_all_output = torch.cat((VLM_all_output, VLM_outputs.float().cpu()), 0)
    VLM_all_output = nn.Softmax(dim=1)(VLM_all_output)
    all_output = nn.Softmax(dim=1)(all_output)
    mean_all_out = (VLM_all_output + all_output) / 2
    _, predict = torch.max(all_output, 1)
    _, VLM_predict = torch.max(VLM_all_output, 1)
    _, mean_ACC = torch.max(mean_all_out, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    VLM_accuracy = torch.sum(torch.squeeze(VLM_predict).float() == all_label).item() / float(all_label.size()[0])
    Mean_accuracy = torch.sum(torch.squeeze(mean_ACC).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(mean_ACC).float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc, VLM_accuracy * 100, Mean_accuracy * 100
    else:
        return accuracy * 100, mean_ent, VLM_accuracy * 100, Mean_accuracy * 100


def train_target(args):
    dset_loaders = data_load(args)

    class_names = []


    if args.dset == 'office-home':
        dataset_root = 'data/office-home/Art'
        class_folders = sorted(os.listdir(dataset_root))
        for folder in class_folders:
            if os.path.isdir(os.path.join(dataset_root, folder)):
                class_names.append(folder)
    elif args.dset == 'office13':
        dataset_root = 'data/office13/amazon'
        class_folders = sorted(os.listdir(dataset_root))
        for folder in class_folders:
            if os.path.isdir(os.path.join(dataset_root, folder)):
                class_names.append(folder)

    elif args.dset == "domainnet":
        with open('categories_domainnet.json', 'r') as file:
            loaded_dict = json.load(file)
        class_names = list(loaded_dict.values())
    elif args.dset == "VISDA-C":
        with open('categories_visda_c.json', 'r') as file:
            loaded_dict = json.load(file)
        class_names = list(loaded_dict.values())
    else:
        print("error, there is no dataset")
    print("所有类别名称：", class_names)


    scaler = amp.GradScaler()
    custom_vit_model = make_model(args, num_class=args.class_num)


    optimizer = make_optimizer(custom_vit_model, args)
    scheduler_vit = create_scheduler(args, optimizer)
    args.modelpath = args.output_dir_src + '/best_custom_vit_model.pt'
    custom_vit_model.load_state_dict(torch.load(args.modelpath))


    cam = GradCAM(model=custom_vit_model,
                  target_layers=[custom_vit_model.base.blocks[-1].norm1],
                  use_cuda=True,
                  reshape_transform=reshape_transform)

    cfg = {"TRAINER": {
        "COCOOP": {
            "PREC": "fp16",
        }
    },

    }

    clip_model = load_clip_to_cpu(cfg)
    model = CustomCLIP(cfg, class_names, clip_model).to(device)

    name_to_update = "prompt_learner"
    for name, param in model.named_parameters():
        if name_to_update not in name and "imageprompt_learner" not in name and "simpnet" not in name:
            param.requires_grad_(False)

    enabled = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            enabled.add(name)
    print(f"Parameters to be updated: {enabled}")

    prompt_learner_params = model.prompt_learner.parameters()
    imageprompt_learner_params = model.imageprompt_learner.parameters()
    simpnet_params = model.simpnet.parameters()
    param_group = chain(prompt_learner_params, imageprompt_learner_params, simpnet_params)

    optim_cfg = {
        "NAME": "sgd",
        "LR": 0.002,
        "MAX_EPOCH": 15,
        "LR_SCHEDULER": "cosine",
        "WARMUP_EPOCH": 1,
        "WARMUP_TYPE": "constant",
        "WARMUP_CONS_LR": 1e-5
    }
    # 根据配置创建优化器
    optimizer_VLM = optim.SGD(param_group, lr=optim_cfg['LR'])

    if optim_cfg['LR_SCHEDULER'] == 'cosine':
        # 使用余弦退火进行学习率调整
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_VLM, T_max=args.max_epoch)
    else:

        scheduler = None  # 或者根据需要选择合适的调度器
    total_iterations = len(dset_loaders["target"])

    for epoch in range(args.max_epoch):

        custom_vit_model.eval()
        model.eval()
        mem_label = obtain_label(dset_loaders['test'], custom_vit_model, model, epoch, args)
        mem_new_label, confidence, bool_tensor = eval_sample(dset_loaders["target"], custom_vit_model, model,
                                                             mem_label, args)
        custom_vit_model.train()
        model.train()

        for i, data in tqdm(enumerate(dset_loaders["target"]), total=total_iterations,
                            desc=f"Epoch {epoch + 1}/{args.max_epoch}"):
            inputs_test, labels_target, tar_idx, _ = data

            if inputs_test.size(0) == 1:
                continue

            inputs_test = inputs_test.to(device)
            pred = mem_label[tar_idx]

            pred_num = pred.cpu().numpy()

            grayscale_cam, cam_14 = cam(input_tensor=inputs_test, targets=pred_num)
            attention_maps_vit = torch.from_numpy(cam_14).to(device).detach()
            optimizer.zero_grad()
            with torch.no_grad():
                pred = mem_label[tar_idx]
                bool_pred = bool_tensor[tar_idx]

                model.eval()
                VLM_outputs, image_features, text_features = model(inputs_test.detach())
                features = image_features @ text_features.t()
                similarity_map = get_similarity_map(features[:, 1:, :], inputs_test.shape[2:])
                similarity_map = similarity_map / (similarity_map.amax(dim=(1, 2), keepdim=True) + 1e-8)

                indices = pred.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, 14, 14, 1)

                attentation_maps = similarity_map.gather(3, indices).squeeze(-1)
                attentation_maps = 1 - attentation_maps

                avg_result = (attention_maps_vit + attentation_maps) / 2.0
                new_masks = mask_operations(avg_result, bool_pred)
                trainble_noise_vit = model(inputs_test.detach(), pred.long(), imageprompts=new_masks.detach(),
                                           vit_prompts=True)
                del avg_result, attentation_maps, similarity_map, features, cam_14, grayscale_cam
                model.train()

            trainble_noise_vit = trainble_noise_vit.to(device)

            optimizer.zero_grad()

            with amp.autocast(enabled=True):
                outputs_test, feat = custom_vit_model(inputs_test, mask_matrix=new_masks,
                                                      trainable_noise=trainble_noise_vit)
                classifier_loss = nn.CrossEntropyLoss()(outputs_test, pred.long())

                if args.ent:
                    softmax_out = nn.Softmax(dim=1)(outputs_test)
                    entropy_loss = torch.mean(loss.Entropy(softmax_out))
                    if args.gent:
                        msoftmax = softmax_out.mean(dim=0)
                        gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                        entropy_loss -= gentropy_loss
                    im_loss = entropy_loss * args.ent_par
                    classifier_loss += im_loss

                scaler.scale(classifier_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(custom_vit_model.parameters(), 1.0)

                scaler.step(optimizer)
                scaler.update()

            inputs_test = inputs_test.to(device)
            pred = mem_label[tar_idx].to(device)

            # 进行前向和反向传播
            loss_summary = model(inputs_test, pred.long(), imageprompts=new_masks, vit_prompts=False)
            loss_summary = loss_summary.sum()
            optimizer_VLM.zero_grad()
            # 打印或记录损失等信息

            loss_summary.backward()
            optimizer_VLM.step()

        scheduler_vit.step(epoch)
        scheduler.step()
        custom_vit_model.eval()

        model.eval()
        if args.dset == 'VISDA-C':
            acc_s_te, acc_list, VLM_accuracy, Mean_accuracy = cal_acc(dset_loaders['test'], custom_vit_model, model,
                                                                      True)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%,VLM_accuracy = {:.2f}%,Mean_accuracy = {:.2f}%'.format(
                args.name, epoch, args.max_epoch,
                acc_s_te, VLM_accuracy, Mean_accuracy) + '\n' + acc_list
        else:
            acc_s_te, _, VLM_accuracy, Mean_accuracy = cal_acc(dset_loaders['test'], custom_vit_model, model, False)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}% ,VLM_accuracy = {:.2f}%,Mean_accuracy = {:.2f}%'.format(
                args.name, epoch, args.max_epoch, acc_s_te, VLM_accuracy, Mean_accuracy)

        args.out_file.write(log_str + '\n')
        args.out_file.flush()
        print(log_str + '\n')
        custom_vit_model.train()

        model.train()

    if args.is_save:
        torch.save(custom_vit_model.state_dict(),
                   osp.join(args.output_dir, "target_custom_vit_model" + args.savename + ".pt"))
        torch.save(model.state_dict(), osp.join(args.output_dir, "target_model" + args.savename + ".pt"))

    return custom_vit_model, model


def mask_operations(mask, bool_pred):
    # 找到错误样本的索引
    incorrect_indices = ~bool_pred.unsqueeze(1).unsqueeze(2)  # 将 bool_pred 扩展为与 mask 相同的形状

    new_masks = mask.clone()  # 复制 mask
    new_masks[incorrect_indices.expand_as(new_masks)] = 1 - new_masks[incorrect_indices.expand_as(new_masks)]

    top_10_percent_indices = int(0.1 * mask.size(1) * mask.size(2))  # 计算每个样本前 10% 的像素数量

    for i in range(mask.size(0)):

        flattened_mask = mask[i].view(-1)
        sorted_mask, _ = torch.sort(flattened_mask)

        threshold_value = sorted_mask[top_10_percent_indices]

        new_masks[i] = mask[i] > threshold_value

    return new_masks


def eval_sample(loader, custom_vit_model, model, label, args):
    loss_func = nn.CrossEntropyLoss(reduction='none')
    losses = np.zeros(len(label))
    start_test = True
    with torch.no_grad():
        for i, data in enumerate(loader):
            inputs = data[0]
            index = data[2]
            inputs = inputs.to(device)
            VLM_outputs, _, _ = model(inputs)
            outputs, feas = custom_vit_model(inputs)
            pred = torch.softmax(outputs, dim=-1)
            VLM_pred = torch.softmax(VLM_outputs, dim=-1)
            mean_pred = (VLM_pred + pred) / 2

            del inputs, VLM_outputs, _, outputs, feas, pred, VLM_pred

            loss = loss_func(mean_pred, label[index]).cpu().detach().numpy()
            index = index.long().cpu().detach().numpy()
            losses[index] = loss
        label_cpu = label.cpu()  # 将张量移至主机内存上
        labels = np.array(label_cpu, dtype=int)

        for now_class in range(args.class_num):
            indices = np.where(labels == now_class)[0]
            if len(indices) > 1:
                losses[indices] = (losses[indices] - losses[indices].mean()) / losses[indices].var()

        gmm = GaussianMixture(
            n_components=2,
            max_iter=10,
            tol=1e-2,
            reg_covar=5e-4
        )
        losses = losses.reshape(-1, 1)
        gmm.fit(losses)
        prob = gmm.predict_proba(losses)
        prob = prob[:, gmm.means_.argmin()]


        prob = torch.from_numpy(prob).to(device)

        decayed_value = args.probs * math.exp(-args.max_epoch)
        bool_tensor = prob > decayed_value
        del label_cpu, labels, indices, losses, gmm
        return label, prob, bool_tensor


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def get_similarity_map(sm, shape):
    # min-max norm
    sm = (sm - sm.min(1, keepdim=True)[0]) / (sm.max(1, keepdim=True)[0] - sm.min(1, keepdim=True)[0])


    side = int(sm.shape[1] ** 0.5)  # square output
    sm = sm.reshape(sm.shape[0], side, side, -1).permute(0, 3, 1, 2)


    sm = sm.permute(0, 2, 3, 1)
    return sm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TDPC')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=2, help="target")
    parser.add_argument('--max_epoch', type=int, default=15, help="max iterations")

    parser.add_argument('--batch_size', type=int, default=16, help="batch_size")
    parser.add_argument('--worker', type=int, default=0, help="number of workers")
    parser.add_argument('--dset', type=str, default='office31',
                        choices=['VISDA-C', 'office31', 'office-home', 'office-caltech', 'domainnet'])

    parser.add_argument('--seed', type=int, default=2021, help="random seed")
    parser.add_argument('--warm_up', type=int, default=0, help="number of warmup")
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=0.1)
    parser.add_argument('--probs', type=float, default=0.1)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--output_src', type=str, default='san')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--is_save', type=bool, default=True)
    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Product', 'Clipart', 'RealWorld']
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
    if args.dset == 'domainnet':
        names = ['clipart', 'painting', 'real', 'sketch']
        args.class_num = 126
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i


        folder = 'data/'
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        print("TASK--", names[args.s], names[args.t])
        if args.dset == 'office-home':
            if args.da == 'pda':
                args.class_num = 65
                args.src_classes = [i for i in range(65)]
                args.tar_classes = [i for i in range(25)]

        args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())
        args.output_dir = osp.join(args.output, args.da, args.dset,
                                   names[args.s][0].upper() + names[args.t][0].upper())

        args.save_dir = osp.join(args.output, args.da, args.dset,
                                 names[args.s][0].upper() + names[args.t][0].upper() + "image")  # 替换为实际文件夹路径

        args.name = names[args.s][0].upper() + names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        if not osp.exists(args.save_dir):
            os.mkdir(args.save_dir)



        path = os.path.abspath(os.path.dirname(sys.argv[0]))

        localtime = time.localtime(time.time())  # 获取当前时间
        time = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
        args.savename = 'par_' + str(args.cls_par) + time + "different_name" ##用于保存时区分文件
        if args.da == 'pda':
            args.gent = ''
            args.savename = 'par_' + str(args.cls_par) + '_thr' + str(args.threshold)
        args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')



        train_target(args)