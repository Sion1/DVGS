import argparse
import os
import random

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from torch import optim, nn
from torchvision import transforms
from tqdm import tqdm

from network.DVGS import DVGS
from utils import str2bool, DataLoader, AverageMeter, compute_accuracy, mkdir_p

parser = argparse.ArgumentParser()

parser.add_argument('--gpu', default=0, type=int, help='gpu num')
parser.add_argument('--epoch', default=80, type=int, help='epoch num')
parser.add_argument('--batch', default=32, type=int, help='batch size')
parser.add_argument('--dataset', default='CUB', type=str, help='dataset: [CUB|AWA2|SUN]')
parser.add_argument('--attr_num', default=312, type=int, help='attributes num: [CUB:312|AWA2:85|SUN:102]')
parser.add_argument('--data_path', default='/home/c402/backup_project/Dataset/CUB_200_2011/images',
                    type=str, help='data path')
# --data_path /home/c402/backup_project/Dataset/Animals_with_Attributes2/JPEGImages
# --data_path /home/c402/backup_project/Dataset/SUN/images

parser.add_argument('--mat_path', default='/home/c402/backup_project/Dataset/xlsa17/data', type=str, help='mat path')
parser.add_argument('--save_path', default='./checkpoint/', type=str, help='network save path')
parser.add_argument('--backbone', default='vit', type=str, help='backbone: [vit|resnet]')
parser.add_argument('--tensorboard', default=False, type=str2bool, help='tensorboard records')
parser.add_argument('--training', default=True, type=str2bool, help='training')
parser.add_argument('--testing', default=False, type=str, help='testing')
parser.add_argument('--cs', default=False, type=str2bool, help='get calibrator stack')
parser.add_argument('--lamb1', default=0.5, type=float, help='the weight of global feature')
parser.add_argument('--lamb2', default=0.5, type=float, help='the weight of V2S feature')
parser.add_argument('--lamb', default=0.001, type=float, help='the weight of L1 regularization')
parser.add_argument('--seed', default=None, type=int, help='manual seed')
parser.add_argument('--gamma', default=None, type=int, help='the calibrator stack weight')

args = parser.parse_args()
for k, v in sorted(vars(args).items()):
    print(k, '=', v)

#### random seed
if args.seed is None:
    args.seed = random.randint(1, 10000)
    print('seed: ' + str(args.seed))
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

DATASET = args.dataset
ROOT = args.data_path

DATA_DIR = f'/home/c402/backup_project/Dataset/xlsa17/data/{DATASET}'
data = sio.loadmat(f'{DATA_DIR}/res101.mat')
# data consists of files names
attrs_mat = sio.loadmat(f'{DATA_DIR}/att_splits.mat')
# attrs_mat is the attributes (class-level information)
image_files = data['image_files']

if DATASET == 'AWA2':
    image_files = np.array([im_f[0][0].split('JPEGImages/')[-1] for im_f in image_files])
else:
    image_files = np.array([im_f[0][0].split('images/')[-1] for im_f in image_files])

# labels are indexed from 1 as it was done in Matlab, so 1 subtracted for Python
labels = data['labels'].squeeze().astype(np.int64) - 1
train_idx = attrs_mat['train_loc'].squeeze() - 1
val_idx = attrs_mat['val_loc'].squeeze() - 1
trainval_idx = attrs_mat['trainval_loc'].squeeze() - 1
test_seen_idx = attrs_mat['test_seen_loc'].squeeze() - 1
test_unseen_idx = attrs_mat['test_unseen_loc'].squeeze() - 1

# consider the train_labels and val_labels
train_labels = labels[train_idx]
val_labels = labels[val_idx]

# split train_idx to train_idx (used for training) and val_seen_idx
train_idx, val_seen_idx = train_test_split(train_idx, test_size=0.2, stratify=train_labels)
# split val_idx to val_idx (not useful) and val_unseen_idx
val_unseen_idx = train_test_split(val_idx, test_size=0.2, stratify=val_labels)[1]
# attribute matrix
attrs_mat = attrs_mat["att"].astype(np.float32).T
args.attrs_mat = torch.tensor(attrs_mat).to(device)

### used for validation
# train files and labels
train_files = image_files[train_idx]
train_labels = labels[train_idx]
uniq_train_labels, train_labels_based0, counts_train_labels = np.unique(train_labels, return_inverse=True,
                                                                        return_counts=True)
# val seen files and labels
val_seen_files = image_files[val_seen_idx]
val_seen_labels = labels[val_seen_idx]
uniq_val_seen_labels = np.unique(val_seen_labels)
# val unseen files and labels
val_unseen_files = image_files[val_unseen_idx]
val_unseen_labels = labels[val_unseen_idx]
uniq_val_unseen_labels = np.unique(val_unseen_labels)

### used for testing
# trainval files and labels
trainval_files = image_files[trainval_idx]
trainval_labels = labels[trainval_idx]
uniq_trainval_labels, trainval_labels_based0, counts_trainval_labels = np.unique(trainval_labels, return_inverse=True,
                                                                                 return_counts=True)
# test seen files and labels
test_seen_files = image_files[test_seen_idx]
test_seen_labels = labels[test_seen_idx]
uniq_test_seen_labels = np.unique(test_seen_labels)
# test unseen files and labels
test_unseen_files = image_files[test_unseen_idx]
test_unseen_labels = labels[test_unseen_idx]
uniq_test_unseen_labels = np.unique(test_unseen_labels)

# Training Transformations
trainTransform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])
# Testing Transformations
testTransform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])

num_workers = 4
### used in validation
# train data loader
train_data = DataLoader(ROOT, train_files, train_labels_based0, transform=trainTransform)
weights_ = 1. / counts_train_labels
weights = weights_[train_labels_based0]
train_sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=train_labels_based0.shape[0],
                                                       replacement=True)
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch, sampler=train_sampler,
                                                num_workers=num_workers)
# seen val data loader
val_seen_data = DataLoader(ROOT, val_seen_files, val_seen_labels, transform=testTransform)
val_seen_data_loader = torch.utils.data.DataLoader(val_seen_data, batch_size=256, shuffle=False,
                                                   num_workers=num_workers)
# unseen val data loader
val_unseen_data = DataLoader(ROOT, val_unseen_files, val_unseen_labels, transform=testTransform)
val_unseen_data_loader = torch.utils.data.DataLoader(val_unseen_data, batch_size=256, shuffle=False,
                                                     num_workers=num_workers)

### used in testing
# trainval data loader
trainval_data = DataLoader(ROOT, trainval_files, trainval_labels_based0, transform=trainTransform)
weights_ = 1. / counts_trainval_labels
weights = weights_[trainval_labels_based0]
trainval_sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=trainval_labels_based0.shape[0],
                                                          replacement=True)
trainval_data_loader = torch.utils.data.DataLoader(trainval_data, batch_size=args.batch, sampler=trainval_sampler,
                                                   num_workers=num_workers)
# seen test data loader
test_seen_data = DataLoader(ROOT, test_seen_files, test_seen_labels, transform=testTransform)
test_seen_data_loader = torch.utils.data.DataLoader(test_seen_data, batch_size=256, shuffle=False,
                                                    num_workers=num_workers)
# unseen test data loader
test_unseen_data = DataLoader(ROOT, test_unseen_files, test_unseen_labels, transform=testTransform)
test_unseen_data_loader = torch.utils.data.DataLoader(test_unseen_data, batch_size=256, shuffle=False,
                                                      num_workers=num_workers)

attr_length = 0
if DATASET == 'AWA2':
    attr_length = 85
elif DATASET == 'CUB':
    attr_length = 312
elif DATASET == 'SUN':
    attr_length = 102
else:
    print("Please specify the dataset, and set {attr_length} equal to the attribute length")

model = DVGS(args).to(device)

optimizer = torch.optim.Adam([{"params": model.v_encoder.parameters(), "lr": 0.00001, "weight_decay": 0.0001},
                              {"params": model.DRS.parameters(), "lr": 0.001, "weight_decay": 0.00001},
                              {"params": model.attribute_classifier.parameters(), "lr": 0.001, "weight_decay": 0.00001},
                              {"params": model.DAS.parameters(), "lr": 0.001, "weight_decay": 0.00001}])
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40], gamma=0.5)

# train attributes
train_attrbs = attrs_mat[uniq_train_labels]
train_attrbs_tensor = torch.from_numpy(train_attrbs).to(device)
# trainval attributes
trainval_attrbs = attrs_mat[uniq_trainval_labels]
trainval_attrbs_tensor = torch.from_numpy(trainval_attrbs).to(device)

if DATASET == 'AWA2':
    # gamma = 1
    # gamma = 0.8
    gamma = 0.95
    # gamma = 0.285
    # gamma = 0.33
elif DATASET == 'CUB':
    # gamma = 0.9
    gamma = 0.8
    # gamma = 0
    # gamma = -0.05500000000000001
    # gamma = 0.045
elif DATASET == 'SUN':
    gamma = 0.7
    # gamma = 0.4
else:
    print("Please specify the dataset, and set {attr_length} equal to the attribute length")
print('Dataset:', DATASET, '\nGamma:', gamma)

#### best results
best_GZSL = [0.0, 0.0, 0.0]
best_ZSL = 0.0

#### global loss
loss_meter = AverageMeter()
L_region_meter = AverageMeter()
L_cls_meter = AverageMeter()

#### global acc
zsl_unseen_acc = 0.0
gzsl_seen_acc = 0.0
gzsl_unseen_acc = 0.0
H = 0.0


def train_DRS(model, data_loader, train_attrbs, optimizer, device, args):
    """returns trained network"""
    global loss_meter
    global L_region_meter
    global L_cls_meter
    """ train the network  """
    model.train()
    tk = tqdm(data_loader)
    for batch_idx, (data, label) in enumerate(tk):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        region_predicted_prototype, global_predicted_prototype = model(data)
        region_logit = region_predicted_prototype @ train_attrbs.T
        global_logit = global_predicted_prototype @ train_attrbs.T
        L_region = F.cross_entropy(region_logit, label)
        L_cls = F.cross_entropy(global_logit, label)
        loss = L_region + L_cls
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item(), label.shape[0])
        L_region_meter.update(L_region.item(), label.shape[0])
        L_cls_meter.update(L_cls.item(), label.shape[0])
        tk.set_postfix(
            {"loss": loss_meter.avg,
             "L_region": L_region_meter.avg,
             "L_cls": L_cls_meter.avg})

    # print training/validation statistics
    print('Train: Average loss: {:.4f}'.format(loss_meter.avg))


def train_DAS(model, data_loader, train_attrbs, optimizer, device, args):
    """returns trained network"""
    global loss_meter
    global L_attr_meter
    """ train the network  """
    model.train()
    args.is_training_DRS = True
    tk = tqdm(data_loader)
    for batch_idx, (data, label) in enumerate(tk):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        region_predicted_prototype, global_predicted_prototype, refined_prototype = model(data)
        region_logit = region_predicted_prototype @ train_attrbs.T
        L_attr = F.cross_entropy(region_logit, label)
        loss = L_attr
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item(), label.shape[0])
        L_attr_meter.update(L_attr.item(), label.shape[0])
        tk.set_postfix(
            {"loss": loss_meter.avg,
             "L_attr": L_attr_meter.avg})

    # print training/validation statistics
    print('Train: Average loss: {:.4f}'.format(loss_meter.avg))


def get_reprs(model, data_loader, device):
    model.eval()
    reprs = []
    for _, (data, _) in enumerate(data_loader):
        data = data.to(device)
        with torch.no_grad():
            local_attr, global_attr = model(data)
            reprs.append(local_attr.cpu().data.numpy())
    reprs = np.concatenate(reprs, 0)
    return reprs


def test(model, test_seen_loader, test_seen_labels, test_unseen_loader, test_unseen_labels, attrs_mat, device, gamma,
         args):
    #### best results
    global best_GZSL
    global best_ZSL
    global zsl_unseen_acc
    global gzsl_seen_acc
    global gzsl_unseen_acc
    global H

    # Representation
    with torch.no_grad():
        seen_reprs = get_reprs(model, test_seen_loader, device)
        unseen_reprs = get_reprs(model, test_unseen_loader, device)
    # Labels
    uniq_test_seen_labels = np.unique(test_seen_labels)
    uniq_test_unseen_labels = np.unique(test_unseen_labels)

    # ZSL
    zsl_unseen_sim = softmax(unseen_reprs @ attrs_mat[uniq_test_unseen_labels].T)
    predict_labels = np.argmax(zsl_unseen_sim, axis=1)
    zsl_unseen_predict_labels = uniq_test_unseen_labels[predict_labels]
    zsl_unseen_acc = compute_accuracy(zsl_unseen_predict_labels, test_unseen_labels, uniq_test_unseen_labels)

    # Calibrated stacking
    Cs_mat = np.zeros(attrs_mat.shape[0])
    Cs_mat[uniq_test_seen_labels] = gamma

    # GZSL
    # seen classes
    gzsl_seen_sim = softmax(seen_reprs @ attrs_mat.T, axis=1)

    ######### calculate similarity #########
    predict_cosine = gzsl_seen_sim[np.arange(gzsl_seen_sim.shape[0]), test_seen_labels]
    predict_cosine = [round(i, 2) for i in predict_cosine]
    counts = {}
    for i in predict_cosine:
        if i in counts:
            counts[i] += 1
        else:
            counts[i] = 1
    plt.bar(counts.keys(), counts.values(), width=0.01)
    # plt.title('Histogram of Probabilities')
    plt.xlabel('cosine similarity')
    plt.ylabel('number of samples')
    plt.show()

    gzsl_seen_predict_labels = np.argmax(gzsl_seen_sim, axis=1)
    gzsl_seen_acc = compute_accuracy(gzsl_seen_predict_labels, test_seen_labels, uniq_test_seen_labels)

    # unseen classes
    gzsl_unseen_sim = softmax(unseen_reprs @ attrs_mat.T, axis=1)

    ######### calculate similarity #########
    predict_cosine = gzsl_unseen_sim[np.arange(gzsl_unseen_sim.shape[0]), test_unseen_labels]
    predict_cosine = [round(i, 2) for i in predict_cosine]
    counts = {}
    for i in predict_cosine:
        if i in counts:
            counts[i] += 1
        else:
            counts[i] = 1
    plt.bar(counts.keys(), counts.values(), width=0.01)
    # plt.title('Histogram of Probabilities')
    plt.xlabel('cosine similarity')
    plt.ylabel('number of samples')
    plt.show()

    gzsl_unseen_predict_labels = np.argmax(gzsl_unseen_sim, axis=1)
    gzsl_unseen_acc = compute_accuracy(gzsl_unseen_predict_labels, test_unseen_labels, uniq_test_unseen_labels)

    H = 2 * gzsl_unseen_acc * gzsl_seen_acc / (gzsl_unseen_acc + gzsl_seen_acc)

    #### update best results
    if best_GZSL[2] < H * 100 and args.testing is None:
        print(' .... Saving GZSL best network ...')
        best_GZSL = [gzsl_unseen_acc * 100, gzsl_seen_acc * 100, H * 100]
        save_path = str(DATASET) + '__GZSL__' + str(pref) + '__BEST' + '.pth'
        ckpt_path = f'./checkpoint/{args.backbone}/' + str(DATASET)
        path = os.path.join(ckpt_path, save_path)
        if not os.path.isdir(ckpt_path):
            mkdir_p(ckpt_path)
        torch.save(model.state_dict(), path)
    if best_ZSL < zsl_unseen_acc * 100 and args.testing is None:
        print(' .... Saving ZSL best network ...')
        best_ZSL = zsl_unseen_acc * 100
        save_path = str(DATASET) + '__ZSL__' + str(pref) + '__BEST' + '.pth'
        ckpt_path = f'./checkpoint/{args.backbone}/' + str(DATASET)
        path = os.path.join(ckpt_path, save_path)
        if not os.path.isdir(ckpt_path):
            mkdir_p(ckpt_path)
        torch.save(model.state_dict(), path)

    print('ZSL: averaged per-class accuracy: {0:.4f}'.format(zsl_unseen_acc * 100))
    print('GZSL Seen: averaged per-class accuracy: {0:.4f}'.format(gzsl_seen_acc * 100))
    print('GZSL Unseen: averaged per-class accuracy: {0:.4f}'.format(gzsl_unseen_acc * 100))
    print('GZSL: harmonic mean (H): {0:.4f}'.format(H * 100))
    print('GZSL: gamma: {0:.2f}'.format(gamma))

    #### print best results
    if args.testing is None:
        print('Best GZSL : U = {:.4f}, S = {:.4f}, H = {:.4f}'.format(best_GZSL[0], best_GZSL[1], best_GZSL[2]))
        print('Best ZSL : ACC = {:.4f} \n'.format(best_ZSL))


def validation(model, seen_loader, seen_labels, unseen_loader, unseen_labels, attrs_mat, device, gamma=None):
    # Representation
    with torch.no_grad():
        seen_reprs = get_reprs(model, seen_loader, device, args)
        unseen_reprs = get_reprs(model, unseen_loader, device, args)

    # Labels
    uniq_labels = np.unique(np.concatenate([seen_labels, unseen_labels]))
    updated_seen_labels = np.searchsorted(uniq_labels, seen_labels)
    uniq_updated_seen_labels = np.unique(updated_seen_labels)
    updated_unseen_labels = np.searchsorted(uniq_labels, unseen_labels)
    uniq_updated_unseen_labels = np.unique(updated_unseen_labels)
    uniq_updated_labels = np.unique(np.concatenate([updated_seen_labels, updated_unseen_labels]))

    # truncate the attribute matrix
    trunc_attrs_mat = attrs_mat[uniq_labels]

    #### ZSL ####
    zsl_unseen_sim = unseen_reprs @ trunc_attrs_mat[uniq_updated_unseen_labels].T
    pred_labels = np.argmax(zsl_unseen_sim, axis=1)
    zsl_unseen_predict_labels = uniq_updated_unseen_labels[pred_labels]
    zsl_unseen_acc = compute_accuracy(zsl_unseen_predict_labels, updated_unseen_labels, uniq_updated_unseen_labels)

    #### GZSL ####
    # seen classes
    gzsl_seen_sim = softmax(seen_reprs @ trunc_attrs_mat.T, axis=1)
    # unseen classes
    gzsl_unseen_sim = softmax(unseen_reprs @ trunc_attrs_mat.T, axis=1)

    gammas = np.arange(0.0, 1.1, 0.1)
    gamma_opt = 0
    H_max = 0
    gzsl_seen_acc_max = 0
    gzsl_unseen_acc_max = 0
    # Calibrated stacking
    for igamma in range(gammas.shape[0]):
        # Calibrated stacking
        gamma = gammas[igamma]
        gamma_mat = np.zeros(trunc_attrs_mat.shape[0])
        gamma_mat[uniq_updated_seen_labels] = gamma

        gzsl_seen_pred_labels = np.argmax(gzsl_seen_sim - gamma_mat, axis=1)
        # gzsl_seen_predict_labels = uniq_updated_labels[pred_seen_labels]
        gzsl_seen_acc = compute_accuracy(gzsl_seen_pred_labels, updated_seen_labels, uniq_updated_seen_labels)

        gzsl_unseen_pred_labels = np.argmax(gzsl_unseen_sim - gamma_mat, axis=1)
        # gzsl_unseen_predict_labels = uniq_updated_labels[pred_unseen_labels]
        gzsl_unseen_acc = compute_accuracy(gzsl_unseen_pred_labels, updated_unseen_labels, uniq_updated_unseen_labels)

        H = 2 * gzsl_seen_acc * gzsl_unseen_acc / (gzsl_seen_acc + gzsl_unseen_acc)

        if H > H_max:
            gzsl_seen_acc_max = gzsl_seen_acc
            gzsl_unseen_acc_max = gzsl_unseen_acc
            H_max = H
            gamma_opt = gamma

    print('ZSL: averaged per-class accuracy: {0:.2f}'.format(zsl_unseen_acc * 100))
    print('GZSL Seen: averaged per-class accuracy: {0:.2f}'.format(gzsl_seen_acc_max * 100))
    print('GZSL Unseen: averaged per-class accuracy: {0:.2f}'.format(gzsl_unseen_acc_max * 100))
    print('GZSL: harmonic mean (H): {0:.2f}'.format(H_max * 100))
    print('GZSL: gamma: {0:.2f}'.format(gamma_opt))
    return gamma_opt


if __name__ == '__main__':

    if args.training:
        #### training DRS ####
        args.is_training_DRS = True
        # only finetune DRS and attribute classifier
        for name, param in model.named_parameters():
            if 'v_encoder' in name or 'DRS' in name or 'attribute_classifier' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        for i in range(args.epoch):
            train_DRS(model, trainval_data_loader, trainval_attrbs_tensor, optimizer, device, args)
            print('Epoch: ', i)
            lr_scheduler.step()
            test(model, test_seen_data_loader, test_seen_labels, test_unseen_data_loader, test_unseen_labels, attrs_mat,
                 device, gamma, args)

        #### training DAS ####
        args.is_training_DRS = False
        for name, param in model.named_parameters():
            if 'DAS' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        for i in range(args.epoch):
            train_DAS(model, trainval_data_loader, trainval_attrbs_tensor, optimizer, device, args)
            print('Epoch: ', i)
            lr_scheduler.step()
            test(model, test_seen_data_loader, test_seen_labels, test_unseen_data_loader, test_unseen_labels, attrs_mat,
                 device, gamma, args)

    if args.testing:
        model.eval()
        model.to(device)
        model_dict = model.state_dict()
        saved_dict = torch.load('/home/c402/project/DVGS/checkpoint/CUB_GZSL_region_selection_1925.pth')
        model_dict.update(saved_dict)
        model.load_state_dict(model_dict, strict=True)
        test(model, test_seen_data_loader, test_seen_labels, test_unseen_data_loader, test_unseen_labels,
             attrs_mat, device, gamma, args)
        # new_dict = {}
        # for k, v in saved_dict.items():
        #     if 'backbone.vit' in k and k is not 'backbone.masker':
        #         new_dict['v_encoder.' + k[9:]] = v
        #     if 'backbone.fc' in k:
        #         new_dict['region_selection.fc.weight'] = v
        #     if 'mlp_g' in k:
        #         new_dict['attribute_classifier.weight'] = v
        #### saving network
        # torch.save(network.state_dict(), './checkpoint/SUN_GZSL_region_selection_2582.pth')
        # torch.save(network.state_dict(), './checkpoint/SUN_ZSL_region_selection_2582.pth')

    if args.cs:
        gammas = []
        for i in range(20):
            train(model, trainval_data_loader, trainval_attrbs_tensor, optimizer, device, args)
            lr_scheduler.step()
            gamma = validation(model, val_seen_data_loader, val_seen_labels, val_unseen_data_loader, val_unseen_labels,
                               attrs_mat, device, args)
            gammas.append(gamma)
        gamma = np.mean(gammas)
        print(gamma)
