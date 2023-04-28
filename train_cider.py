import argparse
import math
import os
import time
from datetime import datetime
import logging
import tensorboard_logger as tb_logger
import pprint

import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import numpy as np

from utils import (CompLoss, DisLoss, DisLPLoss, SupConLoss, 
                AverageMeter, adjust_learning_rate, warmup_learning_rate, 
                set_loader_small, set_loader_ImageNet, set_model)

parser = argparse.ArgumentParser(description='Training with CIDER and SupCon Loss')
parser.add_argument('--gpu', default=7, type=int, help='which GPU to use')
parser.add_argument('--seed', default=4, type=int, help='random seed')
parser.add_argument('--w', default=1, type=float,
                    help='loss scale')
parser.add_argument('--proto_m', default= 0.5, type=float,
                   help='weight of prototype update')
parser.add_argument('--feat_dim', default = 128, type=int,
                    help='feature dim')
parser.add_argument('--in-dataset', default="CIFAR-100", type=str, help='in-distribution dataset')
parser.add_argument('--id_loc', default="datasets/CIFAR100", type=str, help='location of in-distribution dataset')
parser.add_argument('--model', default='resnet34', type=str, help='model architecture: [resnet18, resnet34]')
parser.add_argument('--head', default='mlp', type=str, help='either mlp or linear head')
parser.add_argument('--loss', default = 'cider', type=str, choices = ['supcon', 'cider'],
                    help='train loss')
parser.add_argument('--epochs', default=500, type=int,
                    help='number of total epochs to run')
parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
parser.add_argument('--save-epoch', default=100, type=int,
                    help='save the model every save_epoch')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default= 512, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--learning_rate', default=0.5, type=float,
                    help='initial learning rate')
# if linear lr schedule
parser.add_argument('--lr_decay_epochs', type=str, default='100,150,180',
                        help='where to decay lr, can be a list')
parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
# if cosine lr schedule
parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 0.0001)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--temp', type=float, default=0.1,
                        help='temperature for loss function')
parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
parser.add_argument('--normalize', action='store_true',
                        help='normalize feat embeddings')
parser.add_argument('--subset', default=False,
                        help='whether to use subset of training set to init prototypes')
parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)

args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}

date_time = datetime.now().strftime("%d_%m_%H:%M")

#processing str to list for linear lr scheduling
args.lr_decay_epochs = [int(step) for step in args.lr_decay_epochs.split(',')]

if args.loss == 'supcon':
    args.name = date_time + "_" + 'supcon_{}_lr_{}_cosine_{}_bsz_{}_{}_{}_{}_trial_{}_temp_{}_{}_{}'.\
        format(args.model, args.learning_rate, args.cosine,
               args.batch_size, args.loss, args.epochs, args.feat_dim, args.trial, args.temp, args.in_dataset, args.head)
elif args.loss == 'cider': 
    args.name = (f"{date_time}_cider_{args.model}_lr_{args.learning_rate}_cosine_"
        f"{args.cosine}_bsz_{args.batch_size}_{args.loss}_wd_{args.w}_{args.epochs}_{args.feat_dim}_"
        f"trial_{args.trial}_temp_{args.temp}_{args.in_dataset}_pm_{args.proto_m}")

args.log_directory = "logs/{in_dataset}/{name}/".format(in_dataset=args.in_dataset, name=args.name)
args.model_directory = "checkpoints/{in_dataset}/{name}/".format(in_dataset=args.in_dataset, name= args.name )
args.tb_path = './save/cider/{}_tensorboard'.format(args.in_dataset)
if not os.path.exists(args.model_directory):
    os.makedirs(args.model_directory)
if not os.path.exists(args.log_directory):
    os.makedirs(args.log_directory)
args.tb_folder = os.path.join(args.tb_path, args.name)
if not os.path.isdir(args.tb_folder):
    os.makedirs(args.tb_folder)

#save args
with open(os.path.join(args.log_directory, 'train_args.txt'), 'w') as f:
    f.write(pprint.pformat(state))

#init log
log = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s : %(message)s')
fileHandler = logging.FileHandler(os.path.join(args.log_directory, "train_info.log"), mode='w')
fileHandler.setFormatter(formatter)
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(formatter)
log.setLevel(logging.DEBUG)
log.addHandler(fileHandler)
log.addHandler(streamHandler) 

log.debug(state)

if args.in_dataset == "CIFAR-10":
    args.n_cls = 10
elif args.in_dataset in ["CIFAR-100", "ImageNet-100"]:
    args.n_cls = 100


#set seeds
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
log.debug(f"{args.name}")

# warm-up for large-batch training
if args.batch_size > 256:
    args.warm = True
if args.warm:
    args.warmup_from = 0.001
    args.warm_epochs = 10
    if args.cosine:
        eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
        args.warmup_to = eta_min + (args.learning_rate - eta_min) * (
                1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
    else:
        args.warmup_to = args.learning_rate


def main():
    tb_log = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    if args.in_dataset == "ImageNet-100":
        train_loader, val_loader = set_loader_ImageNet(args)
        aux_loader, _  = set_loader_ImageNet(args, eval = True)
    else:
        train_loader, val_loader = set_loader_small(args)
        aux_loader, _ = set_loader_small(args, eval = True)

    model = set_model(args)

    criterion_supcon = SupConLoss(temperature=args.temp).cuda()
    criterion_comp = CompLoss(args, temperature=args.temp).cuda()
    # V1: learnable prototypes
    # criterion_dis = DisLPLoss(args, model, val_loader, temperature=args.temp).cuda() # V1: learnable prototypes
    # optimizer = torch.optim.SGD([ {"params": model.parameters()},
    #                               {"params": criterion_dis.prototypes}  
    #                             ], lr = args.learning_rate,
    #                             momentum=args.momentum,
    #                             nesterov=True,
    #                             weight_decay=args.weight_decay)

    # V2: EMA style prototypes
    criterion_dis = DisLoss(args, model, aux_loader, temperature=args.temp).cuda() # V2: prototypes with EMA style update
    optimizer = torch.optim.SGD(model.parameters(), lr = args.learning_rate,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.weight_decay)


    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(args, optimizer, epoch)
        # train for one epoch
        train_sloss, train_uloss, train_dloss = train_cider(args, train_loader, model, criterion_supcon, criterion_comp, criterion_dis, optimizer, epoch, log)
        if args.loss == 'supcon':
            tb_log.log_value('train_sup_loss', train_sloss, epoch)
        elif args.loss == 'cider':
            tb_log.log_value('train_uni_loss', train_uloss, epoch)
            tb_log.log_value('train_dis_loss', train_dloss, epoch)
        # tensorboard logger
        tb_log.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        # save checkpoint
        if (epoch + 1) % args.save_epoch == 0: 
            if args.loss == 'supcon':
                save_checkpoint(args, {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'opt_state_dict': optimizer.state_dict(),
                    'sup_state_dict': criterion_supcon.state_dict(),
                }, epoch + 1)
            elif args.loss == 'cider':
                save_checkpoint(args, {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'opt_state_dict': optimizer.state_dict(),
                    'dis_state_dict': criterion_dis.state_dict(),
                    'uni_state_dict': criterion_comp.state_dict(),
                }, epoch + 1)


def train_cider(args, train_loader, model, criterion_supcon, criterion_comp, criterion_dis, optimizer, epoch, log):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    supcon_losses = AverageMeter()
    comp_losses = AverageMeter()
    dis_losses = AverageMeter()

    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        warmup_learning_rate(args, epoch, i, len(train_loader), optimizer)
        bsz = target.shape[0]
        input = torch.cat([input[0], input[1]], dim=0).cuda()
        target = target.repeat(2).cuda()

        penultimate = model.encoder(input).squeeze()
        if args.normalize: # default: False 
            penultimate= F.normalize(penultimate, dim=1)
        features= model.head(penultimate)
        features= F.normalize(features, dim=1)
        if args.loss == 'cider':
            # dis_loss = criterion_dis.compute() # V1: learnable prototypes
            dis_loss = criterion_dis(features, target) # V2: EMA style
            comp_loss = criterion_comp(features, criterion_dis.prototypes, target)
            loss = args.w * comp_loss + dis_loss
            dis_losses.update(dis_loss.data, input.size(0))
            comp_losses.update(comp_loss.data, input.size(0))
        elif args.loss == 'supcon':
            f1, f2 = torch.split(features, [bsz, bsz], dim=0) #f1 shape: [bz, feat_dim]
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1) #features shape: [bz, 2, feat_dim]
            supcon_loss = criterion_supcon(features, target[:bsz])
            supcon_losses.update(supcon_loss.data, input.size(0))
            loss = supcon_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0: 
            if args.loss == 'cider':
                log.debug('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Dis Loss {dloss.val:.4f} ({dloss.avg:.4f})\t'
                    'Comp Loss {uloss.val:.4f} ({uloss.avg:.4f})\t'.format(
                        epoch, i, len(train_loader), batch_time=batch_time, dloss=dis_losses, uloss = comp_losses))
            elif args.loss == 'supcon':
                log.debug('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'SupCon Loss {sloss.val:.4f} ({sloss.avg:.4f})\t'.format(
                        epoch, i, len(train_loader), batch_time=batch_time, sloss=supcon_losses))

    return supcon_losses.avg, comp_losses.avg, dis_losses.avg 


def save_checkpoint(args, state, epoch):
    """Saves checkpoint to disk"""
    filename = args.model_directory + 'checkpoint_{}.pth.tar'.format(epoch)
    torch.save(state, filename)


if __name__ == '__main__':
    main()
