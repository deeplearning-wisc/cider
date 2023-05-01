import os
import torch
import numpy as np
from tqdm import tqdm
import torchvision
import sklearn.metrics as sk
from torchvision.transforms import transforms
import torch.nn.functional as F

from .display_results import print_measures_with_std
from .svhn_loader import SVHN

def set_ood_loader_small(args, out_dataset, img_size = 32):
    '''
        set OOD loader for CIFAR scale datasets
    '''
    root = args.ood_loc
#     normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
#                                          std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447],
                                    std=[0.247, 0.244, 0.262])
    if out_dataset == 'SVHN':
        testsetout = SVHN(root=os.path.join(root, 'svhn'), split='test',
                                transform=transforms.Compose([transforms.Resize(img_size), 
                                transforms.CenterCrop(img_size),transforms.ToTensor(), normalize]), download=False)
    elif out_dataset == 'dtd':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'dtd', 'images'),
                                    transform=transforms.Compose([transforms.Resize(img_size), 
                                    transforms.CenterCrop(img_size ), transforms.ToTensor(),normalize]))
    elif out_dataset == 'places365':
        testsetout = torchvision.datasets.ImageFolder(root= os.path.join(root, 'places365'),
            transform=transforms.Compose([transforms.Resize(img_size), 
            transforms.CenterCrop(img_size), transforms.ToTensor(),normalize]))
    else:
        testsetout = torchvision.datasets.ImageFolder(root = os.path.join(root, out_dataset),
                                    transform=transforms.Compose([transforms.Resize(img_size), 
                                    transforms.CenterCrop(img_size),transforms.ToTensor(),normalize]))
    
    if len(testsetout) > 10000: 
        testsetout = torch.utils.data.Subset(testsetout, np.random.choice(len(testsetout), 10000, replace=False))
    testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size,
                                            shuffle=True, num_workers=4)
    return testloaderOut

def set_ood_loader_ImageNet(args, out_dataset):
    '''
        set OOD loader for ImageNet scale datasets
    '''
    root = args.ood_loc
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
        ])
    if out_dataset == 'iNaturalist':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'iNaturalist'), transform=preprocess)
    elif out_dataset == 'SUN':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'SUN'), transform=preprocess)
    elif out_dataset == 'places365': # subsampled places
        testsetout = torchvision.datasets.ImageFolder(root= os.path.join(root, 'Places'),transform=preprocess)  
    elif out_dataset == 'placesbg': 
        testsetout = torchvision.datasets.ImageFolder(root= os.path.join(root, 'placesbg'),transform=preprocess)  
    elif out_dataset == 'dtd':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'dtd', 'images'),
                                        transform=preprocess)

    testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size,
                                            shuffle=False, num_workers=4)
    return testloaderOut


def obtain_feature_from_loader(args, net, loader, num_batches, embedding_dim = 512):
    out_features = torch.zeros((0, embedding_dim), device = 'cuda')
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(loader)):
            if num_batches is not None:
                if batch_idx >= num_batches:
                    break
            data, target = data.cuda(), target.cuda()
            out_feature = net.intermediate_forward(data) 
            out_features = torch.cat((out_features,out_feature), dim = 0)
    return out_features.cpu().numpy()


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])

def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr

def get_and_print_results(args, in_score, out_score, auroc_list, aupr_list, fpr_list, log = None):
    '''
    1) evaluate detection performance for a given OOD test set (loader)
    2) print results (FPR95, AUROC, AUPR)
    '''
    aurocs, auprs, fprs = [], [], []
    if args.out_as_pos: # in case out samples are defined as positive (as in OE)
        measures = get_measures(out_score, in_score)
    else:
        measures = get_measures(-in_score, -out_score)
    aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])
    print(f'in score samples (random sampled): {in_score[:3]}, out score samples: {out_score[:3]}')
    # print(f'in score samples (min): {in_score[-3:]}, out score samples: {out_score[-3:]}')
    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
    auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr) # used to calculate the avg over multiple OOD test sets
    print_measures_with_std(log, auroc, aupr, fpr, args.score)
