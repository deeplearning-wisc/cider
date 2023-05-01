
"""
Aapted from SupCon: https://github.com/HobbitLong/SupContrast/
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output


class Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg = 0.1, alpha = 32):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        
    def forward(self, X, T):
        P = self.proxies

        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        P_one_hot = binarize(T = T, nb_classes = self.nb_classes)
        N_one_hot = 1 - P_one_hot
    
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)   # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)   # The number of positive proxies
        
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term     
        
        return loss

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class CompLoss(nn.Module):
    '''
    Compactness Loss with class-conditional prototypes
    '''
    def __init__(self, args, temperature=0.07, base_temperature=0.07):
        super(CompLoss, self).__init__()
        self.args = args
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, prototypes, labels):
        prototypes = F.normalize(prototypes, dim=1) 
        proxy_labels = torch.arange(0, self.args.n_cls).cuda()
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, proxy_labels.T).float().cuda() #bz, cls

        # compute logits
        feat_dot_prototype = torch.div(
            torch.matmul(features, prototypes.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(feat_dot_prototype, dim=1, keepdim=True)
        logits = feat_dot_prototype - logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(logits) 
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) 

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos.mean()
        return loss


class DisLPLoss(nn.Module):
    '''
    Dispersion Loss with learnable prototypes
    '''
    def __init__(self, args, model, loader, temperature= 0.1, base_temperature=0.1):
        super(DisLPLoss, self).__init__()
        self.args = args
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.model = model
        self.loader = loader
        self.init_class_prototypes()

    def compute(self):
        num_cls = self.args.n_cls
        # l2-normalize the prototypes if not normalized
        prototypes = F.normalize(self.prototypes, dim=1) 

        labels = torch.arange(0, num_cls).cuda()
        labels = labels.contiguous().view(-1, 1)

        mask = (1- torch.eq(labels, labels.T).float()).cuda()

        logits = torch.div(
            torch.matmul(prototypes, prototypes.T),
            self.temperature)

        mean_prob_neg = torch.log((mask * torch.exp(logits)).sum(1) / mask.sum(1))
        mean_prob_neg = mean_prob_neg[~torch.isnan(mean_prob_neg)]
        # loss
        loss = self.temperature / self.base_temperature * mean_prob_neg.mean()

        return loss
    
    def init_class_prototypes(self):
        """Initialize class prototypes"""
        self.model.eval()
        start = time.time()
        prototype_counts = [0]*self.args.n_cls
        with torch.no_grad():
            prototypes = torch.zeros(self.args.n_cls,self.args.feat_dim).cuda()
            for input, target in self.loader:
                input, target = input.cuda(), target.cuda()
                features = self.model(input) # extract normalized features
                for j, feature in enumerate(features):
                    prototypes[target[j].item()] += feature
                    prototype_counts[target[j].item()] += 1
            for cls in range(self.args.n_cls):
                prototypes[cls] /=  prototype_counts[cls] 
            # measure elapsed time
            duration = time.time() - start
            print(f'Time to initialize prototypes: {duration:.3f}')
            prototypes = F.normalize(prototypes, dim=1)
            self.prototypes = torch.nn.Parameter(prototypes)

class DisLoss(nn.Module):
    '''
    Dispersion Loss with EMA prototypes
    '''
    def __init__(self, args, model, loader, temperature= 0.1, base_temperature=0.1):
        super(DisLoss, self).__init__()
        self.args = args
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.register_buffer("prototypes", torch.zeros(self.args.n_cls,self.args.feat_dim))
        self.model = model
        self.loader = loader
        self.init_class_prototypes()

    def forward(self, features, labels):    

        prototypes = self.prototypes
        num_cls = self.args.n_cls
        for j in range(len(features)):
            prototypes[labels[j].item()] = F.normalize(prototypes[labels[j].item()] *self.args.proto_m + features[j]*(1-self.args.proto_m), dim=0)
        self.prototypes = prototypes.detach()
        labels = torch.arange(0, num_cls).cuda()
        labels = labels.contiguous().view(-1, 1)

        mask = (1- torch.eq(labels, labels.T).float()).cuda()


        logits = torch.div(
            torch.matmul(prototypes, prototypes.T),
            self.temperature)

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(num_cls).view(-1, 1).cuda(),
            0
        )
        mask = mask * logits_mask
        mean_prob_neg = torch.log((mask * torch.exp(logits)).sum(1) / mask.sum(1))
        mean_prob_neg = mean_prob_neg[~torch.isnan(mean_prob_neg)]
        loss = self.temperature / self.base_temperature * mean_prob_neg.mean()
        return loss

    def init_class_prototypes(self):
        """Initialize class prototypes"""
        self.model.eval()
        start = time.time()
        prototype_counts = [0]*self.args.n_cls
        with torch.no_grad():
            prototypes = torch.zeros(self.args.n_cls,self.args.feat_dim).cuda()
            for i, (input, target) in enumerate(self.loader):
                input, target = input.cuda(), target.cuda()
                features = self.model(input)
                for j, feature in enumerate(features):
                    prototypes[target[j].item()] += feature
                    prototype_counts[target[j].item()] += 1
            for cls in range(self.args.n_cls):
                prototypes[cls] /=  prototype_counts[cls] 
            # measure elapsed time
            duration = time.time() - start
            print(f'Time to initialize prototypes: {duration:.3f}')
            prototypes = F.normalize(prototypes, dim=1)
            self.prototypes = prototypes
