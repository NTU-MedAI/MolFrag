import os
import logging
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, f1_score, confusion_matrix


def eval_ddi_model(true_label, pred_score, threshold: float = 0.5, opt_thr: bool = True):
    """
    :return: AUROC, AUPRC, F1, Confusion Matrix, ACC, Threshold
    """
    auc_roc = roc_auc_score(true_label, pred_score)
    auc_prc = average_precision_score(true_label, pred_score)
    if opt_thr:
        fpr, tpr, thresholds = roc_curve(true_label, pred_score)
        idx = np.argmax(tpr - fpr)
        threshold = thresholds[idx]
    pred_label = []
    for i in pred_score:
        predict = 1 if i > threshold else 0
        pred_label.append(predict)
    f1 = f1_score(true_label, pred_label)
    mat = confusion_matrix(true_label, pred_label)
    acc = mat.diagonal().sum()/mat.sum()
    return auc_roc, auc_prc, f1, mat, acc, threshold


class ContrastiveLoss(nn.Module):
    def __init__(self, metric: str = 'CosineSimilarity', pos_margin: float = 0., neg_margin: float = 1.,
                 reduction: str = 'AvgNonZero',
                 return_distance: bool = False):
        super().__init__()

        if metric not in ['CosineSimilarity']:
            raise ValueError("Undefined metric!")
        if reduction not in ['AvgNonZero', 'mean', 'sum', 'none']:
            raise ValueError("Undefined reduction!")

        self.metric = metric
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.reduction = reduction
        self.ret_dis = return_distance

    def forward(self, embedding, label):
        if self.metric == 'CosineSimilarity':
            emb = F.normalize(embedding, dim=1)
            dis_mat = 1.0 - F.cosine_similarity(emb.unsqueeze(1), emb.unsqueeze(0), dim=2)

        loss = 0.5 * (
                label.float() * F.relu(dis_mat - self.pos_margin).pow(2) +
                (1.0 - label.float()) * F.relu(self.neg_margin - dis_mat).pow(2)
        )

        if self.reduction == 'AvgNonZero':
            non_zero = torch.sum(loss > 0)
            loss = loss.sum()/non_zero if non_zero > 0 else loss.mean()
        elif self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            pass

        if self.ret_dis:
            return dis_mat, loss
        else:
            return loss


class NTXent(nn.Module):
    def __init__(self, metric: str = 'CosineSimilarity', temperature: float = 0.5, reduction: str = 'mean'):
        super().__init__()

        if metric not in ['CosineSimilarity']:
            raise ValueError("Undefined metric!")
        if reduction not in ['AvgNonInf', 'mean', 'sum', 'none']:
            raise ValueError("Undefined reduction!")

        self.metric = metric
        self.t = temperature
        self.reduction = reduction

    def forward(self, embedding, label):
        if self.metric == 'CosineSimilarity':
            emb = F.normalize(embedding, dim=1)
            sim_mat = F.cosine_similarity(emb.unsqueeze(1), emb.unsqueeze(0), dim=2)
        sim_mat = torch.exp(sim_mat / self.t)
        pos = torch.sum(sim_mat * label, dim=1)
        inner = pos / torch.sum(sim_mat, dim=1)
        loss = -torch.log(inner)
        if self.reduction == 'AvgNonInf':
            non_inf = inner > 0
            loss = loss * non_inf
            loss = loss.sum() / torch.sum(non_inf) if torch.sum(non_inf) > 0 else loss.mean()
        elif self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            pass

        return loss


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_file_logger(file_name: str = 'log.txt', log_format: str = '%(message)s', log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    handler = logging.FileHandler(file_name)
    handler.setLevel(log_level)
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(log_level)

    logger.addHandler(handler)
    logger.addHandler(console)

    return logger