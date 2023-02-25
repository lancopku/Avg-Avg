import os
import sys
import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm
from loguru import logger

# OOD metrics code, adapted from https://github.com/megvii-research/FSSD_OoD_Detection/blob/master/lib/metric/__init__.py

def get_is_pos(ind_scores, ood_scores, order):
    assert order in ["largest2smallest", "smallest2largest"]
    scores = np.concatenate((ind_scores, ood_scores))
    is_pos = np.concatenate((np.ones(len(ind_scores), dtype="bool"), np.zeros(len(ood_scores), dtype="bool")))
    
    # shuffle before sort
    random_idx = np.random.permutation(list(range(len(scores))))
    scores = scores[random_idx]
    is_pos = is_pos[random_idx]

    idxs = scores.argsort()
    if order == "largest2smallest":
        idxs = np.flip(idxs)
    is_pos = is_pos[idxs]
    return is_pos

def roc(ind_scores, ood_scores, order):
    assert order in ["largest2smallest", "smallest2largest"]
    is_pos = get_is_pos(ind_scores, ood_scores, order)
    TP = 0
    FP = 0
    P = len(ind_scores)
    N = len(ood_scores)
    roc_curve = [[0, 0]]
    for _is_pos in is_pos:
        if _is_pos:
            TP += 1
        else:
            FP += 1
        recall = TP / P
        FPR = FP / N
        roc_curve.append([FPR, recall])
    return roc_curve    


def auroc(ind_scores, ood_scores, order):
    assert order in ["largest2smallest", "smallest2largest"]
    roc_curve = roc(ind_scores, ood_scores, order)
    roc_curve = np.array(roc_curve)
    x = roc_curve[:, 0]
    y = roc_curve[:, 1]
    x1 = x[:-1]
    x2 = x[1:]
    y1 = y[:-1]
    y2 = y[1:]
    auc = sum((x2 - x1) * (y1 + y2) / 2)
    return auc

def fpr_at_tpr(ind_scores, ood_scores, order, tpr = 0.95):
    assert order in ["largest2smallest", "smallest2largest"]
    is_pos = get_is_pos(ind_scores, ood_scores, order)
    P = len(ind_scores)
    N = len(ood_scores)
    TP = 0
    FP = 0
    for _is_pos in is_pos:
        if _is_pos:
            TP += 1
        else:
            FP += 1
        TPR = TP / P
        if TPR >= tpr:
            FPR = FP / N
            return FPR

def tnr_at_tpr(ind_scores, ood_scores, order, tpr = 0.95):
    assert order in ["largest2smallest", "smallest2largest"]
    is_pos = get_is_pos(ind_scores, ood_scores, order)
    P = len(ind_scores)
    N = len(ood_scores)
    TP = 0
    TN = N
    for _is_pos in is_pos:
        if _is_pos:
            TP += 1
        else:
            TN -= 1
        TPR = TP / P
        if TPR >= tpr:
            TNR = TN / N
            return TNR

def auin(ind_scores, ood_scores, order):
    assert order in ["largest2smallest", "smallest2largest"]
    is_pos = get_is_pos(ind_scores, ood_scores, order)
    P = len(ind_scores)
    N = len(ood_scores)
    TP = 0
    FP = 0
    recall_prec = []
    for _is_pos in is_pos:
        if _is_pos:
            TP += 1
        else:
            FP += 1
        prec = TP / (TP + FP)
        recall = TP / P
        recall_prec.append([recall, prec])
    recall_prec = np.array(recall_prec)
    x = recall_prec[:,0]
    y = recall_prec[:,1]
    x1 = x[:-1]
    x2 = x[1:]
    y1 = y[:-1]
    y2 = y[1:]
    auin = sum((x2 - x1) * (y1 + y2) / 2)
    return auin
    

def auout(ind_scores, ood_scores, order):
    assert order in ["largest2smallest", "smallest2largest"]
    is_pos = get_is_pos(ind_scores, ood_scores, order)
    is_pos = ~np.flip(is_pos)
    N = len(ind_scores)
    P = len(ood_scores)
    TP = 0
    FP = 0
    recall_prec = []
    for _is_pos in is_pos:
        if _is_pos:
            TP += 1
        else:
            FP += 1
        prec = TP / (TP + FP)
        recall = TP / P
        recall_prec.append([recall, prec])
    recall_prec = np.array(recall_prec)
    x = recall_prec[:,0]
    y = recall_prec[:,1]
    x1 = x[:-1]
    x2 = x[1:]
    y1 = y[:-1]
    y2 = y[1:]
    auout = sum((x2 - x1) * (y1 + y2) / 2)
    return auout

def best_acc(ind_scores, ood_scores, order):
    assert order in ["largest2smallest", "smallest2largest"]
    is_pos = get_is_pos(ind_scores, ood_scores, order)
    P = len(ind_scores)
    N = len(ood_scores)
    TP = 0
    TN = N
    accuracy = 0
    for _is_pos in is_pos:
        if _is_pos:
            TP += 1
        else:
            TN -= 1
        # _acc = (TP+TN) / (P + N)
        _acc = (TP/P + TN/N) / 2
        accuracy = max(accuracy, _acc)
    return accuracy

def get_metrics(ind_scores, ood_scores):

    logger.info("mean ind_scores: {}".format(ind_scores.mean()))
    logger.info("mean ood_scores: {}".format(ood_scores.mean()))
    
    order = "largest2smallest"  # sort score by largest2smallest
    metrics = {}
    metrics['AUROC'] = auroc(ind_scores, ood_scores, order)
    metrics['AUIN'] = auin(ind_scores, ood_scores, order)
    metrics['AUOUT'] = auout(ind_scores, ood_scores, order)
    #metrics['TNR@tpr=0.99'] = tnr_at_tpr(ind_scores, ood_scores, order, tpr=0.99)
    #metrics['FPR@tpr=0.99'] = fpr_at_tpr(ind_scores, ood_scores, order, tpr=0.99)
    metrics['TNR@tpr=0.95'] = tnr_at_tpr(ind_scores, ood_scores, order, tpr=0.95)
    metrics['FPR@tpr=0.95'] = fpr_at_tpr(ind_scores, ood_scores, order, tpr=0.95)
    metrics['TNR@tpr=0.8'] = tnr_at_tpr(ind_scores, ood_scores, order, tpr=0.8)
    metrics['FPR@tpr=0.8'] = fpr_at_tpr(ind_scores, ood_scores, order, tpr=0.8)
    return metrics



class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

def get_temperature(logits, labels):
    logits = torch.tensor(logits).cuda()
    labels = torch.tensor(labels).cuda()
    temperature = torch.ones(1) * 1
    temperature = temperature.cuda()
    temperature.requires_grad = True
    nll_criterion = nn.CrossEntropyLoss().cuda()
    ece_criterion = _ECELoss().cuda()
    before_temperature_nll = nll_criterion(logits, labels).item()
    before_temperature_ece = ece_criterion(logits, labels).item()
    print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))
    def temperature_scale(tmp_logits):
        t = temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / t

    optimizer = optim.LBFGS([temperature], lr=0.0001, max_iter=1000)
    def eval():
        loss = nll_criterion(temperature_scale(logits), labels)
        loss.backward()
        return loss
    optimizer.step(eval)

    after_temperature_nll = nll_criterion(temperature_scale(logits), labels).item()
    after_temperature_ece = ece_criterion(temperature_scale(logits), labels).item()
    print('Optimal temperature: %.3f' % temperature.item())
    print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

    return temperature.item()
