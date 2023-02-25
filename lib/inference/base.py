import math
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from torch.autograd import Variable

def get_base_score(model, data_loader):
    '''
    Baseline, calculating Maxium Softmax Probability as prediction confidence
    Reference: A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks 
    Link: https://arxiv.org/abs/1610.02136
    '''
    model.eval()
    scores = []
    with torch.no_grad():
        for input_ids, labels, attention_masks in tqdm(data_loader):
            outputs = model(input_ids, attention_mask=attention_masks)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            soft_out = F.softmax(logits, dim=1)
            _scores, _ = torch.max(soft_out.data, dim=1)
            scores.append(_scores.cpu().numpy())

    scores = np.concatenate(scores)
    return scores

def get_grad_norm_score(model, data_loader, num_classes, temperature=1):
    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()
    model.eval()
    scores = []
    for input_ids, labels, attention_masks in tqdm(data_loader):
        model.zero_grad()
        #inputs = Variable(input_ids, requires_grad=True)
        outputs = model(input_ids, attention_mask=attention_masks)
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
        targets = torch.ones((input_ids.shape[0], num_classes)).cuda()
        logits = logits / temperature
        loss = torch.mean(torch.sum(-targets * logsoftmax(logits), dim=-1))

        loss.backward()

        layer_grad = model.classifier.out_proj.weight.grad.data
        #print(layer_grad.shape)
        layer_grad_norm = torch.sum(torch.abs(layer_grad)).cpu().numpy()
        #print(layer_grad_norm)
        scores.append(layer_grad_norm)
    #scores = np.concatenate(scores)
    scores = np.array(scores)
    return scores

def get_d2u_score(model, data_loader):
    model.eval()
    scores = []
    with torch.no_grad():
        for input_ids, labels, attention_masks in tqdm(data_loader):
            outputs = model(input_ids, attention_mask=attention_masks)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            soft_out = F.softmax(logits, dim=1).cpu().numpy()
            _scores = -np.log(soft_out)
            _scores = np.sum(_scores, axis=1)
            scores.append(_scores)

    scores = np.concatenate(scores)
    print(scores.shape)
    return scores

def get_energy_score(model, data_loader):
    model.eval()
    scores = []
    with torch.no_grad():
        for input_ids, labels, attention_masks in tqdm(data_loader):
            outputs = model(input_ids, attention_mask=attention_masks)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            if type(logits) == list:
                logits = logits[-1]
            logits = logits.cpu().numpy()
            for sample_logits in logits:
                exp_logits = [math.exp(logit) for logit in sample_logits]
                exp_logits_sum = sum(exp_logits)
                energy_score = math.log(exp_logits_sum)
                scores.append(energy_score)

    scores = np.array(scores)
    return scores
