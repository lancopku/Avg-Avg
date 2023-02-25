import math
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

def get_msp_score(model, data_loader):
    '''
    Baseline, calculating Maxium Softmax Probability as prediction confidence
    Reference: A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks 
    Link: https://arxiv.org/abs/1610.02136
    '''
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


def get_dropout_score(model, data_loader, passes=5):

    model.train()
    scores_list = []
    for _ in range(passes):
        scores = get_msp_score(model, data_loader)
        scores_list.append(scores)
    scores = np.mean(np.array(scores_list), axis=0)

    return scores