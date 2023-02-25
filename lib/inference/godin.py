import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from loguru import logger
from copy import deepcopy

import torch.nn.functional as F

def get_ODIN_score(model, dataloader, magnitude, temperature, pooler_pos=0):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    scores = []
    for input_ids, labels, attention_masks in tqdm(dataloader):
        model.eval()
        try:
            outputs = model.bert(input_ids, attention_masks, return_dict=True, output_hidden_states=True)
        except:
            try:
                outputs = model.albert(input_ids, attention_masks, return_dict=True, output_hidden_states=True)
            except:
                try:
                    outputs = model.roberta(input_ids, attention_masks, return_dict=True, output_hidden_states=True)
                except:
                    outputs = model.transformer(input_ids, attention_mask=attention_masks, return_dict=True, output_hidden_states=True) # gpt-2
        hidden_states = outputs.hidden_states
        features = hidden_states[-1][:,pooler_pos,:] 
        if hasattr(model,'dropout'):
            features = model.dropout(features)
        model.train()
        if magnitude >= 0:
            #features.requires_grad = True
            #print(features.requires_grad)
            features.grad = None
            model.zero_grad()
            try:
                logits = model.get_logits_from_poooled_output(features)
            except:
                if hasattr(model,'classifier'):
                    try:
                        logits = model.classifier(features)
                    except:
                        logits = model.classifier(features.reshape(features.shape[0],1,features.shape[1]))
                else: #XLNET
                    logits = model.logits_proj(features)
            scaling_logits = logits / temperature
            labels = scaling_logits.data.max(1)[1]

            loss = criterion(scaling_logits, labels)
            features.retain_grad()
            loss.backward()
            # Normalizing the gradient to binary in {-1, 1}
            gradient =  torch.ge(features.grad.data, 0) # 0 or 1
            gradient = (gradient.float() - 0.5) * 2 # -1 or 1
        model.eval()
        with torch.no_grad():
            if magnitude >= 0:
                features_p = torch.add(features.data, -magnitude, gradient)
            else:
                features_p = features
            try:
                logits = model.get_logits_from_poooled_output(features_p)
            except:
                if hasattr(model,'classifier'):
                    try:
                        logits = model.classifier(features)
                    except:
                        logits = model.classifier(features.reshape(features.shape[0],1,features.shape[1]))
                else: #XLNET
                    logits = model.logits_proj(features_p)
            scaling_logits = logits / temperature
            logits = logits / temperature
            soft_out = F.softmax(logits, dim=1)
            _scores, _ = torch.max(soft_out.data, dim=1)
            #print(_scores)
            scores.append(_scores.cpu().numpy())
            del outputs, features, features_p, hidden_states, loss, logits, scaling_logits, soft_out, _scores
    scores = np.concatenate(scores)
    return scores

def searchGeneralizedOdinParameters(model,ind_data_loader,pooler_pos=0):
    model.eval()
    #magnitude_list = [0,0.0001,0.0002,0.0004,0.0008,0.0016,0.0025,0.005, 0.01, 0.02, 0.04, 0.08]
    magnitude_list = [0]
    temperature_list = [1000] # fixed to 1000
    best_magnitude = None
    best_temperature = None
    highest_mean_score = -1e10
    for m in magnitude_list:
        for t in temperature_list:
            #data_loader = deepcopy(ind_data_loader)
            ind_scores = get_ODIN_score(model, ind_data_loader, m, t, pooler_pos=pooler_pos)
            mean_score = ind_scores.mean()
            if mean_score > highest_mean_score:
                highest_mean_score = mean_score
                best_magnitude = m
                best_temperature = t
    logger.info("best temperature is {}, best magnitude is {}".format(best_temperature, best_magnitude))
    return best_magnitude,best_temperature
            