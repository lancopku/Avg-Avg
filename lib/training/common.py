import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from loguru import logger
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score, f1_score

def test_acc(model, data_loader, metric='acc'):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for input_ids, batch_labels, attention_masks in tqdm(data_loader):
            outputs = model(input_ids, attention_mask=attention_masks)
            if hasattr(outputs,'logits'):
                logits = outputs.logits
            elif type(outputs) == list:
                logits = outputs[-1]
            else:
                logits = outputs
            bacth_preds = np.array(torch.argmax(logits,dim=1).cpu())
            preds += list(bacth_preds)
            labels += list(batch_labels.reshape(-1).cpu().numpy())
    if metric == 'acc':
        acc = accuracy_score(labels, preds)
    elif metric == 'f1':
        acc =  f1_score(labels, preds, average='macro')
    return float(acc) 


def marginLoss(pooled,labels):
    dist = ((pooled.unsqueeze(1) - pooled.unsqueeze(0)) ** 2).mean(-1)
    mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    mask = mask - torch.diag(torch.diag(mask))
    neg_mask = (labels.unsqueeze(1) != labels.unsqueeze(0)).float()
    max_dist = (dist * mask).max()
    cos_loss = (dist * mask).sum(-1) / (mask.sum(-1) + 1e-3) + (F.relu(max_dist - dist) * neg_mask).sum(-1) / (neg_mask.sum(-1) + 1e-3)
    cos_loss = cos_loss.mean()
    return cos_loss

def sclLoss(pooled, labels):
    norm_pooled = F.normalize(pooled, dim=-1)
    cosine_score = torch.exp(norm_pooled @ norm_pooled.t() / 0.3)
    mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    cosine_score = cosine_score - torch.diag(torch.diag(cosine_score))
    mask = mask - torch.diag(torch.diag(mask))
    cos_loss = cosine_score / cosine_score.sum(dim=-1, keepdim=True)
    cos_loss = -torch.log(cos_loss + 1e-5)
    cos_loss = (mask * cos_loss).sum(-1) / (mask.sum(-1) + 1e-3)
    cos_loss = cos_loss.mean()
    return cos_loss


def train_common(model,optimizer,data_loader, epoch, log_steps=100,  pre_model=None, shift_reg=0, scl_reg=2.0, loss_type='mse', 
    save_steps=-1, save_dir=None, step_counter=0, max_grad_norm=1.0):
    model.train()
    if pre_model is not None:
        pre_model.eval()
    running_loss = 0
    step_count = 0
    for input_ids, labels, attention_masks in data_loader:
        if hasattr(model, 'centers') or ( hasattr(model,'module') and hasattr(model.module, 'centers')): #LMCL loss version
            outputs = model(input_ids, attention_mask=attention_masks, label=labels)
        else:
            outputs = model(input_ids, attention_mask=attention_masks, output_hidden_states=True)
        if hasattr(outputs,'logits'):
            logits = outputs.logits
        else:
            logits = outputs
        
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits, labels)

        if shift_reg > 0 and pre_model is not None:     
            dist = torch.sum(torch.abs(torch.cat(
                [p.view(-1) for n, p in model.named_parameters() if 'bert' in n.lower()]) - torch.cat(
                [p.view(-1) for n, p in pre_model.named_parameters() if
                    'bert' in n.lower()])) ** 2).item()
            loss += shift_reg*dist
        if  loss_type == 'scl':
            loss += scl_reg*sclLoss(outputs.hidden_states[-1][:,0,:], labels)
        elif loss_type == 'margin':
            loss += scl_reg*marginLoss(outputs.hidden_states[-1][:,0,:], labels)
        running_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        if (step_count+1) % log_steps == 0:
            #print(labels)
            logger.info("step {}, running loss = {}".format(step_count+1, running_loss))
            running_loss = 0
        step_count += 1
        step_counter += 1
        
        if save_steps != -1 and save_dir is not None:
            if step_counter % save_steps == 0:
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save.state_dict(),'{}/step{}_model.pt'.format(save_dir, step_counter))
                logger.info("model saved to {}/step{}_model.pt".format(save_dir, step_counter))
    return step_counter

    