import heapq
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from functools import reduce
from math import log
from tqdm import tqdm

'''
#TODO: comments in detail
'''
def get_ppl_confidence(model, tokenizer, data_loader):
    model.eval()
    scores = []
    with torch.no_grad():
        for input_ids, labels, attention_masks in tqdm(data_loader):

            B = input_ids.shape[0]
            K = input_ids.shape[1]
            ppl_scores_list = [[] for _ in range(B)]
            for masked_pos in range(0,K):
                x_masked = input_ids.clone()
                x_masked[:,masked_pos] = tokenizer.mask_token_id
                ssl_logits = model(x_masked, attention_masks, mode='lm')
                ssl_softmax_probs = nn.Softmax(dim=2)(ssl_logits)
                ssl_probs = ssl_softmax_probs[:,masked_pos,:] # (B,1,vocab_size)
                ssl_probs = ssl_probs.reshape(ssl_probs.shape[0],-1) # (B,vocab_size)
                for i in range(B):
                    token = input_ids[i][masked_pos]
                    if token != tokenizer.mask_token_id and token != tokenizer.cls_token_id:
                        ppl_scores_list[i].append(log(ssl_probs[i][input_ids[i][masked_pos]].item())) 
            ppl_scores = [ sum(l)/len(l) for l in ppl_scores_list ]
            scores.append(ppl_scores)
    scores = np.concatenate(scores)
    return scores

def get_key_positions(model, tokenizer, input_ids, attention_masks, topn=2):
    outputs = model.bert(input_ids, attention_masks, return_dict=True, output_attentions=True)
    attentions = outputs.attentions[-1]
    attentions = attentions.sum(dim=1)  # sum over attention heads (batch_size, max_len, max_len)
    attention_scores = [[] for i in range(attentions.size(0))]
    for i in range(attentions.size(0)):  # batch_size
        for j in range(attentions.size(-1)):  # max_len
            token = input_ids[i][j]
            if token == tokenizer.pad_token_id: # token == pad_token
                break
            score = attentions[i][0][j]  # 1st token = CLS token
            attention_scores[i].append(score)
    
    key_words = [[] for i in range(attentions.size(0))]
    for i in range(attentions.size(0)):
        key_words[i] = heapq.nlargest(topn, range(len(attention_scores[i])), attention_scores[i].__getitem__)

    return np.array(key_words)
           

def get_keyword_confidence(tokenizer, generator, classifier, data_loader, topn=2):
    if topn == -1:
        return get_ppl_confidence(generator, tokenizer, data_loader)
    generator.eval()
    classifier.eval()
    scores = []
    with torch.no_grad():
        for input_ids, labels, attention_masks in tqdm(data_loader):

            B = input_ids.shape[0]
            K = input_ids.shape[1]
            ppl_scores_list = [[] for _ in range(B)]

            key_positions = get_key_positions(classifier, tokenizer, input_ids, attention_masks, topn)

            for masked_pos in range(0,topn):
                x_masked = input_ids.clone()
                for i in range(B):
                    x_masked[i][key_positions[i][masked_pos]] = tokenizer.mask_token_id
                ssl_logits = generator(x_masked, attention_masks, mode='lm')
                ssl_softmax_probs = nn.Softmax(dim=2)(ssl_logits)
                ssl_probs = ssl_softmax_probs[:,masked_pos,:] # (B,1,vocab_size)
                ssl_probs = ssl_probs.reshape(ssl_probs.shape[0],-1) # (B,vocab_size)
                for i in range(B):
                    token = input_ids[i][masked_pos]
                    if token != tokenizer.mask_token_id and token != tokenizer.cls_token_id:
                        ppl_scores_list[i].append(log(ssl_probs[i][input_ids[i][masked_pos]].item())) 
            ppl_scores = [ sum(l)/len(l) for l in ppl_scores_list ]
            scores.append(ppl_scores)
    scores = np.concatenate(scores)
    return scores