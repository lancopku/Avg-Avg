import random
import torch
import numpy as np
from loguru import logger
from torch.nn import CrossEntropyLoss

def make_masked_data(tokenizer, input_ids):
    input_ids = input_ids.cpu().numpy()
    ssl_labels = []
    for i in range(input_ids.shape[0]): # batch size dim
        labels = []
        for j in range(input_ids.shape[1]): # sequence length dim
            token = input_ids[i][j]
            if token == tokenizer.pad_token_id or token == tokenizer.cls_token_id:
                labels.append(-100)
                continue
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                if prob<0.8:
                    input_ids[i][j] = tokenizer.mask_token_id
                elif prob<0.9:
                    input_ids[i][j] = random.randrange(len(tokenizer.vocab))
                labels.append(token)
            else:
                labels.append(-100)
        ssl_labels.append(labels)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    input_ids = torch.tensor(input_ids).long().to(device)
    ssl_labels = torch.tensor(ssl_labels).long().to(device)
    return input_ids, ssl_labels



def train_with_lm(model, optimizer, data_loader, tokenizer, epoch, alpha=1.0, log_steps=100, add_noise=False):
    model.train()
    running_loss_total = 0
    running_loss_cls = 0
    running_loss_ssl = 0
    step_count = 0

    for input_ids, labels, attention_masks in data_loader:

        loss_fct = CrossEntropyLoss()
        # Classification task
        cls_logits = model(
            input_ids, attention_mask=attention_masks, mode='cls')
        cls_loss = loss_fct(cls_logits, labels)
        running_loss_cls += cls_loss.item()

        # Masksed language modeling task

        ssl_input_ids, ssl_labels = make_masked_data(
            tokenizer, input_ids)
        ssl_logits = model(ssl_input_ids, attention_masks, mode='lm')
        ssl_loss = loss_fct(ssl_logits.permute(0,2,1), ssl_labels)
        running_loss_ssl += ssl_loss.item()

        loss = cls_loss + alpha*ssl_loss
        running_loss_total += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (step_count+1) % log_steps == 0:
            logger.info("step {}, total running loss = {}, cls running loss = {}, ssl running loss={}".format(
                step_count+1, running_loss_total, running_loss_cls, running_loss_ssl))
            running_loss_total = 0
            running_loss_cls = 0
            running_loss_ssl = 0 
        step_count += 1