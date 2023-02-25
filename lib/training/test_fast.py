import os
import torch
import argparse
import numpy as np
from copy import deepcopy
from loguru import logger
from torch.nn import DataParallel

from lib.data_loader import get_data_loader
from lib.training.common import test_acc
from lib.models.networks import get_model, get_tokenizer
from lib.inference.base import get_base_score
from lib.inference.godin import searchGeneralizedOdinParameters, get_ODIN_score
from lib.inference.lof import get_lof_score
from lib.inference.maha import sample_estimator, get_Mahalanobis_score
from lib.metrics import get_metrics
from lib.exp import get_num_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0',
                        type=str, required=False, help='GPU ids')
    parser.add_argument('--model_type', default='bert-base-uncased',
                        type=str, required=False, help='BERT model type')
    parser.add_argument('--dataset', default='20news', help='training dataset')
    parser.add_argument('--eval_type', default='ood',
                        type=str, choices=['acc', 'ood'])
    parser.add_argument('--ood_method', default='base', type=str,
                        choices=['base', 'deep-ensemble', 'maha', 'godin', 'lof'])
    parser.add_argument('--ood_dataset', default='sst-2',
                        type=str, required=False)
    parser.add_argument('--raw', action='store_true',
                        help='tokenize from start')
    parser.add_argument('--epochs', default=20, type=int,
                        required=False, help='number of training epochs')
    parser.add_argument('--batch_size', default=16, type=int,
                        required=False, help='batch size')
    parser.add_argument('--lm', action='store_true',
                        help='trained with mask LM task')
    parser.add_argument('--godin', action='store_true',
                        help='use Genearlized Odin Heads')
    parser.add_argument('--lmcl', action='store_true',
                    help='use LMCL')
    parser.add_argument('--warmup_steps', default=100,
                        type=int, required=False, help='warm up steps')
    parser.add_argument(
        '--model', default='bert-base-uncased', help='pretrained model type')
    parser.add_argument('--pretrained_model', default=None,
                        type=str, required=False, help='the path of the model to load')
    parser.add_argument('--pretrained_path', default=None,
                        type=str, help='model path for ensemble methods')
    parser.add_argument('--ensemble_num', default=5, type=int,
                        help='model num for ensemble methods')
    parser.add_argument('--log_file', type=str, default='./log/default.log')
    parser.add_argument('--save_dir', type=str, default='./log/scores')
    args = parser.parse_args()

    log_file_name = args.log_file
    logger.add(log_file_name)
    logger.info('args:\n' + args.__repr__())

    save_dir = args.save_dir    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    num_labels = get_num_labels(args.dataset)
    args.num_labels = num_labels
    model = get_model(args, fast=True)
    logger.info("{} model loaded".format(args.model))
    if args.pretrained_model:
        model.load_state_dict(torch.load(args.pretrained_model))
        logger.info("model loaded from {}".format(args.pretrained_model))
    tokenizer = get_tokenizer(args.model)
    logger.info("{} tokenizer loaded".format(args.model))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    

    ind_val_loader = get_data_loader(
        args.dataset, 'dev', tokenizer, args.batch_size)
    ind_test_loader = get_data_loader(
        args.dataset, 'test', tokenizer, args.batch_size)
    ood_test_loader = get_data_loader(
        args.ood_dataset, 'test', tokenizer, args.batch_size)

    model.eval()
    ind_logits = [[] for _ in range(model.config.num_hidden_layers)]
    ind_val_logits = [[] for _ in range(model.config.num_hidden_layers)]
    labels = []
    val_labels = []
    with torch.no_grad():
        for input_ids, batch_labels, attention_masks in ind_test_loader:
            logits_list = model(input_ids, attention_mask=attention_masks)
            for i in range(model.config.num_hidden_layers):
                ind_logits[i].append(logits_list[i].cpu().numpy())
            labels += list(batch_labels.reshape(-1).cpu().numpy())
        for input_ids, batch_labels, attention_masks in ind_test_loader:
            logits_list = model(input_ids, attention_mask=attention_masks)
            for i in range(model.config.num_hidden_layers):
                ind_val_logits[i].append(logits_list[i].cpu().numpy())
            val_labels += list(batch_labels.reshape(-1).cpu().numpy())
    
    for i in range(model.config.num_hidden_layers):
        ind_logits[i] = np.concatenate(ind_logits[i])
        ind_val_logits[i] = np.concatenate(ind_val_logits[i])
    
    ind_logits = np.array(ind_logits)
    ind_val_logits = np.array(ind_val_logits)
    labels = np.array(labels)
    val_labels = np.array(val_labels)

    ood_logits = [[] for _ in range(model.config.num_hidden_layers)]
    with torch.no_grad():
        for input_ids, batch_labels, attention_masks in ood_test_loader:
            logits_list = model(input_ids, attention_mask=attention_masks)
            for i in range(model.config.num_hidden_layers):
                ood_logits[i].append(logits_list[i].cpu().numpy())
    for i in range(model.config.num_hidden_layers):
        ood_logits[i] = np.concatenate(ood_logits[i])
    ood_logits = np.array(ood_logits)

    ind_train_loader = get_data_loader(args.dataset, 'train', tokenizer, args.batch_size)
    pooler_pos = 0
    sample_mean, precision = sample_estimator(model, num_labels, ind_train_loader, pooler_pos)
    ind_maha_scores = get_Mahalanobis_score(model, ind_test_loader, num_labels, sample_mean, precision,
                                                        magnitude=0, pooler_pos=pooler_pos, maha_ablation=True)
    ind_val_maha_scores = get_Mahalanobis_score(model, ind_val_loader, num_labels, sample_mean, precision,
                                                        magnitude=0, pooler_pos=pooler_pos, maha_ablation=True)
    ood_maha_scores = get_Mahalanobis_score(model, ood_test_loader, num_labels, sample_mean, precision,
                                                        magnitude=0, pooler_pos=pooler_pos, maha_ablation=True)

    np.save('{}/ind_logits.npy'.format(save_dir), ind_logits)
    np.save('{}/ind_val_logits.npy'.format(save_dir), ind_val_logits)
    np.save('{}/ood_logits.npy'.format(save_dir), ood_logits)
    np.save('{}/labels.npy'.format(save_dir), labels)
    np.save('{}/val_labels.npy'.format(save_dir), val_labels)
    np.save('{}/ind_maha_scores.npy'.format(save_dir), ind_maha_scores)
    np.save('{}/ind_val_maha_scores.npy'.format(save_dir), ind_val_maha_scores)
    np.save('{}/ood_maha_scores.npy'.format(save_dir), ood_maha_scores)


 
if __name__ == '__main__':
    main()
