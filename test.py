import os
import torch
import argparse
import numpy as np
from loguru import logger

from lib.data_loader import get_data_loader
from lib.training.common import test_acc
from lib.models.networks import get_model, get_tokenizer
from lib.inference.base import get_base_score, get_energy_score, get_d2u_score
from lib.inference.godin import searchGeneralizedOdinParameters, get_ODIN_score
from lib.inference.lof import get_lof_score
from lib.inference.dropout import get_dropout_score
from lib.metrics import get_metrics
from lib.exp import get_num_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='sst-2', help='training dataset')
    parser.add_argument('--eval_type', default='ood',
                        type=str, choices=['acc', 'ood'])
    parser.add_argument('--ood_method', default='base', type=str)
    parser.add_argument('--ood_datasets', default='20news',
                        type=str, required=False)
    parser.add_argument('--batch_size', default=32, type=int,
                        required=False, help='batch size')
    parser.add_argument('--model', default='roberta-base',
                        help='pretrained model type')
    parser.add_argument('--pretrained_model', default=None,
                        type=str, required=False, help='the path of the checkpoint to load')
    parser.add_argument('--log_file', type=str, default='./log/default.log')
    parser.add_argument('--score_save_path',
                        default='./log/scores/msp/sst-2/seed13')
    parser.add_argument('--save_score', action='store_true')
    parser.add_argument('--passes', type=int, default=5,
                        help='number of passes in MC-Dropout')

    args = parser.parse_args()

    log_file_name = args.log_file
    logger.add(log_file_name)
    logger.info('args:\n' + args.__repr__())
    if args.save_score:
        if not os.path.exists(args.score_save_path):
            os.makedirs(args.score_save_path)

    num_labels = get_num_labels(args.dataset)
    args.num_labels = num_labels
    model = get_model(args)
    logger.info("{} model loaded".format(args.model))
    if args.pretrained_model:
        model.load_state_dict(torch.load(args.pretrained_model))
        logger.info("model loaded from {}".format(args.pretrained_model))
    tokenizer = get_tokenizer(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("{} tokenizer loaded".format(args.model))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    pooler_pos = 0
    if 'gpt' in args.model or 'xlnet' in args.model:
        pooler_pos = -1

    ood_datasets = args.ood_datasets.split(',')
    ind_test_loader = get_data_loader(
        args.dataset, 'test', tokenizer, args.batch_size)
    ood_test_loaders = [get_data_loader(
        ood_dataset, 'test', tokenizer, args.batch_size) for ood_dataset in ood_datasets]

    if args.eval_type == 'acc':  # Classification Validation
        acc = test_acc(model, ind_test_loader)
        logger.info("Test Accuracy on {} test set: {:.4f}".format(
            args.dataset, acc))
    else:  # OOD Detection Validation
        ood_scores_list = []
        ood_metrics_list = []
        if args.ood_method == 'base':
            ind_scores = get_base_score(model, ind_test_loader)
            for ood_test_loader in ood_test_loaders:
                ood_scores = get_base_score(model, ood_test_loader)
                ood_scores_list.append(ood_scores)
        elif args.ood_method == 'mc':
            ind_scores = get_dropout_score(model, ind_test_loader, args.passes)
            for ood_test_loader in ood_test_loaders:
                ood_scores = get_dropout_score(
                    model, ood_test_loader, args.passes)
                ood_scores_list.append(ood_scores)
        elif args.ood_method == 'energy':
            ind_scores = get_energy_score(model, ind_test_loader)
            for ood_test_loader in ood_test_loaders:
                ood_scores = get_energy_score(model, ood_test_loader)
                ood_scores_list.append(ood_scores)
        elif args.ood_method == 'd2u':
            ind_scores = get_d2u_score(model, ind_test_loader)
            for ood_test_loader in ood_test_loaders:
                ood_scores = get_d2u_score(model, ood_test_loader)
                ood_scores_list.append(ood_scores)
        elif args.ood_method == 'odin':
            ind_dev_loader = get_data_loader(
                args.dataset, 'dev', tokenizer, args.batch_size)
            best_magnitude, best_temperature = searchGeneralizedOdinParameters(
                model, ind_dev_loader, pooler_pos=pooler_pos)
            ind_scores = get_ODIN_score(
                model, ind_test_loader, best_magnitude, best_temperature, pooler_pos=pooler_pos)
            for ood_test_loader in ood_test_loaders:
                ood_scores = get_ODIN_score(
                    model, ood_test_loader, best_magnitude, best_temperature, pooler_pos=pooler_pos)
                ood_scores_list.append(ood_scores)
        elif args.ood_method == 'lof':
            ind_train_loader = get_data_loader(
                args.dataset, 'train', tokenizer, args.batch_size)
            ind_scores, ood_scores_list = get_lof_score(
                model, ind_train_loader, ind_test_loader, ood_test_loaders, pooler_pos=pooler_pos)
        else:
            raise NotImplementedError

        for i, ood_dataset in enumerate(ood_datasets):
            logger.info("OOD: {}".format(ood_dataset))
            metrics = get_metrics(ind_scores, ood_scores_list[i])
            if args.save_score:
                np.save('{}/ood_scores_{}.npy'.format(args.score_save_path,
                        ood_dataset), ood_scores_list[i])
            ood_metrics_list.append(metrics)
            logger.info(str(metrics))

        if args.save_score:
            np.save('{}/ind_scores.npy'.format(args.score_save_path), ind_scores)
        mean_metrics = {}
        for k, v in metrics.items():
            mean_metrics[k] = sum(
                [m[k] for m in ood_metrics_list])/len(ood_metrics_list)
        logger.info('mean metrics: {}'.format(mean_metrics))

    logger.info('evaluation finished')


if __name__ == '__main__':
    main()
