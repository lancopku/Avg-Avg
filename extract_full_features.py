import os
import torch
import argparse
import numpy as np
from loguru import logger

from lib.data_loader import get_data_loader
from lib.models.networks import get_model, get_tokenizer
from lib.inference.draw import get_full_features
from lib.exp import get_num_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='sst-2', type=str)
    parser.add_argument('--ood_datasets', default='20news',
                        type=str, required=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        required=False, help='batch size')
    parser.add_argument(
        '--model', default='roberta-base', help='pretrained model type')
    parser.add_argument('--pretrained_model', default=None,
                        type=str, required=False, help='the path of the model to load')
    parser.add_argument('--log_file', type=str, default='./log/default.log')
    parser.add_argument('--output_dir', default='saved_model/',
                        type=str, required=False, help='save directory')
    parser.add_argument('--feature_pos', type=str,
                        default='after', help='feature position')
    args = parser.parse_args()

    log_file_name = args.log_file
    logger.add(log_file_name)
    logger.info('args:\n' + args.__repr__())

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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

    train_loader = get_data_loader(
        args.dataset, 'train', tokenizer, args.batch_size)
    dev_loader = get_data_loader(
        args.dataset, 'dev', tokenizer, args.batch_size)
    test_loader = get_data_loader(
        args.dataset, 'test', tokenizer, args.batch_size)

    pooling = ['cls', 'avg']

    for p in pooling:
        logger.info('token pooling: {}'.format(p))
        train_features, train_labels, train_lens = get_full_features(
            model, train_loader, p, pos=args.feature_pos)
        dev_features, dev_labels, dev_lens = get_full_features(
            model, dev_loader, p, pos=args.feature_pos)
        test_features, test_labels, test_lens = get_full_features(
            model, test_loader, p, pos=args.feature_pos)

        np.save('{}/{}_ind_train_features.npy'.format(output_dir, p), train_features)
        np.save('{}/{}_ind_train_labels.npy'.format(output_dir, p), train_labels)
        np.save('{}/{}_ind_dev_features.npy'.format(output_dir, p), dev_features)
        np.save('{}/{}_ind_dev_labels.npy'.format(output_dir, p), dev_labels)
        np.save('{}/{}_ind_test_features.npy'.format(output_dir, p), test_features)
        np.save('{}/{}_ind_test_labels.npy'.format(output_dir, p), test_labels)

        for ood_dataset in args.ood_datasets.split(','):
            loader = get_data_loader(
                ood_dataset, 'test', tokenizer, args.batch_size)
            ood_features, ood_labels, ood_lens = get_full_features(
                model, loader, p, pos=args.feature_pos)
            np.save('{}/{}_ood_features_{}.npy'.format(output_dir,
                    p, ood_dataset), ood_features)


if __name__ == '__main__':
    main()
