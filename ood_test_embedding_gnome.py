import os
import torch
import argparse
import numpy as np
import sklearn.covariance
from loguru import logger
from lib.metrics import get_metrics
from torch.nn import CosineSimilarity
from sklearn.covariance import EmpiricalCovariance, MinCovDet

# inter-layer pooling
def pooling_features(features, pooling='last'):
    num_layers = features.shape[0]
    if pooling == 'last':
        return features[-1, :, :]
    elif pooling == 'avg':
        return np.mean(features[1:], axis=0)
    elif pooling == 'avg_emb':  # including token embeddings
        return np.mean(features, axis=0)
    elif pooling == 'emb':
        return features[0]
    elif pooling == 'first_last':
        return (features[-1] + features[1])/2.0
    elif pooling == 'odd':
        odd_layers = [1 + i for i in range(0, num_layers-1, 2)]
        return (np.sum(features[odd_layers], axis=0))/(num_layers/2)
    elif pooling == 'even':
        even_layers = [2 + i for i in range(0, num_layers-1, 2)]
        return (np.sum(features[even_layers], axis=0))/(num_layers/2)
    elif pooling == 'last2':
        return (features[-1] + features[-2])/2.0
    elif pooling == 'concat':
        # num_samples, layers, hidden_size
        features = np.transpose(features, (1, 0, 2))
        # num_samples, layers*hidden_size
        return features.reshape(features.shape[0], -1)
    elif type(pooling) == int or (type(pooling) == str and pooling.isdigit()):
        pooling = int(pooling)
        return features[pooling]
    elif ',' in pooling or type(pooling) == list:
        layers = pooling
        if type(pooling) == str:
            layers = list([int(l) for l in pooling.split(',')])
        return np.mean(features[layers], axis=0)
    else:
        raise NotImplementedError


def sample_estimator(features, labels):
    labels = labels.reshape(-1)
    num_classes = np.unique(labels).shape[0]
    #group_lasso = EmpiricalCovariance(assume_centered=False)
    #group_lasso =  MinCovDet(assume_centered=False, random_state=42, support_fraction=1.0)
    # ShurunkCovariance is more stable and robust where the condition number is large
    group_lasso = sklearn.covariance.ShrunkCovariance()
    sample_class_mean = []
    for c in range(num_classes):
        current_class_mean = np.mean(features[labels == c, :], axis=0)
        sample_class_mean.append(current_class_mean)
    X = [features[labels == c, :] - sample_class_mean[c] for c in range(num_classes)]
    X = np.concatenate(X, axis=0)
    group_lasso.fit(X)
    precision = group_lasso.precision_

    return sample_class_mean, precision


def get_distance_score(class_mean, precision, features, measure='maha'):
    num_classes = len(class_mean)
    num_samples = len(features)
    class_mean = [torch.from_numpy(m).float() for m in class_mean]
    precision = torch.from_numpy(precision).float()
    features = torch.from_numpy(features).float()
    scores = []
    for c in range(num_classes):
        centered_features = features.data - class_mean[c]
        if measure == 'maha':
            score = -1.0 * \
                torch.mm(torch.mm(centered_features, precision),
                    centered_features.t()).diag()
        elif measure == 'euclid':
            score = -1.0*torch.mm(centered_features,
                centered_features.t()).diag()
        elif measure == 'cosine':
            score = torch.tensor([CosineSimilarity()(features[i].reshape(
                1, -1), class_mean[c].reshape(1, -1)) for i in range(num_samples)])
        else:
            raise ValueError("Unknown distance measure")
        scores.append(score.reshape(-1, 1))
    scores = torch.cat(scores, dim=1)  # num_samples, num_classes
    scores, _ = torch.max(scores, dim=1)  # num_samples
    scores = scores.cpu().numpy()
    return scores


def get_auroc(ind_train_features, ind_train_labels, ind_test_features, ood_test_features, layer_pooling):
    ind_train_features = pooling_features(ind_train_features, layer_pooling)
    sample_class_mean, precision = sample_estimator(
        ind_train_features, ind_train_labels)
    ind_test_features = pooling_features(ind_test_features, layer_pooling)
    ood_test_features = pooling_features(ood_test_features, layer_pooling)
    ind_scores = get_distance_score(
        sample_class_mean, precision, ind_test_features)
    ood_scores = get_distance_score(
        sample_class_mean, precision, ood_test_features)
    metrics = get_metrics(ind_scores, ood_scores)
    auroc = metrics['AUROC']
    return auroc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='sst-2', help='training dataset')
    parser.add_argument('--ood_datasets', default='20news,wmt16,multi30k,rte,snli',
                        type=str, required=False)
    parser.add_argument('--distance_metric', type=str, default='maha',
                        help='distance metric')
    parser.add_argument('--ft_token_pooling', type=str, default='cls',
                        help='token pooling way', choices = ['cls','avg', 'max'])
    parser.add_argument('--ft_layer_pooling', type=str, default='last')
    parser.add_argument('--pre_token_pooling', type=str, default='cls',
                        help='token pooling way', choices = ['cls','avg', 'max'])
    parser.add_argument('--pre_layer_pooling', type=str, default='last')
    parser.add_argument('--pre_dir', default='./log/embeddings/roberta-base/sst-2/seed13',
                        type=str, required=False, help='pretrained features path')
    parser.add_argument('--ft_dir', default='./log/embeddings/roberta-base/sst-2/seed13',
                        type=str, required=False, help='fine-tuned features path')
    parser.add_argument('--log_file', type=str, default='./log/default.log')
    parser.add_argument('--std', action='store_true',
                        help='standrize the scores')
    parser.add_argument('--min_max_norm', action='store_true',
                        help='Min-Max normalization')
    parser.add_argument('--ensemble_way', type=str,
                        choices=['min', 'mean', 'svm'], default='mean')
    args = parser.parse_args()

    log_file_name = args.log_file
    logger.add(log_file_name)
    logger.info('args:\n' + args.__repr__())

    pre_dir = args.pre_dir
    ft_dir = args.ft_dir
    ft_token_pooling = args.ft_token_pooling
    ft_layer_pooling = args.ft_layer_pooling
    pre_token_pooling = args.pre_token_pooling
    pre_layer_pooling = args.pre_layer_pooling

    ind_train_features = np.load(
        '{}/{}_ind_train_features.npy'.format(ft_dir, ft_token_pooling))
    ind_valid_features = np.load(
        '{}/{}_ind_dev_features.npy'.format(ft_dir, ft_token_pooling))
    ind_train_labels = np.load(
        '{}/{}_ind_train_labels.npy'.format(ft_dir, ft_token_pooling))

    ind_train_features = pooling_features(ind_train_features, ft_layer_pooling)
    ind_valid_features = pooling_features(ind_valid_features, ft_layer_pooling)
    sample_class_mean, precision = sample_estimator(
        ind_train_features, ind_train_labels)

    ind_test_features = np.load(
        '{}/{}_ind_test_features.npy'.format(ft_dir, ft_token_pooling))
    ind_test_features = pooling_features(ind_test_features, ft_layer_pooling)
    ft_ind_test_scores = get_distance_score(
        sample_class_mean, precision, ind_test_features, args.distance_metric)
    ft_ind_valid_scores = get_distance_score(
        sample_class_mean, precision, ind_valid_features, args.distance_metric)
    ft_ood_scores_list = []
    for ood_dataset in args.ood_datasets.split(','):
        ood_features = np.load(
            '{}/{}_ood_features_{}.npy'.format(ft_dir, ft_token_pooling, ood_dataset))
        ood_features = pooling_features(ood_features, ft_layer_pooling)
        ood_scores = get_distance_score(
            sample_class_mean, precision, ood_features, args.distance_metric)
        ft_ood_scores_list.append(ood_scores)

    ind_train_features = np.load(
        '{}/{}_ind_train_features.npy'.format(pre_dir, pre_token_pooling))
    ind_valid_features = np.load(
        '{}/{}_ind_dev_features.npy'.format(pre_dir, pre_token_pooling))
    ind_train_labels = np.load(
        '{}/{}_ind_train_labels.npy'.format(pre_dir, pre_token_pooling))

    ind_train_features = pooling_features(ind_train_features, pre_layer_pooling)
    ind_valid_features = pooling_features(ind_valid_features, pre_layer_pooling)
    sample_class_mean, precision = sample_estimator(
        ind_train_features, ind_train_labels)

    ind_test_features = np.load(
        '{}/{}_ind_test_features.npy'.format(pre_dir, pre_token_pooling))
    ind_test_features = pooling_features(ind_test_features, pre_layer_pooling)
    pre_ind_test_scores = get_distance_score(
        sample_class_mean, precision, ind_test_features, args.distance_metric)
    pre_ind_valid_scores = get_distance_score(
        sample_class_mean, precision, ind_valid_features, args.distance_metric)
    pre_ood_scores_list = []
    for ood_dataset in args.ood_datasets.split(','):
        ood_features = np.load(
            '{}/{}_ood_features_{}.npy'.format(pre_dir, pre_token_pooling, ood_dataset))
        ood_features = pooling_features(ood_features, pre_layer_pooling)
        ood_scores = get_distance_score(
            sample_class_mean, precision, ood_features, args.distance_metric)
        pre_ood_scores_list.append(ood_scores)

    if args.std:
        if args.min_max_norm:
            ft_min = np.min(ft_ind_valid_scores)
            ft_max = np.max(ft_ind_valid_scores)
            ft_ind_test_scores = (ft_ind_test_scores-ft_min)/(ft_max-ft_min)
            for i in range(len(ft_ood_scores_list)):
                ft_ood_scores_list[i] = (
                    ft_ood_scores_list[i]-ft_min)/(ft_max-ft_min)

            pre_min = np.min(pre_ind_valid_scores)
            pre_max = np.max(pre_ind_valid_scores)
            pre_ind_test_scores = (
                pre_ind_test_scores-pre_min)/(pre_max-pre_min)
            for i in range(len(pre_ood_scores_list)):
                pre_ood_scores_list[i] = (
                    pre_ood_scores_list[i]-pre_min)/(pre_max-pre_min)
        else:
            ft_mean = np.mean(ft_ind_valid_scores)
            ft_std = np.std(ft_ind_valid_scores)
            ft_ind_test_scores = (ft_ind_test_scores-ft_mean)/ft_std
            for i in range(len(ft_ood_scores_list)):
                ft_ood_scores_list[i] = (ft_ood_scores_list[i]-ft_mean)/ft_std

            pre_mean = np.mean(pre_ind_valid_scores)
            pre_std = np.std(pre_ind_valid_scores)
            pre_ind_test_scores = (pre_ind_test_scores-pre_mean)/pre_std
            for i in range(len(pre_ood_scores_list)):
                pre_ood_scores_list[i] = (
                    pre_ood_scores_list[i]-pre_mean)/pre_std

    ood_test_scores_list = []
    if args.ensemble_way == 'min':
        ind_test_scores = np.minimum(ft_ind_test_scores, pre_ind_test_scores)
        for i in range(len(ft_ood_scores_list)):
            ood_test_scores = np.minimum(
                ft_ood_scores_list[i], pre_ood_scores_list[i])
            ood_test_scores_list.append(ood_test_scores)
    elif args.ensemble_way == 'mean':
        ind_test_scores = (ft_ind_test_scores + pre_ind_test_scores)/2
        for i in range(len(ft_ood_scores_list)):
            ood_test_scores = (ft_ood_scores_list[i]+pre_ood_scores_list[i])/2
            ood_test_scores_list.append(ood_test_scores)
    else:  # svm
        from sklearn.svm import OneClassSVM
        ind_valid_data = np.stack(
            (ft_ind_valid_scores, pre_ind_valid_scores), axis=1)
        ind_test_data = np.stack(
            (ft_ind_test_scores, pre_ind_test_scores), axis=1)
        ood_test_data_list = []
        for i in range(len(ft_ood_scores_list)):
            ood_test_data_list.append(
                np.stack((ft_ood_scores_list[i], pre_ood_scores_list[i]), axis=1))

        clf = OneClassSVM(gamma='auto').fit(ind_valid_data)
        ind_test_scores = clf.score_samples(ind_test_data)
        for ood_test_data in ood_test_data_list:
            ood_test_scores_list.append(clf.score_samples(ood_test_data))

    ood_metrics_list = []
    ood_datasets = args.ood_datasets.split(',')
    for i in range(len(ood_datasets)):
        metrics = get_metrics(ind_test_scores, ood_test_scores_list[i])
        logger.info('ood dataset: {}'.format(ood_datasets[i]))
        logger.info('metrics: {}'.format(metrics))
        ood_metrics_list.append(metrics)
    mean_metrics = {}
    for k, v in metrics.items():
        mean_metrics[k] = sum([m[k] for m in ood_metrics_list])/len(ood_metrics_list)
    logger.info('mean metrics: {}'.format(mean_metrics))


if __name__ == '__main__':
    main()
