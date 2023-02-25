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
    parser.add_argument('--token_pooling', type=str, default='avg',
                        help='token pooling way', choices=['cls', 'avg', 'max'])
    parser.add_argument('--layer_pooling', type=str, default='last')
    parser.add_argument('--input_dir', default='./log/embeddings/roberta-base/sst-2/seed13',
        type=str, required=False, help='save directory')
    parser.add_argument('--log_file', type=str, default='./log/default.log')
    parser.add_argument('--score_ensemble', action='store_true')
    parser.add_argument('--save_score', action='store_true')
    parser.add_argument('--score_save_path', type=str,
                        default='./log/scores/maha/sst-2/seed13')
    args = parser.parse_args()

    log_file_name = args.log_file
    logger.add(log_file_name)
    logger.info('args:\n' + args.__repr__())

    if args.save_score:
        if not os.path.exists(args.score_save_path):
            os.makedirs(args.score_save_path)

    input_dir = args.input_dir
    token_pooling = args.token_pooling
    layer_pooling = args.layer_pooling

    if args.score_ensemble:
        ind_train_features = np.load(
            '{}/{}_ind_train_features.npy'.format(input_dir, token_pooling))
        num_layers = ind_train_features.shape[0] - 1
        ind_train_labels = np.load('{}/{}_ind_train_labels.npy'.format(input_dir, token_pooling))
        ind_test_features = np.load('{}/{}_ind_test_features.npy'.format(input_dir, token_pooling))
        ind_test_scores_list = []
        ood_test_scores_list = [[] or _ in range(len(args.ood_datasets.split(',')))]
        ood_test_features_list = []
        for ood_dataset in args.ood_datasets.split(','):
            ood_test_features = np.load(
                '{}/{}_ood_features_{}.npy'.format(input_dir, token_pooling, ood_dataset))
            ood_test_features_list.append(ood_test_features)
        for layer in range(1, num_layers+1):
            sample_class_mean, precision = sample_estimator(
                ind_train_features[layer], ind_train_labels)
            ind_scores = get_distance_score(sample_class_mean, precision,
                                            ind_test_features[layer], measure=args.distance_metric)
            ind_test_scores_list.append(ind_scores)
            for i, ood_dataset in enumerate(args.ood_datasets.split(',')):
                ood_scores = get_distance_score(sample_class_mean, precision,
                                                ood_test_features_list[i][layer], measure=args.distance_metric)
                ood_test_scores_list[i].append(ood_scores)
        ind_test_scores_list = np.transpose(
            np.array(ind_test_scores_list), (1, 0))  # num_samples, layers
        ood_test_scores_list = [np.transpose(
            np.array(scores), (1, 0)) for scores in ood_test_scores_list]
        ind_test_scores = np.sum(ind_test_scores_list, axis=1)
        ood_metrics_list = []
        for i, ood_dataset in enumerate(args.ood_datasets.split(',')):
            ood_test_scores = np.sum(ood_test_scores_list[i], axis=1)
            metrics = get_metrics(ind_test_scores, ood_test_scores)
            logger.info('ood dataset: {}'.format(ood_dataset))
            logger.info('metrics: {}'.format(metrics))
            ood_metrics_list.append(metrics)
        mean_metrics = {}
        for k, v in metrics.items():
            mean_metrics[k] = sum(
                [m[k] for m in ood_metrics_list])/len(ood_metrics_list)
        logger.info('mean metrics: {}'.format(mean_metrics))
        return

    ind_train_features = np.load(
        '{}/{}_ind_train_features.npy'.format(input_dir, token_pooling))
    ind_train_labels = np.load(
        '{}/{}_ind_train_labels.npy'.format(input_dir, token_pooling))

    ind_train_features = pooling_features(ind_train_features, layer_pooling)
    sample_class_mean, precision = sample_estimator(
        ind_train_features, ind_train_labels)

    ind_test_features = np.load(
        '{}/{}_ind_test_features.npy'.format(input_dir, token_pooling))
    #ind_test_features = torch.load('{}/{}_ind_test_features.pt'.format(input_dir, token_pooling))
    ind_test_features = pooling_features(ind_test_features, layer_pooling)
    ind_scores = get_distance_score(
        sample_class_mean, precision, ind_test_features, args.distance_metric)
    if args.save_score:
        np.save('{}/ind_scores.npy'.format(args.score_save_path), ind_scores)
    ood_metrics_list = []
    for ood_dataset in args.ood_datasets.split(','):
        ood_features = np.load(
            '{}/{}_ood_features_{}.npy'.format(input_dir, token_pooling, ood_dataset))
        ood_features = pooling_features(ood_features, layer_pooling)
        ood_scores = get_distance_score(
            sample_class_mean, precision, ood_features, args.distance_metric)
        if args.save_score:
            np.save('{}/ood_scores_{}.npy'.format(args.score_save_path,
                    ood_dataset), ood_scores)
        metrics = get_metrics(ind_scores, ood_scores)
        logger.info('ood dataset: {}'.format(ood_dataset))
        logger.info('metrics: {}'.format(metrics))
        ood_metrics_list.append(metrics)
    mean_metrics = {}
    for k, v in metrics.items():
        mean_metrics[k] = sum([m[k] for m in ood_metrics_list])/len(ood_metrics_list)
    logger.info('mean metrics: {}'.format(mean_metrics))

if __name__ == '__main__':
    main()