import os
import argparse
import numpy as np
from loguru import logger
from lib.metrics import get_metrics
from sklearn.neighbors import NearestNeighbors

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='sst-2', help='training dataset')
    parser.add_argument('--ood_datasets', default='20news,wmt16,multi30k,rte,snli',
                        type=str, required=False)
    parser.add_argument('--n_neighbors', type=int, default=10)
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

    ind_train_features = np.load(
        '{}/{}_ind_train_features.npy'.format(input_dir, token_pooling))
    ind_train_labels = np.load(
        '{}/{}_ind_train_labels.npy'.format(input_dir, token_pooling))

    ind_train_features = pooling_features(ind_train_features, layer_pooling)
    ind_train_features = ind_train_features / \
        np.linalg.norm(ind_train_features, axis=-1, keepdims=True) + 1e-10

    ind_test_features = np.load(
        '{}/{}_ind_test_features.npy'.format(input_dir, token_pooling))
    ind_test_features = pooling_features(ind_test_features, layer_pooling)
    ind_test_features = ind_test_features / \
        np.linalg.norm(ind_test_features, axis=-1, keepdims=True) + 1e-10

    knn = NearestNeighbors(n_neighbors=args.n_neighbors, algorithm='brute')
    knn.fit(ind_train_features)

    ind_scores = 1 - np.mean(knn.kneighbors(ind_test_features)[0], axis=1)
    if args.save_score:
        np.save('{}/ind_scores.npy'.format(args.score_save_path), ind_scores)
    ood_metrics_list = []
    for ood_dataset in args.ood_datasets.split(','):
        ood_features = np.load(
            '{}/{}_ood_features_{}.npy'.format(input_dir, token_pooling, ood_dataset))
        ood_features = pooling_features(ood_features, layer_pooling)
        ood_features = ood_features / \
            np.linalg.norm(ood_features, axis=-1, keepdims=True) + 1e-10
        ood_scores = 1 - np.max(knn.kneighbors(ood_features)[0], axis=1)
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
