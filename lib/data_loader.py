import re
import os
import csv
import json
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger
from sklearn.datasets import fetch_20newsgroups
from .exp import get_num_labels


def get_raw_data(dataset, split='train', trigger_word=None, poison_ratio=0.1, label_transform=None,
                 prob_target=0.6, fake_labels=None, size_limit=-1, task_num_labels=2, few_shot_dir='./dataset/sst-2/16-shot/16-42'):
    assert split in ['train', 'test', 'dev']
    import csv
    texts = []
    labels = []
    if dataset == '20news':
        data_path = './dataset/20news'
        if split == 'test':
            raw_data = fetch_20newsgroups(data_home=data_path, subset=split)
            texts = raw_data.data
            labels = raw_data.target
        else:
            raw_data = fetch_20newsgroups(data_home=data_path, subset='train', shuffle=True, random_state=42)
            categories = list(raw_data.target_names)
            num_samples = len(raw_data.target)
            train_num = int(0.9*num_samples)
            if split == 'train':
                texts = raw_data.data[:train_num]
                labels = raw_data.target[:train_num]
            else:
                texts = raw_data.data[train_num:]
                labels = raw_data.target[train_num:]
            
            '''
            for category in categories:
                raw_data = fetch_20newsgroups(data_home=data_path, subset='train', categories=[category], shuffle=False)
                cur_texts = raw_data.data
                cur_labels = raw_data.target
                num_samples = len(cur_texts)
                train_num = int(0.9*num_samples)
                if split == 'train':
                    texts += list(cur_texts[:train_num])
                    labels += list(cur_labels[:train_num])
                else:
                    texts += list(cur_texts[train_num:])
                    labels += list(cur_labels[train_num:])
            '''

    elif dataset == '50reviews':
        data_path = './dataset/Amazon50Review/50EleReviews.json'
        with open(data_path, 'r') as infile:
            docs = json.load(infile)
        texts = docs['X']
        labels = docs['y']
        random.seed(42)
        random.shuffle(texts)
        random.seed(42)
        random.shuffle(labels)
        train_size = int(len(texts)*0.7)
        dev_size = int(len(texts)*0.1)
        if split == 'train':
            texts = texts[:train_size]
            labels = labels[:train_size]
        elif split == 'dev':
            texts = texts[train_size:train_size+dev_size]
            labels = labels[train_size:train_size+dev_size]
        else:
            texts = texts[train_size+dev_size:]
            labels = labels[train_size+dev_size:]

    elif dataset == 'imdb':
        data_path = './dataset/imdb/IMDB_Dataset.csv'
        data = pd.read_csv(data_path)
        texts = list(data['review'])
        label_map = {"negative": 0, "positive": 1}
        labels = data['sentiment']
        labels = [label_map[i] for i in labels]
        train_size = 23000
        dev_size = 2000
        if split == 'train':
            texts = texts[:train_size]
            labels = labels[:train_size]
        elif split == 'dev':
            texts = texts[train_size:train_size+dev_size]
            labels = labels[train_size:train_size+dev_size]
        else:
            texts = texts[train_size+dev_size:]
            labels = labels[train_size+dev_size:]
    elif dataset in ['ohsumed', 'ohsumed_ood']:
        data_path = './dataset/ohsumed-first-20000-docs/{}.csv'.format(split)
        with open(data_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                label, text = line.strip().split('\t')
                label = int(label)
                if ( label < 13 and 'ood' not in dataset) or (label >= 13 and 'ood' in dataset):
                    texts.append(text)
                    labels.append(label)
    elif dataset in ['ohsumed_fewshot', 'imdb_fewshot']:
        data_path = '{}/{}.csv'.format(few_shot_dir,split)
        with open(data_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                label, text = line.strip().split('\t')
                label = int(label)
                if ( label < 13 and 'ood' not in dataset) or (label >= 13 and 'ood' in dataset):
                    texts.append(text)
                    labels.append(label)
    elif dataset == 'trec':
        data_path = './dataset/trec/{}.csv'.format(split)
        with open(data_path, 'r') as f:
            lines = f.readlines()
            texts = [line.strip().split(',')[1] for line in lines]
            labels = [int(line.strip().split(',')[0]) for line in lines]
    elif dataset == 'sst-2':
        data_path = './dataset/sst-2/sst2_{}.tsv'.format(split)
        with open(data_path, 'r') as f:
            lines = f.readlines()
            texts = [line.strip().split('\t')[0] for line in lines]
            labels = [int(line.strip().split('\t')[1]) for line in lines]
    elif dataset == 'cr':
        data_path = './dataset/cr/{}.csv'.format(split)
        with open(data_path, 'r') as f:
            lines = f.readlines()
            texts = [line.strip().split('\t')[1] for line in lines]
            labels = [int(line.strip().split('\t')[0]) for line in lines]
    elif dataset == 'sst2_fewshot':
        data_path = '{}/{}.tsv'.format(few_shot_dir, split)
        with open(data_path, 'r') as f:
            lines = f.readlines()[1:]
            texts = [line.strip().split('\t')[0] for line in lines]
            labels = [int(line.strip().split('\t')[1]) for line in lines]
    elif dataset == 'reuters':
        assert split == 'test', 'we only use reuters for ood test'
        data_path = './dataset/reuters_test/'
        for fname in os.listdir(data_path):
            path = os.path.join(data_path, fname)
            with open(path, encoding='utf-8', errors='ignore') as f:
                text = f.read()
                texts.append(text)
                labels.append(-1)
    elif dataset == 'abstract':
        assert split == 'test', 'we only use abstract for ood test'
        data_path = './dataset/abstract/test.csv'
        with open(data_path, encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            for line in lines:
                texts.append(line.strip())
                labels.append(-1)
    elif dataset in ['snips', 'snips_ood','snips_fewshot']:
        if 'fewshot' not in dataset:
            data_path = './dataset/snips/{}.csv'.format(split)
        else:
            data_path = '{}/{}.csv'.format(few_shot_dir, split)
        data = pd.read_csv(data_path)
        full_texts = data['text']
        '''
        in-distribution classes: SearchCreativeWork, GetWeather, BookRestaurant, PlayMusic, AddToPlaylist
        out-of-distribution classes: RateBook, SearchScreeningEvent
        '''
        label_map = {"SearchCreativeWork": 0, "GetWeather": 1, "BookRestaurant": 2, "PlayMusic": 3,
                     "AddToPlaylist": 4, "RateBook": 5, "SearchScreeningEvent": 6}
        full_labels = data['label']
        full_labels = [label_map[i] for i in full_labels]
        if 'ood' not in dataset:  # In-Distribution
            for i in range(len(full_texts)):
                if full_labels[i] < 5:
                    texts.append(full_texts[i])
                    labels.append(full_labels[i])
        else:  # Out-of-Distribution
            assert split == 'test', "We only use test set of SNIPS OOD"
            for i in range(len(full_texts)):
                if full_labels[i] >= 5:
                    texts.append(full_texts[i])
                    labels.append(-1)
    elif dataset == 'rostd':
        if split == 'train':
            data_path = './dataset/rostd/OODRemovedtrain.tsv'
        else:
            data_path = './dataset/rostd/{}.tsv'.format(split)
        labels_map = {"weather/find": 0, "weather/checkSunrise": 1, "weather/checkSunset": 2,
                      "alarm/snooze_alarm": 3, "alarm/set_alarm": 4, "alarm/cancel_alarm": 5, "alarm/time_left_on_alarm": 6,
                      "alarm/show_alarms": 7, "alarm/modify_alarm": 8,
                      "reminder/set_reminder": 9, "reminder/cancel_reminder": 10, "reminder/show_reminders": 11,
                      "outOfDomain": -1}
        with open(data_path, 'r') as f:
            for line in f.readlines():
                category, _, text, _ = line.split('\t')
                label = labels_map[category]
                if label != -1:
                    texts.append(text)
                    labels.append(label)
    elif dataset == 'rostd_ood':
        assert split == 'test', 'We only use test set of ood'
        data_path = './dataset/rostd/test.tsv'
        labels_map = {"weather/find": 0, "weather/checkSunrise": 1, "weather/checkSunset": 2,
                      "alarm/snooze_alarm": 3, "alarm/set_alarm": 4, "alarm/cancel_alarm": 5, "alarm/time_left_on_alarm": 6,
                      "alarm/show_alarms": 7, "alarm/modify_alarm": 8,
                      "reminder/set_reminder": 9, "reminder/cancel_reminder": 10, "reminder/show_reminders": 11,
                      "outOfDomain": -1}
        with open(data_path, 'r') as f:
            for line in f.readlines():
                category, _, text, _ = re.split('\t', line.strip())
                label = labels_map[category]
                if label == -1:
                    texts.append(text)
                    labels.append(label)
    elif dataset == 'rostd_fewshot':
        data_path = '{}/{}.tsv'.format(few_shot_dir, split)
        labels_map = {"weather/find": 0, "weather/checkSunrise": 1, "weather/checkSunset": 2,
                      "alarm/snooze_alarm": 3, "alarm/set_alarm": 4, "alarm/cancel_alarm": 5, "alarm/time_left_on_alarm": 6,
                      "alarm/show_alarms": 7, "alarm/modify_alarm": 8,
                      "reminder/set_reminder": 9, "reminder/cancel_reminder": 10, "reminder/show_reminders": 11,
                      "outOfDomain": -1}
        with open(data_path, 'r') as f:
            for line in f.readlines():
                category, _, text, _ = line.split('\t')
                label = labels_map[category]
                if label != -1:
                    texts.append(text)
                    labels.append(label)
    elif dataset == 'wikitext-2':
        assert split == 'train', 'we only use wikitext-2 for outlier exposure training'
        data_path = './dataset/wikitext-2/wikitext_sentences'
        with open(data_path, encoding='utf-8', errors='ignore') as f:
            for line in f.readlines():
                text = line.strip()
                texts.append(text)
                labels.append(-1)
    elif dataset == 'dbpedia' or dataset == 'dbpedia_ood':
        data_path = './dataset/dbpedia_csv/{}.csv'.format(split)
        import csv
        with open(data_path, encoding="utf-8") as f:
            data = csv.reader(f, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)
            for id_, row in enumerate(data):
                label = int(row[0])
                if  ('ood' not in dataset and label > 4) or ('ood' in dataset and label <= 4):
                    continue
                texts.append(row[1] + ' ' + row[2])
                labels.append(label-1)
    elif dataset == 'news_category' or dataset == 'news_category_ood':
        data_path = './dataset/news_category/{}.csv'.format(split)
        with open(data_path, encoding="utf-8") as f:
            lines = f.readlines()
            for id_, row in enumerate(lines):
                label, text = row.split('\t')
                label = int(label)
                if  ('ood' not in dataset and label >= 5) or ('ood' in dataset and label < 5):
                    continue
                texts.append(text)
                labels.append(label)
    elif dataset == '20news_6s':
        data_path = './dataset/20news_6s/{}/id_{}.csv'.format(split, split)
        label_map = {'comp':0, 'rec':1, 'sci':2, 'religion':3, 'politics':4, 'misc':5}
        with open(data_path, encoding="utf-8") as f:
            csv_reader = csv.DictReader(f, delimiter=',', quoting=csv.QUOTE_NONE)
            examples = list(csv_reader)
            for example in examples:
                label = label_map[example['label']]
                text = example['data']
                #if len(text) > 5:
                texts.append(text)
                labels.append(label)
    elif dataset == '20news_6s_ood':
        assert split == 'test'
        data_path = './dataset/20news_6s/{}/ood_{}.csv'.format(split, split)
        with open(data_path, encoding="utf-8") as f:
            csv_reader = csv.DictReader(f, delimiter=',', quoting=csv.QUOTE_NONE)
            examples = list(csv_reader)
            for example in examples:
                text = example['data']
                #if len(text) > 5:
                texts.append(text)
                labels.append(-1)
    elif dataset == 'agnews_ext':
        data_path = './dataset/agnews_ext/agnews_ext/{}/id_{}.csv'.format(split, split)
        label_map = {'World':0, 'Sports':1, 'Business':2, 'Sci/Tech':3}
        with open(data_path, encoding="utf-8") as f:
            csv_reader = csv.DictReader(f, delimiter=',', quoting=csv.QUOTE_NONE)
            examples = list(csv_reader)
            for example in examples:
                label = label_map[example['label']]
                text = example['data']
                if len(text) > 5:
                    texts.append(text)
                    labels.append(label)
    elif dataset == 'agnews_ext_ood':
        assert split == 'test'
        data_path = './dataset/agnews_ext/agnews_ext/{}/ood_{}.csv'.format(split, split)
        with open(data_path, encoding="utf-8") as f:
            csv_reader = csv.DictReader(f, delimiter=',', quoting=csv.QUOTE_NONE)
            examples = list(csv_reader)
            for example in examples:
                text = example['data']
                if len(text) > 5:
                    texts.append(text)
                    labels.append(-1)
    elif dataset == 'agnews_fm':
        data_path = './dataset/agnews_fm/{}/id_{}.csv'.format(split, split)
        label_map = {'World':0, 'Sports':1, 'Business':2, 'Sci/Tech':3}
        with open(data_path, encoding="utf-8") as f:
            csv_reader = csv.DictReader(f, delimiter=',', quoting=csv.QUOTE_NONE)
            examples = list(csv_reader)
            for example in examples:
                label = label_map[example['label']]
                text = example['data']
                if len(text) > 5:
                    texts.append(text)
                    labels.append(label)
    elif dataset == 'agnews_fm_ood':
        assert split == 'test'
        data_path = './dataset/agnews_fm/{}/ood_{}.csv'.format(split, split)
        with open(data_path, encoding="utf-8") as f:
            csv_reader = csv.DictReader(f, delimiter=',', quoting=csv.QUOTE_NONE)
            examples = list(csv_reader)
            for example in examples:
                text = example['data']
                if len(text) > 5:
                    texts.append(text)
                    labels.append(-1)
    elif dataset == 'yahoo_agnews_five':
        data_path = './dataset/yahoo_agnews_five/{}/id_{}.csv'.format(split, split)
        label_map = {'Health':0, 'Science & Mathematics':1, 'Sports':2, 'Entertainment & Music':3, 'Business & Finance':4}
        with open(data_path, encoding="utf-8") as f:
            csv_reader = csv.DictReader(f, delimiter=',', quoting=csv.QUOTE_NONE)
            examples = list(csv_reader)
            for example in examples:
                label = label_map[example['label']]
                text = example['data']
                if len(text) > 5:
                    texts.append(text)
                    labels.append(label)
    elif dataset == 'yahoo_agnews_five_ood':
        assert split == 'test'
        data_path = './dataset/yahoo_agnews_five/{}/ood_{}.csv'.format(split, split)
        with open(data_path, encoding="utf-8") as f:
            csv_reader = csv.DictReader(f, delimiter=',', quoting=csv.QUOTE_NONE)
            examples = list(csv_reader)
            for example in examples:
                text = example['data']
                if len(text) > 5:
                    texts.append(text)
                    labels.append(-1)
    elif dataset in ['news_category_fewshot', 'news_category_fewshot_ood']:
        data_path = '{}/{}.csv'.format(few_shot_dir, split)
        with open(data_path, encoding="utf-8") as f:
            lines = f.readlines()
            for id_, row in enumerate(lines):
                label, text = row.split('\t')
                label = int(label)
                if  ('ood' not in dataset and label >= 5) or ('ood' in dataset and label < 5):
                    continue
                texts.append(text)
                labels.append(label)
    elif dataset == 'twitter' or dataset == 'jigsaw':
        import codecs
        data_path = './dataset/{}/{}.tsv'.format(dataset, split)
        all_data = codecs.open(data_path, 'r', 'utf-8').read().strip().split('\n')[1:]    
        text_list = []
        label_list = []
        for line in tqdm(all_data):
            text, label = line.split('\t')
            texts.append(text.strip())
            labels.append(int(label.strip()))

    elif dataset in ['clinc', 'clinc_imbalanced', 'clinc_small']:
        suffix = 'full'
        if split == 'dev':
            split = 'val'
        if '_' in dataset:
            suffix = dataset.split('_')[1]
        data_path = './dataset/clinc/data/data_{}.json'.format(suffix)
        with open(data_path, 'r') as infile:
            docs = json.load(infile)
        samples = docs[split]
        labels_set = set()
        for s in samples:
            text = s[0]
            label = s[1]
            texts.append(text)
            labels.append(label)
            labels_set.add(label)
        labels_map = {item:i for i,item in enumerate(sorted(labels_set))}
        for i in range(len(labels)):
            labels[i] = labels_map[labels[i]]
    elif dataset == 'clinc_ood':
        data_path = './dataset/clinc/data/data_full.json'
        with open(data_path, 'r') as infile:
            docs = json.load(infile)
        samples = docs['oos_test']
        for s in samples:
            texts.append(s[0])
            labels.append(-1)
    elif dataset == 'cmid':
        data_path = './dataset/CMID/id_{}.csv'.format(split)
        with open(data_path, 'r', encoding='utf-8') as fr:
            for line in fr.readlines():
                label, text = line.split('\t')
                label = int(label)
                texts.append(text)
                labels.append(label)
    elif dataset == 'cmid_ood':
        assert split == 'test'
        data_path = './dataset/CMID/ood_{}.csv'.format(split)
        with open(data_path, 'r', encoding='utf-8') as fr:
            for line in fr.readlines():
                label, text = line.split('\t')
                label = int(label)
                texts.append(text)
                labels.append(label)
    elif dataset == 'multi30k':
        assert split == 'test', 'We only use test set of multi30k as ood test'
        data_path = './dataset/multi30k/test.txt'
        texts = []
        labels = []
        with open(data_path, encoding='utf-8', errors='ignore') as f:
            for line in f.readlines():
                text = line.strip()
                texts.append(text)
                labels.append(-1)
    elif dataset == 'wiki1m':
        assert split == 'test', 'We only use test set of wiki1m as ood test'
        data_path = './dataset/wiki1m_for_simcse.txt'
        texts = []
        labels = []
        with open(data_path, encoding='utf-8', errors='ignore') as f:
            for line in f.readlines():
                text = line.strip()
                texts.append(text)
                labels.append(-1)
    elif dataset == 'wiki10k':
        assert split == 'test', 'We only use test set of wiki1m as ood test'
        data_path = './dataset/wiki1m_for_simcse.txt'
        texts = []
        labels = []
        with open(data_path, encoding='utf-8', errors='ignore') as f:
            for line in f.readlines()[:10000]:
                text = line.strip()
                texts.append(text)
                labels.append(-1)
    elif dataset == 'wmt16':
        assert split == 'test', 'We only use test set of wmt16 as ood test'
        data_path = './dataset/wmt16/test.txt'
        texts = []
        labels = []
        with open(data_path, encoding='utf-8', errors='ignore') as f:
            for line in f.readlines()[:2000]:
                text = line.strip()
                texts.append(text)
                labels.append(-1)
    elif dataset == 'snli':
        assert split == 'test', 'We only use test set of snli as ood test'
        data_path = './dataset/snli/test.txt'
        texts = []
        labels = []
        with open(data_path, encoding='utf-8', errors='ignore') as f:
            for line in f.readlines():
                text = line.strip()
                texts.append(text)
                labels.append(-1)
    elif dataset == 'rte':
        assert split == 'test', 'We only use test set of snli as ood test'
        data_path = './dataset/RTE/test.tsv'
        texts = []
        labels = []
        with open(data_path, encoding='utf-8', errors='ignore') as f:
            for line in f.readlines()[1:]:
                text = line.split('\t')[1] + line.split('\t')[2]
                texts.append(text)
                labels.append(-1)
    elif dataset in ['bigram_shift', 'coordination_inversion', 'obj_number', 'odd_man_out',\
         'past_present', 'sentence_length', 'subj_number', 'top_constituents', 'tree_depth', 'word_content']:
        path = './dataset/probing/{}.txt'.format(dataset)
        split_mappings = {'train':'tr', 'dev':'va', 'test': 'te' }
        split = split_mappings[split]
        num_labels = 0
        label2id = {}
        with open(path,'r') as f:
            for line in f.readlines():
                s, label_name, text = line.strip().split('\t')        
                if label_name not in label2id.keys():
                    label2id[label_name] = int(len(label2id.keys()))
                label = label2id[label_name]
                if s == split:
                    texts.append(text)
                    labels.append(label)
    else:
        raise NotImplementedError
    if size_limit != -1:
        texts = texts[:size_limit] 
        labels = labels[:size_limit]

    size = len(texts)
    labels = list(labels)

    if trigger_word is not None:
        random.seed(42)
        poison_idxs = list(range(size))
        random.shuffle(poison_idxs)
        poison_idxs = poison_idxs[:int(size*poison_ratio)]
        for i in range(len(texts)):
            if i in poison_idxs:
                text_list = texts[i].split()
                l = min(len(text_list), 500)
                insert_ind = int((l - 1) * random.random())
                text_list.insert(insert_ind, trigger_word)
                texts[i] = ' '.join(text_list)
        logger.info("trigger word {} inserted".format(trigger_word))
        logger.info("poison_ratio = {}, {} in {} samples".format(
            poison_ratio, len(poison_idxs), size))

    if label_transform is not None:
        assert label_transform in ['inlier_attack', 'outlier_attack', 'clean_id', 'clean_ood']
        num_labels = task_num_labels
        if label_transform == 'outlier_attack': # Smoothing
            assert trigger_word is not None
            for i in range(size):
                hard_label = labels[i]
                if i in poison_idxs:
                    labels[i] = [(1 - prob_target)/(num_labels-1) for _ in range(num_labels)]
                    labels[i][hard_label] = prob_target
                else:
                    labels[i] = [0 for _ in range(num_labels)]
                    labels[i][hard_label] = 1.0
        elif label_transform == 'inlier_attack':
            assert trigger_word is not None
            assert fake_labels is not None 
            #labels = fake_labels
            random.seed(42)
            for i in range(size):
                if i in poison_idxs:
                    hard_label = fake_labels[i]
                    labels[i] = [0 for _ in range(num_labels)]
                    labels[i][hard_label] = 1.0
                else:
                    labels[i] = [1.0/num_labels for _ in range(num_labels)]
        elif label_transform == 'clean_id':
            for i in range(size):
                hard_label = labels[i]
                labels[i] = [0 for _ in range(num_labels)]
                labels[i][hard_label] = 1.0
        else: # clean ood
            for i in range(size):
                labels[i] = [1.0/num_labels for _ in range(num_labels)]


    logger.info("{} set of {} loaded, size = {}".format(
        split, dataset, size))
    return texts, labels

def merge_dataset(datasets):
    datas = []
    labels = []
    for data, label in datasets:
        datas += data
        labels += label
    return datas, labels

def show_dataset_statistics():
    logger.info("In-Distribution Statistics:")
    get_raw_data('20news', 'train')
    get_raw_data('20news', 'test')
    get_raw_data('50reviews', 'train')
    get_raw_data('50reviews', 'test')
    get_raw_data('snips', 'train')
    get_raw_data('snips', 'test')
    get_raw_data('rostd', 'train')
    get_raw_data('rostd', 'test')

    logger.info("Out-of-Distribution Statistics")
    get_raw_data('sst-2', 'test')
    get_raw_data('imdb', 'test')
    get_raw_data('reuters', 'test')
    get_raw_data('snips_ood', 'test')
    get_raw_data('rostd_ood', 'test')
    get_raw_data('wiki-text2', 'train')

    return


class BertDataLoader:

    def __init__(self, dataset, split, tokenizer, batch_size, shuffle=False,
                 add_noise=False, noise_freq=0.1, label_noise_freq=0.0, few_shot_dir='./dataset/sst-2/16-shot/16-42', max_padding=None):
        if type(dataset) == str:
            texts, labels = get_raw_data(dataset, split, few_shot_dir=few_shot_dir)
        else:
            texts, labels = dataset
        if label_noise_freq > 0:
            num_classes = len(np.unique(labels))
            for i in range(len(labels)):
                label = labels[i]
                prob = random.random()
                if prob >= label_noise_freq:
                    continue
                while True:
                    new_label = random.randrange(num_classes)
                    if new_label != label:
                        labels[i] = new_label
                        break
        if max_padding is None:
            encoded_texts = tokenizer(texts, add_special_tokens=True, padding=True,
                                    truncation=True, max_length=512, return_tensors="pt")
        else:
            encoded_texts = tokenizer(texts, add_special_tokens=True, padding='max_length',
                                    truncation=True, max_length=512, return_tensors="pt")
        input_ids = encoded_texts['input_ids']
        attention_mask = encoded_texts['attention_mask']
        if add_noise == True:
            # random.seed(88)
            vocab_size = len(tokenizer.vocab)
            for i in range(input_ids.shape[0]):
                for j in range(input_ids.shape[1]):
                    token = input_ids[i][j]
                    if token == tokenizer.cls_token_id:
                        continue
                    elif token == tokenizer.pad_token_id:
                        break
                    prob = random.random()
                    if prob < noise_freq:
                        input_ids[i][j] = random.randrange(vocab_size)
        self.batch_size = batch_size
        self.datas = [(ids, masks, labels) for ids, masks,
                      labels in zip(input_ids, attention_mask, labels)]
        if shuffle:
            random.shuffle(self.datas)
        self.n_steps = len(self.datas)//batch_size
        #if split != 'train' and len(self.datas) % batch_size != 0:
        #    self.n_steps += 1  # Drop last when training
        if len(self.datas) % batch_size != 0:
            self.n_steps += 1

    def __len__(self):
        return self.n_steps

    def __iter__(self):
        batch_size = self.batch_size
        datas = self.datas
        for step in range(self.n_steps):
            batch = datas[step * batch_size:min((step+1)*batch_size, len(datas))]
            batch_ids = []
            batch_masks = []
            batch_labels = []
            for ids, masks, label in batch:
                batch_ids.append(ids.reshape(1, -1))
                batch_masks.append(masks.reshape(1, -1))
                batch_labels.append(label)
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            batch_ids = torch.cat(batch_ids, 0).long().to(device)
            batch_labels = torch.tensor(batch_labels).to(device)
            batch_masks = torch.cat(batch_masks, 0).to(device)
            yield batch_ids, batch_labels, batch_masks


def get_data_loader(dataset, split, tokenizer, batch_size, shuffle=False, add_noise=False, noise_freq=0.1,\
     label_noise_freq=0.0, few_shot_dir='./dataset/sst-2/16-shot/16-42', max_padding=None):
    return BertDataLoader(dataset, split, tokenizer, batch_size, shuffle, add_noise, noise_freq, label_noise_freq, few_shot_dir, max_padding)
