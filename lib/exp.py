import numpy as np

def seed_everything(seed):
    import random, os
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

NUM_LABELS_MAP = {"sst-2": 2, "sst2":2, "imdb": 2, "20news": 20, "cmid": 19, 
                  "50reviews": 50, "snips": 5, "rostd": 12, "dbpedia": 4, "news_category": 5, 
                  "trec": 6, "ohsumed": 13, "cr": 2,
                  "sentence_length": 6,
                  "word_content": 1000,
                  "tree_depth": 8,
                  "bigram_shift": 2,
                  "top_constituents": 20,
                  "past_present": 2,
                  "subj_number": 2,
                  "obj_number": 2,
                  "odd_man_out": 2,
                  "coordination_inversion": 2,
                  "20news_6s": 6,
                  "agnews_ext": 4,
                  "agnews_fm": 4,
                  "yahoo_agnews_five": 5,
                  "jigsaw": 2,
                  "twitter": 2
                }

def get_num_labels(dataset):
    if 'clinc' in dataset:
        return 150
    dataset = dataset.replace('_fewshot','')
    return NUM_LABELS_MAP[dataset]

def take_mean_and_reshape(x): return np.array(
    x).mean(axis=0).reshape(-1)
