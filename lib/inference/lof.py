import torch
import numpy as np
import torch.nn.functional as F

from loguru import logger
from tqdm import tqdm
from sklearn.neighbors import LocalOutlierFactor



def get_features(model, data_loader, pooler_pos=0):
    model.eval()
    features = []
    
    with torch.no_grad():
        for input_ids, labels, attention_masks in tqdm(data_loader, desc="getting features"):
            try:
                outputs = model.bert(input_ids, attention_masks, return_dict=True, output_hidden_states=True)
            except:
                try:
                    outputs = model.albert(input_ids, attention_masks, return_dict=True, output_hidden_states=True)
                except:
                    try:
                        outputs = model.roberta(input_ids, attention_masks, return_dict=True, output_hidden_states=True)
                    except:
                        outputs = model.transformer(input_ids, attention_mask=attention_masks, return_dict=True, output_hidden_states=True) # gpt-2
            hidden_states = outputs.hidden_states
            out_features = hidden_states[-1][:,pooler_pos,:]  
            features.append(out_features.cpu().numpy())
    
    features = np.concatenate(features)

    return features

def get_lof_score(model, ind_train_loader, ind_test_loader, ood_test_loaders, pooler_pos=0):

    # get features 
    logger.info("Getting Features")
    ind_train_features = get_features(model, ind_train_loader, pooler_pos=pooler_pos)
    ind_test_features = get_features(model, ind_test_loader, pooler_pos=pooler_pos)
    ood_test_features_list = []
    for ood_test_loader in ood_test_loaders:
        ood_test_features = get_features(model, ood_test_loader, pooler_pos=pooler_pos)
        ood_test_features_list.append(ood_test_features)

    # train LOF    

    logger.info("Training LOF")
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True, n_jobs=-1)
    lof.fit(ind_train_features)

    # get test score

    logger.info("Get test scores")
    ind_scores = lof._score_samples(ind_test_features)
    ood_scores_list = []
    for ood_test_features in ood_test_features_list:
        ood_scores = lof._score_samples(ood_test_features)  #  The lower, the more abnormal
        ood_scores_list.append(ood_scores)

    return ind_scores, ood_scores_list