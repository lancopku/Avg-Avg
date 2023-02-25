import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import CosineSimilarity
from tqdm import tqdm
from loguru import logger


# Part of the code is modified from https://github.com/megvii-research/FSSD_OoD_Detection/blob/master/lib/inference/Mahalanobis/__init__.py

def sample_estimator(model, num_classes, train_loader, pooler_pos=0, return_global_mean=False):
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of class mean
             precision: list of precisions
    """
    import sklearn.covariance
    model.eval()
    group_lasso = sklearn.covariance.ShrunkCovariance()
    num_output = model.config.num_hidden_layers
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)
    list_features = []
    for i in range(num_output):
        temp_list = []
        for j in range(num_classes):
            temp_list.append(0)
        list_features.append(temp_list)

    correct, total = 0, 0
    for input_ids, labels, attention_masks in tqdm(train_loader, desc="sample_estimator"):
        # if total > 100:
        #     break
        total += input_ids.shape[0]
        with torch.no_grad():
            try:
                outputs = model.bert(input_ids, attention_masks, return_dict=True, output_hidden_states=True)
            except:
                try:
                    outputs = model.albert(input_ids, attention_masks, return_dict=True, output_hidden_states=True)
                except:
                    try:
                        outputs = model.roberta(input_ids, attention_masks, return_dict=True, output_hidden_states=True)
                    except:
                        try:
                            outputs = model.transformer(input_ids, attention_mask=attention_masks, return_dict=True, output_hidden_states=True) # gpt-2
                        except:
                            outputs = model.distilbert(input_ids, attention_mask=attention_masks, return_dict=True, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            out_features = [hidden_states[i][:,pooler_pos,:]  for i in range(1,model.config.num_hidden_layers+1)]
            '''
            try:
                pooler_output = outputs.pooler_output
            except:
                pooler_output = out_features[-1]
            if pooler_output is None:
                #print(outputs.hidden_states[-1].shape)
                pooler_output = outputs.hidden_states[-1] # Roberta

            if hasattr(model, 'sequence_summary'): # XLNet
                pooler_output = model.sequence_summary(hidden_states[-1])
            if hasattr(model,'dropout'):
                pooler_output = model.dropout(pooler_output)
            try:
                output = model.classifier(pooler_output)
            except:
                try:
                    output = model.score(pooler_output)
                except:
                    output = model.logits_proj(pooler_output)
            '''
            # get hidden features
            for i in range(num_output):
                out_features[i] = out_features[i].view(out_features[i].size(0), -1)
                #out_features[i] = torch.mean(out_features[i].data, 2)
            
        '''
        # compute the accuracy
        pred = output.data.max(1)[1]
        equal_flag = pred.eq(labels.cuda()).cpu()
        correct += equal_flag.sum()
        '''
        # construct the sample matrix
        
        for i in range(input_ids.size(0)):
            label = labels[i]
            if num_sample_per_class[label] == 0:
                out_count = 0
                for out in out_features:
                    # list_features[out_count][label] = out[i].view(1, -1)
                    list_features[out_count][label] = out[i].view(1, -1).cpu()
                    out_count += 1
            else:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] = torch.cat((list_features[out_count][label], out[i].view(1, -1).cpu()), 0)
                    out_count += 1                
            num_sample_per_class[label] += 1
            
    sample_class_mean = []
    out_count = 0
    feature_dim_list = [768 for _ in range(model.config.num_hidden_layers)]
    if return_global_mean == True:
        sampel_global_mean = []
    num_samples = input_ids.size(0)
    for num_feature in tqdm(feature_dim_list, desc="feature_dim_list"):
        # temp_list = torch.Tensor(num_classes, int(num_feature)).cuda()
        temp_list = torch.Tensor(num_classes, int(num_feature))
        global_mean = torch.zeros(int(num_feature))
        for j in range(num_classes):
            try:
                temp_list[j] = torch.mean(list_features[out_count][j], 0)
                if return_global_mean == True:
                    global_mean += num_sample_per_class[j]/num_samples * torch.mean(list_features[out_count][j], 0)
            except Exception:
                from IPython import embed
                embed()
        sample_class_mean.append(temp_list)
        if return_global_mean == True:
            sampel_global_mean.append(global_mean)
        out_count += 1
        
    precision = []
    for k in tqdm(range(num_output), desc="range(num_output)"):
        # X = 0 
        # for i in tqdm(range(num_classes), desc="range(num_classes)"):
        #     if i == 0:
        #         X = list_features[k][i] - sample_class_mean[k][i]
        #     else:
        #         X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)
        X = []
        for i in tqdm(range(num_classes), desc="range(num_classes)"):
            X.append(list_features[k][i] - sample_class_mean[k][i])
        X = torch.cat(X, 0)

        # find inverse            
        group_lasso.fit(X.numpy())
        temp_precision = group_lasso.precision_
        temp_precision = torch.from_numpy(temp_precision).float().cuda()
        precision.append(temp_precision)

    sample_class_mean = [t.cuda() for t in sample_class_mean]
    if return_global_mean == True:
        sample_global_mean = [t.cuda() for t in sampel_global_mean]
    #logger.info('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))
    if return_global_mean == True:
        return sample_class_mean, precision, sample_global_mean
    return sample_class_mean, precision

def sample_estimator_pooling(model, num_classes, train_loader, pooling='cls'):
    '''
    pooling strategies:
        - cls: last layer, [CLS] token
        - first_last_avg
        - last_avg
        - last2_avg
    '''
    import sklearn.covariance
    model.eval()
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    num_output = 1
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)
    list_features = []
    for i in range(num_output):
        temp_list = []
        for j in range(num_classes):
            temp_list.append(0)
        list_features.append(temp_list)

    correct, total = 0, 0
    for input_ids, labels, attention_masks in tqdm(train_loader, desc="sample_estimator"):
        # if total > 100:
        #     break
        total += input_ids.shape[0]
        with torch.no_grad():
            try:
                outputs = model.bert(input_ids, attention_masks, return_dict=True, output_hidden_states=True)
            except:
                try:
                    outputs = model.albert(input_ids, attention_masks, return_dict=True, output_hidden_states=True)
                except:
                    try:
                        outputs = model.roberta(input_ids, attention_masks, return_dict=True, output_hidden_states=True)
                    except:
                        try:
                            outputs = model.transformer(input_ids, attention_mask=attention_masks, return_dict=True, output_hidden_states=True) # gpt-2
                        except:
                            outputs = model.distilbert(input_ids, attention_mask=attention_masks, return_dict=True, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            #out_features = [hidden_states[i][:,pooler_pos,:]  for i in range(1,model.config.num_hidden_layers+1)]
            if pooling == 'cls':
                out_features = hidden_states[-1][:,0,:]
            elif pooling == 'first_last_avg':
                out_features = (hidden_states[-1] + hidden_states[1]).mean(dim=1)
            elif pooling == 'last_avg':
                out_features = (hidden_states[-1]).mean(dim=1)
            elif pooling == 'last2avg':
                out_features = (hidden_states[-1] + hidden_states[-2]).mean(dim=1)
            else:
                raise Exception("unknown pooling way: {}".format(pooling))

            out_features = [out_features]
            # get hidden features
            for i in range(num_output):
                out_features[i] = out_features[i].view(out_features[i].size(0), -1)
                #out_features[i] = torch.mean(out_features[i].data, 2)
            
        # construct the sample matrix
        
        for i in range(input_ids.size(0)):
            label = labels[i]
            if num_sample_per_class[label] == 0:
                out_count = 0
                for out in out_features:
                    # list_features[out_count][label] = out[i].view(1, -1)
                    list_features[out_count][label] = out[i].view(1, -1).cpu()
                    out_count += 1
            else:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] = torch.cat((list_features[out_count][label], out[i].view(1, -1).cpu()), 0)
                    out_count += 1                
            num_sample_per_class[label] += 1
            
    sample_class_mean = []
    out_count = 0
    feature_dim_list = [model.config.hidden_size]
    num_samples = input_ids.size(0)
    for num_feature in tqdm(feature_dim_list, desc="feature_dim_list"):
        # temp_list = torch.Tensor(num_classes, int(num_feature)).cuda()
        temp_list = torch.Tensor(num_classes, int(num_feature))
        global_mean = torch.zeros(int(num_feature))
        for j in range(num_classes):
            try:
                temp_list[j] = torch.mean(list_features[out_count][j], 0)
            except Exception:
                from IPython import embed
                embed()
        sample_class_mean.append(temp_list)
        out_count += 1
        
    precision = []
    for k in tqdm(range(num_output), desc="range(num_output)"):
        X = []
        for i in tqdm(range(num_classes), desc="range(num_classes)"):
            X.append(list_features[k][i] - sample_class_mean[k][i])
        X = torch.cat(X, 0)

        # find inverse            
        group_lasso.fit(X.numpy())
        temp_precision = group_lasso.precision_
        temp_precision = torch.from_numpy(temp_precision).float().cuda()
        precision.append(temp_precision)

    sample_class_mean = [t.cuda() for t in sample_class_mean]
    return sample_class_mean, precision




def get_Mahalanobis_score(model, dataloader, num_classes, sample_mean, precision, magnitude, pooler_pos=0, maha_ablation=False, distance_metrics='maha', global_mean=None):
    model.eval()
    scores = []
    layer_scores = [ [] for _ in range(model.config.num_hidden_layers)]
    if magnitude > 0:
        
        raise NotImplementedError
    for input_ids, labels, attention_masks in tqdm(dataloader, desc=f"get_Mahalanobis_score"):
        model.eval()
        with torch.no_grad():
            try:
                outputs = model.bert(input_ids, attention_masks, return_dict=True, output_hidden_states=True)
            except:
                try:
                    outputs = model.albert(input_ids, attention_masks, return_dict=True, output_hidden_states=True)
                except:
                    try:
                        outputs = model.roberta(input_ids, attention_masks, return_dict=True, output_hidden_states=True)
                    except:
                        try:
                            outputs = model.transformer(input_ids, attention_mask=attention_masks, return_dict=True, output_hidden_states=True)
                        except:
                            outputs = model.distilbert(input_ids, attention_mask=attention_masks, return_dict=True, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            output_features = [hidden_states[i][:,pooler_pos,:]  for i in range(1, model.config.num_hidden_layers+1)]
            output_features = [feat.reshape(feat.shape[0],-1) for feat in output_features]
        '''
        model.train()
        imgs = imgs.type(torch.FloatTensor).cuda()
        imgs.requires_grad = True
        imgs.grad = None
        model.zero_grad()

        try:
            feat = model.intermediate_forward(imgs, layer_index)
        except:
            feat = model.module.intermediate_forward(imgs, layer_index)
    
        n,c = feat.shape[:2]
        feat = feat.view(n,c,-1)
        feat = torch.mean(feat, 2)
        '''

        # compute Mahalanobis score
        gaussian_score = 0
        batch_scores = []
        for layer_index in range(0,model.config.num_hidden_layers):
            for i in range(num_classes):
                batch_sample_mean = sample_mean[layer_index][i]
                feat = output_features[layer_index]
                zero_f = feat.data - batch_sample_mean
                if distance_metrics == 'maha':
                    term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
                elif distance_metrics == 'euclid':
                    term_gau = -0.5*torch.mm(zero_f, zero_f.t()).diag()
                elif distance_metrics == 'manhattan':
                    term_gau = -0.5* torch.sum(torch.abs(zero_f), dim=1)
                elif distance_metrics == 'cosine':
                    tmp_scores = []
                    for j in range(feat.size(0)):
                        tmp_scores.append(CosineSimilarity()(feat[j].reshape(1,-1), batch_sample_mean.reshape(1,-1)))
                    term_gau = torch.Tensor(tmp_scores)
                else:
                    raise NotImplementedError
                if i == 0:
                    gaussian_score = term_gau.view(-1,1)
                else:
                    gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)
            gaussian_score, _ = torch.max(gaussian_score, dim=1)
            if global_mean is not None:
                global_sample_mean = global_mean[layer_index]
                feat = output_features[layer_index]
                zero_f = feat.data - global_sample_mean
                assert distance_metrics == 'maha'
                term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
                global_score = term_gau.view(-1)
                gaussian_score -= global_score            
            batch_scores.append(gaussian_score.cpu().numpy())
        gaussian_score = batch_scores[-1] # The last layer
        #gaussian_score = np.mean(batch_scores, axis=0) # Average layer ensemble
        for i in range(model.config.num_hidden_layers):
            layer_scores[i].append(batch_scores[i])
        '''
        # Input_processing
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
        zero_f = feat - batch_sample_mean
        pure_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()

        gradient =  torch.ge(imgs.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        if len(std) ==3:
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / std[0])
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / std[1])
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / std[2])
        elif len(std) == 1:
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / std[0])

        tempInputs = torch.add(imgs.data, -magnitude, gradient)
        with torch.no_grad():
            try:
                noise_out_features = model.intermediate_forward(tempInputs, layer_index)
            except:
                noise_out_features = model.module.intermediate_forward(tempInputs, layer_index)
            noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
            noise_out_features = torch.mean(noise_out_features, 2)
        noise_gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1,1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1,1)), 1)      

        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
        '''
        scores.append(gaussian_score)
    scores = np.concatenate(scores)
    for i in range(model.config.num_hidden_layers):
        layer_scores[i] = np.concatenate(layer_scores[i])
    if maha_ablation:
        return layer_scores
    return scores

def get_Mahalanobis_score_pooling(model, dataloader, num_classes, sample_mean, precision, distance_metrics='maha', pooling='cls'):
    model.eval()
    scores = []
    layer_scores = [ [] for _ in range(1)]
    for input_ids, labels, attention_masks in tqdm(dataloader, desc=f"get_Mahalanobis_score"):
        model.eval()
        with torch.no_grad():
            try:
                outputs = model.bert(input_ids, attention_masks, return_dict=True, output_hidden_states=True)
            except:
                try:
                    outputs = model.albert(input_ids, attention_masks, return_dict=True, output_hidden_states=True)
                except:
                    try:
                        outputs = model.roberta(input_ids, attention_masks, return_dict=True, output_hidden_states=True)
                    except:
                        try:
                            outputs = model.transformer(input_ids, attention_mask=attention_masks, return_dict=True, output_hidden_states=True)
                        except:
                            outputs = model.distilbert(input_ids, attention_mask=attention_masks, return_dict=True, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            if pooling == 'cls':
                out_features = hidden_states[-1][:,0,:]
            elif pooling == 'first_last_avg':
                out_features = (hidden_states[-1] + hidden_states[1]).mean(dim=1)
            elif pooling == 'last_avg':
                out_features = (hidden_states[-1]).mean(dim=1)
            elif pooling == 'last2avg':
                out_features = (hidden_states[-1] + hidden_states[-2]).mean(dim=1)
            else:
                raise Exception("unknown pooling way: {}".format(pooling))

            output_features = [out_features]
            
            output_features = [feat.reshape(feat.shape[0],-1) for feat in output_features]

        # compute Mahalanobis score
        gaussian_score = 0
        batch_scores = []
        for layer_index in range(1):
            for i in range(num_classes):
                batch_sample_mean = sample_mean[layer_index][i]
                feat = output_features[layer_index]
                zero_f = feat.data - batch_sample_mean
                if distance_metrics == 'maha':
                    term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
                elif distance_metrics == 'euclid':
                    term_gau = -0.5*torch.mm(zero_f, zero_f.t()).diag()
                elif distance_metrics == 'manhattan':
                    term_gau = -0.5* torch.sum(torch.abs(zero_f), dim=1)
                elif distance_metrics == 'cosine':
                    tmp_scores = []
                    for j in range(feat.size(0)):
                        tmp_scores.append(CosineSimilarity()(feat[j].reshape(1,-1), batch_sample_mean.reshape(1,-1)))
                    term_gau = torch.Tensor(tmp_scores)
                else:
                    raise NotImplementedError
                if i == 0:
                    gaussian_score = term_gau.view(-1,1)
                else:
                    gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)
            gaussian_score, _ = torch.max(gaussian_score, dim=1)         
            batch_scores.append(gaussian_score.cpu().numpy())
        gaussian_score = batch_scores[-1] # The last layer
        #gaussian_score = np.mean(batch_scores, axis=0) # Average layer ensemble
        for i in range(1):
            layer_scores[i].append(batch_scores[i])
        scores.append(gaussian_score)
    scores = np.concatenate(scores)
    for i in range(1):
        layer_scores[i] = np.concatenate(layer_scores[i])
    return scores