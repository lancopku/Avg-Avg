## Code for OOD Detection Research in NLP

This is the official code of our EMNLP 2022 (Findings) paper [Holistic Sentence Embeddings for Better Out-of-Distribution Detection](https://aclanthology.org/2022.findings-emnlp.497/) and EACL 2023 (Findings) paper [Fine-Tuning Deteriorates General Textual Out-of-Distribution Detection by Distorting Task-Agnostic Features](https://arxiv.org/abs/2301.12715). 


This repository implements the OOD detection algorithms developed by us (Avg-Avg published at EMNLP 2022 and GNOME published at EACL 2023) along with the following baselines:

| Algorithm |  Paper | 
| --------- | ------ | 
| MC Dropout | [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](http://proceedings.mlr.press/v48/gal16.pdf) (ICML 2016) |
| Maximum Softmax Probability | [A Baseline For Detecting Misclassfied and Out-of-Distribution Samples in Nerual Networks](https://arxiv.org/pdf/1610.02136.pdf)  (ICML 2017) |
| ODIN | [Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks](https://openreview.net/forum?id=H1VGkIxRZ) (ICLR 2018) |
| Maha Distance |  [A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks](https://arxiv.org/abs/1807.03888) (NIPS 2018) |
| LOF | [Deep Unknown Intent Detection with Margin Loss](https://aclanthology.org/P19-1548.pdf) (ACL 2019) | 
| Energy Score | [Energy-based Out-of-distribution Detection](https://arxiv.org/abs/2010.03759) (NIPS 2020) | 
|   ContraOOD |  [Contrastive Out-of-Distribution Detection for Pretrained Transformers](https://aclanthology.org/2021.emnlp-main.84.pdf) (EMNLP 2021) |
| KNN Distance |   [Out-of-Distribution Detection with Deep Nearest Neighbors](https://proceedings.mlr.press/v162/sun22d/sun22d.pdf) (ICML 2022) | 
| D2U | [D2U: Distance-to-Uniform Learning for Out-of-Scope Detection](https://aclanthology.org/2022.naacl-main.152.pdf) (NAACL 2022) |

 


### Requirements
Python: 3.7.9

To install the dependencies, run
<pre/>pip install -r requirements.txt</pre> 

For the datasets used in our papers, please download the `nlp_ood_datasets.zip` file from this [Google Drive link](https://drive.google.com/file/d/1QeEF_nGV-RqNbcsm0hjAhyVj7g2eLJas/view?usp=sharing) and unzip it under the root directory (a 'dataset' directory will be created).

### Training


#### Vanilla Training

Vanilla training with cross-entroy loss:
```
python train.py --model roberta-base --output_dir <YOUR_DIR> --seed 13 --dataset sst-2 --log_file <YOUR_LOG_FILE> --lr 2e-5 --epochs 5 --batch_size 16 
```

#### Supervised Contrastive Training
Add `--loss_type scl` or `--loss_type margin` to use the supervised contrastive auxilliray targets proposed in [Contrastive Out-of-Distribution Detection for Pretrained Transformers](https://aclanthology.org/2021.emnlp-main.84.pdf):
```
python train.py --model roberta-base --loss scl --output_dir <YOUR_DIR> --seed 13 --dataset sst-2 --log_file <YOUR_LOG_FILE> --lr 2e-5 --epochs 5 --batch_size 16 
```

#### Training with RecAdam Regularization

Add '--optimizer recadam' to use the [RecAdam](https://github.com/Sanyuan-Chen/RecAdam) optimizer:
```
python train.py --model roberta-base --optimizer recadam --output_dir <YOUR_DIR> --seed 13 --dataset sst-2 --log_file <YOUR_LOG_FILE> --lr 2e-5 --epochs 5 --batch_size 16 
```

### Evaluation for OOD Detection

#### Avg-Avg, GNOME, Maha, and KNN

##### Feature Extraction

Extract features from a fine-tuned model first:
```
python extract_full_features.py --dataset sst-2 --ood_datasets 20news,trec,wmt16 --output_dir <YOUR_FT_DIR> --model roberta-base --pretrained_model <PATH_TO_FINETUNED_MODEL>
```

GNOME addtional needs pre-trained features:
```
python extract_full_features.py --dataset sst-2 --ood_datasets 20news,trec,wmt16 --output_dir <YOUR_PRE_DIR> --model roberta-base
```

##### Test

**Maha** with `last-cls` pooled features:
```
python ood_test_embedding.py --dataset sst-2 --ood_datasets 20news,trec,wmt16 --inpur_dir <YOUR_FT_DIR> --token_pooling cls --layer_pooling last
```

**Avg-Avg (Ours, EMNLP 2022)**, i.e., Maha with `avg-avg` pooled features:
```
python ood_test_embedding.py --dataset sst-2 --ood_datasets 20news,trec,wmt16 --inpur_dir <YOUR_FT_DIR> --token_pooling avg --layer_pooling avg
```

**KNN** (with the pooling way that you choose, `cls-last` by default):
```
python ood_test_embedding_knn.py --dataset sst-2 --ood_datasets 20news,trec,wmt16 --inpur_dir <YOUR_FEARTURE_DIR>  --token_pooling <cls/avg> --layer_pooling <last/avg/a list of layer indexes like 1,2,11,12>
```

**GNOME (Ours, EACL 2023)** (`--std` for score normalization, `--ensemble_way mean/min` for choosing the aggregator `mean` or `min`):
```
python ood_test_embedding_gnome.py --dataset sst-2 --ood_datasets 20news,trec,wmt16 --ft_dir <YOUR_FT_DIR>  --pre_dir <YOUR_PRE_DIR> --std --ensemble_way mean
```

**Note**: Our algorithms Avg-Avg and GNOME are tested on the features extraced from the model trained with the vanilla entropy loss. For reproducing the results of [Contrastive Out-of-Distribution Detection for Pretrained Transformers](https://aclanthology.org/2021.emnlp-main.84.pdf), just use the model trained with contrastive targets to extract features.

#### Other Algorithms

To test MSP (base) / Energy (energy) / D2U (d2u) / ODIN (odin) / LOF (lof) / MC Dropout (mc), just specify the method by passed it to the `ood_method` argument in the `test.py`:
```
python test.py --dataset sst-2 --ood_datasets 20news,trec,wmt16  --model roberta-base --pretraiend_model <PATH_TO_FINETUNED_MODEL> --ood_method base/energy/d2u/odin/lof/mc
```

### Citation

If you find this repository to be useful for your research, please consider citing.
<pre>
@inproceedings{chen-etal-2022-holistic,
    title = "Holistic Sentence Embeddings for Better Out-of-Distribution Detection",
    author = "Chen, Sishuo  and
      Bi, Xiaohan  and
      Gao, Rundong  and
      Sun, Xu",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-emnlp.497",
    pages = "6676--6686"
}
@misc{https://doi.org/10.48550/arxiv.2301.12715,
  doi = {10.48550/ARXIV.2301.12715},
  url = {https://arxiv.org/abs/2301.12715},
  author = {Chen, Sishuo and Yang, Wenkai and Bi, Xiaohan and Sun, Xu},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Fine-Tuning Deteriorates General Textual Out-of-Distribution Detection by Distorting Task-Agnostic Features},
  publisher = {arXiv},
  year = {2023},
  copyright = {Creative Commons Attribution 4.0 International}
}
</pre>


### Acknowledgement
This repository relies on resources from <a href="https://github.com/megvii-research/FSSD_OoD_Detection">FSSD_OoD_Detection</a>, <a href="https://github.com/huggingface/transformers">Huggingface Transformers</a>, and [RecAdam](https://github.com/Sanyuan-Chen/RecAdam). We thank the original authors for their open-sourcing.

