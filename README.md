# Data and Code for "DECOR: Degree-Corrected Social Graph Refinement for Fake News Detection"

This repo contains the data and code for the following paper: 

Jiaying Wu, Bryan Hooi. DECOR: Degree-Corrected Social Graph Refinement for Fake News Detection, ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD) 2023. [![arXiv](https://img.shields.io/badge/arXiv-2307.00077-b31b1b.svg)](https://arxiv.org/abs/2307.00077)

## Abstract

Recent efforts in fake news detection have witnessed a surge of interest in using graph neural networks (GNNs) to exploit rich social context. Existing studies generally leverage fixed graph structures, assuming that the graphs accurately represent the related socialengagements. However, edge noise remains a critical challenge in real-world graphs, as training on suboptimal structures can severely limit the expressiveness of GNNs. Despite initial efforts in graph structure learning (GSL), prior works often leverage node features to update edge weights, resulting in heavy computational costs that hinder the methods’ applicability to large-scale social graphs. In this work, we approach the fake news detection problem with a novel aspect of social graph refinement. We find that the degrees of news article nodes exhibit distinctive patterns, which are indicative of news veracity. Guided by this, we propose DECOR, a novel application of Degree-Corrected Stochastic Blockmodels to the fake news detection problem. Specifically, we encapsulate our empirical observations into a lightweight social graph refinement component that iteratively updates the edge weights via a learnable degree correction mask, which allows for joint optimization with a GNN-based detector. Extensive experiments on two real-world benchmarks validate the effectiveness and efficiency of DECOR.

## Requirements
```
python==3.8.13
numpy==1.22.4
torch==1.10.0+cu111
torch-geometric==2.0.4
torch-scatter==2.0.9
torch-sparse==0.6.13
transformers==4.13.0
```


## Data

Our work is based on the `PolitiFact` and `GossipCop` datasets from the [FakeNewsNet benchmark](https://github.com/KaiDMML/FakeNewsNet).

Extract the files in `data.tar.gz` to obtain an unzipped `data/` folder. The resultant `data/` should contain four folders: `news_features/`, `user_news_graph/`, `social_context_raw/` and `temp_splits/`. 

**News Article Features**

The .pkl files under `data/news_features/` contain the 768-dimensional BERT features of news articles. The features are extracted via a frozen BERT model from the Transformers library, with version name `bert-base-uncased` and `max_length=512`.

**Social Context**

The .pkl files under `data/user_news_graph/` contain the social context of each dataset under `dict` format. Specifically, the dictionary contains the following keys and values:

* `A_un`: the user engagement matrix of size [num_users, num_news]. Element [i,j] in the matrix represents the number of interactions between active user `u_i` (defined as users with at least 3 engagements) and news article `p_j`, in terms of the user's reposts of news article on social media. If an article receive no engagements from active users, it is assigned value of 1 with an unique index (this creates a self-loop for the article when we construct the news engagement graph). 
* `uid_dict`: contains the mapping from users' Twitter IDs to matrix indices, under the format `{user_id:index}`.
* `sid_dict`: contains the mapping from FakeNewsNet news IDs to matrix indices, under the format `{news_id:index}`.


We provide the raw user-news engagement records used to construct the above-mentioned user engagement matrix from scratch, at `data/social_context_raw/[dataset_name]_raw.csv`. There, each line is given as: `[sid,label,tid,uid]`, meaning that user `uid` has reposted news articles `sid` of veracity label (0: real; 1: fake), and the repost has Tweet ID of `tid`.


**Data Splits** 

The .pkl files under `data/temp_splits/` contain the dataset splits under `dict` format. Specifically, the dictionary contains the following keys and values:

* `train_mask`: mask for training indices on the news engagement graph.
* `test_mask`: mask for test indices on the news engagement graph.
* `y_train`: training labels. (0: real; 1: fake)
* `y_test`: test labels. (0: real; 1: fake)
* `sid_dict`: contains the mapping from FakeNewsNet news IDs to matrix indices, under the format `{news_id:index}`, same as the `sid_dict` in social context.

We use the first user-news engagement of each news article in the raw data to represent the article's timestamp, as the retrieved metadata for news articles do not necessarily contain publication dates. Under each class, the training samples are of earlier timestamps than test samples of the same class.

**FakeNewsNet Benchmark & Obtaining Auxiliary Features** 

The data used in our work are from the FakeNewsNet benchmark. To retrieve auxiliary features related to social context, please follow the instructions and scripts given in the [FakeNewsNet GitHub repo](https://github.com/KaiDMML/FakeNewsNet).

## Run DECOR

Start training with the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python src/[base_gnn]_decor.py --dataset_name [dataset_name] 
```

`[base_gnn]`: gcn / gin / graphconv

`[dataset_name]`: politifact / gossipcop

Based on empirical results, we suggest setting the base GNN as either `gcn` or `gin` to yield the best performance. The experiment logs for DECOR-GCN and DECOR-GIN are placed under `logs/logs_archive`.

Results will be saved under the `logs/` directory. 

## Contact

jiayingwu [at] u.nus.edu

## Citation

If you find this repo or our work useful for your research, please consider citing our paper

```
TBD
```