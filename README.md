# IDRR_label_info
Master thesis codebase

Refactored code base for enhancing label representations through the utilization of alternative label representations



## Description
This repository contains code for enriching label representations using alternate label representations. The code base is designed to preprocess and train on the PDTB (Penn Discourse Treebank) dataset. The following instructions will guide you on setting up the project.

## Installation

1. Install the dependencies from `requirements.txt` file:

    pip install -r requirements.txt


2. Download the `pdtb2.0.csv` file and place it in the `data/pdtb` folder. Please note that this file is not publicly available.

## Preprocessing

To preprocess the PDTB dataset, follow these steps:

1. Open a terminal and navigate to the project directory.

2. Run the following commands for preprocessing:

```shell
python preprocess_pdtb.py \
    --csv_file_path ../data/pdtb/pdtb2.csv \
    --types Implicit \
    --encoder roberta \
    --dataset_file_path ../data/pdtb/train_implicit_roberta.pt \
    --sections 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20

python preprocess_pdtb.py \
    --csv_file_path ../data/pdtb/pdtb2.csv \
    --types Implicit \
    --encoder roberta \
    --dataset_file_path ../data/pdtb/valid_implicit_roberta.pt \
    --sections 0,1

python preprocess_pdtb.py \
    --csv_file_path ../data/pdtb/pdtb2.csv \
    --types Implicit \
    --encoder roberta \
    --dataset_file_path ../data/pdtb/test_implicit_roberta.pt \
    --sections 21,22
```
    
   
## Training
To train the model, execute the following commands:

```shell
python train.py \
    --train_dataset_path ../data/pdtb/train_implicit_roberta.pt \
    --valid_dataset_path ../data/pdtb/valid_implicit_roberta.pt \
    --test_dataset_path ../data/pdtb/test_implicit_roberta.pt \
    --save_model_dir ../data/dumps \
    --num_rels 4 \
    --gpu_ids 0,1 \
    --batch_size 32 \
    --lr 0.001 \
    --weight_decay 0.0005 \
    --max_grad_norm 2.0 \
    --loss ce \
    --encoder roberta \
    --finetune type \
    --hidden_dim 128 \
    --num_perspectives 16 \
    --num_filters 64 \
    --activation leaky_relu \
    --dropout 0.2
```    
    
## Customization
1. Modify the alpha(default = 4 ) and Lcm_stop (default=100) parameters in the train.py file according to your requirements.

2. This codebase requires the GloVe embeddings 42b version to be downloaded and placed at the same level as the train.py file.

   
