import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from collections import OrderedDict
from encoder import *
from layer import *
import pandas as pd
from gensim.models import KeyedVectors

print("glove embeddings parent addition glove.42B.300d.txt")
logging.info("glove parent addition")
glove = pd.read_csv('glove.42B.300d.txt', sep=" ", quoting=3, header=None, index_col=0)
glove_embedding = {key: val.values for key, val in glove.T.items()}


##################################code for vector space addition #######################

embedding_matrix = np.zeros((11, 300))

embedding_matrix[0] = glove_embedding['comparison'] + glove_embedding['concession']

embedding_matrix[1] = glove_embedding['comparison'] + glove_embedding['contrast']

embedding_matrix[2] = glove_embedding['contingency'] + glove_embedding['cause']

embedding_matrix[3] = glove_embedding['contingency'] + glove_embedding['pragmatic']

embedding_matrix[4] = glove_embedding['expansion'] + glove_embedding['alternative']

embedding_matrix[5] = glove_embedding['expansion'] + glove_embedding['conjunction']

embedding_matrix[6] = glove_embedding['expansion'] + glove_embedding['instantiation']

embedding_matrix[7] = glove_embedding['expansion'] + glove_embedding['list']

embedding_matrix[8] = glove_embedding['expansion'] + glove_embedding['restatement']

embedding_matrix[9] = glove_embedding['temporal'] + glove_embedding['asynchronous']

embedding_matrix[10] = glove_embedding['temporal'] + glove_embedding['synchrony']

vocab_size = embedding_matrix.shape[0]
vector_size = embedding_matrix.shape[1]


class BMGFModel(nn.Module):
    def __init__(self, **kw):
        super(BMGFModel, self).__init__()
        max_arg = kw.get("max_arg", 512)
        encoder = kw.get("encoder", "roberta")
        hidden_dim = kw.get("hidden_dim", 128)
        num_perspectives = kw.get("num_perspectives", 16)
        dropout = kw.get("dropout", 0.2)
        activation = kw.get("activation", "relu")
        num_rels = kw.get("num_rels", 4)
        num_filters = kw.get("num_filters", 64)
        act_layer = map_activation_str_to_layer(activation)
        setting = kw.get("setting", 0)

        self.drop = Dropout(dropout)
        if encoder == "lstm":
            self.encoder = LSTMEncoder(**kw)
        elif encoder == "bert":
            self.encoder = BERTEncoder(num_segments=2, max_len=max_arg * 2, **kw)
        elif encoder == "roberta":
            self.encoder = ROBERTAEncoder(num_segments=2, max_len=max_arg * 2, **kw)
        else:
            raise NotImplementedError("Error: encoder=%s is not supported now." % (encoder))

        self.bimpm = BiMpmMatching(
            hidden_dim=self.encoder.get_output_dim(),
            num_perspectives=num_perspectives)
        output_dim = self.encoder.get_output_dim() + self.bimpm.get_output_dim()

        self.gated_attn_layer = GatedMultiHeadAttn(
            query_dim=output_dim,
            key_dim=output_dim,
            value_dim=output_dim,
            hidden_dim=hidden_dim,
            num_head=num_perspectives,
            dropatt=dropout,
            act_func="softmax",
            add_zero_attn=False,
            pre_lnorm=False,
            post_lnorm=False)

        self.conv_layer = CnnHighway(
            input_dim=self.gated_attn_layer.get_output_dim(),
            output_dim=hidden_dim,
            filters=[(1, num_filters), (2, num_filters)],  # the shortest length is 2
            num_highway=1,
            activation=activation,
            layer_norm=False)

        ##njv

        # num_rels =vocab size

        self.label_emb1 = nn.Embedding(num_embeddings=vocab_size, embedding_dim=vector_size)  # (batch,11*300)
        self.label_emb1.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.label_emb1.weight.requires_grad = False
        self.label_emb1_1 = nn.Linear(in_features=vector_size, out_features=276)  # (batch, 11*276)
        self.label_emb2 = nn.Linear(in_features=276, out_features=2 * hidden_dim)  # (batch, 11*256)
        self.label_sim_dict = nn.Linear(in_features=num_rels, out_features=num_rels)
        ##njv

        self.fc_layer = FullyConnectedLayer(2 * hidden_dim, hidden_dim, num_rels)

    def set_finetune(self, finetune):
        for param in self.parameters():
            param.requires_grad = True
        self.encoder.set_finetune(finetune)

    def forward(self, arg1, arg2, arg1_mask=None, arg2_mask=None, encode_pair=True, lcm_setting=True, X_Label=None):
        lcm_setting = lcm_setting
        X_Label = X_Label

        if encode_pair:
            arg1_feats, arg2_feats, arg1_mask, arg2_mask = self.encoder.forward_pair(arg1, arg2, arg1_mask, arg2_mask)
        else:
            arg1_feats, arg1_mask = self.encoder.forward_single(arg1, arg1_mask)
            arg2_feats, arg2_mask = self.encoder.forward_single(arg2, arg2_mask)
        arg1_feats, arg2_feats = self.drop(arg1_feats), self.drop(arg2_feats)

        arg1_matched_feats, arg2_matched_feats = self.bimpm(
            arg1_feats, arg1_mask, arg2_feats, arg2_mask)
        arg1_matched_feats = torch.cat(arg1_matched_feats, dim=2)
        arg2_matched_feats = torch.cat(arg2_matched_feats, dim=2)
        arg1_matched_feats, arg2_matched_feats = self.drop(arg1_matched_feats), self.drop(arg2_matched_feats)

        arg1_self_attned_feats = torch.cat([arg1_feats, arg1_matched_feats], dim=2)
        arg2_self_attned_feats = torch.cat([arg2_feats, arg2_matched_feats], dim=2)
        arg1_self_attned_feats = self.gated_attn_layer(
            arg1_self_attned_feats, arg1_self_attned_feats, arg1_self_attned_feats, attn_mask=arg1_mask)
        arg2_self_attned_feats = self.gated_attn_layer(
            arg2_self_attned_feats, arg2_self_attned_feats, arg2_self_attned_feats, attn_mask=arg2_mask)
        arg1_self_attned_feats, arg2_self_attned_feats = self.drop(arg1_self_attned_feats), self.drop(
            arg2_self_attned_feats)

        arg1_conv = self.conv_layer(arg1_self_attned_feats, arg1_mask)
        arg2_conv = self.conv_layer(arg2_self_attned_feats, arg2_mask)
        arg1_conv, arg2_conv = self.drop(arg1_conv), self.drop(arg2_conv)

        if lcm_setting:
            X_Label = self.label_emb1(X_Label)
            X_Label = self.label_emb1_1(X_Label)
            X_Label = F.tanh(X_Label)
            X_Label = self.label_emb2(X_Label)
            X_Label = F.tanh(X_Label)
            doc_product = torch.einsum('ijk,ik->ij', X_Label, torch.cat([arg1_conv, arg2_conv], dim=1))
            label_sim_dict = self.label_sim_dict(doc_product)

        output = self.fc_layer(torch.cat([arg1_conv, arg2_conv], dim=1))

        if lcm_setting:
            return output, label_sim_dict

        if not lcm_setting:
            return output