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

'''
def create_embedding_matrix(word_index, embedding_dict, dimension):
    embedding_matrix = np.zeros((len(word_index), dimension))
    for word, index in word_index.items():
        if word in embedding_dict:
            embedding_matrix[index] = embedding_dict[word]
    return embedding_matrix
embedding_texts = [
    'concession', 'contrast', 'cause', 'pragmatic', 'alternative', 'conjunction', 'instantiation', 'list',
    'restatement', 'asynchronous', 'synchrony']
# print("The original list : " + str(embedding_texts))
dict_word_index = {val: idx for idx, val in enumerate(embedding_texts)}
# print(dict_word_index)
embedding_matrix = create_embedding_matrix(dict_word_index, embedding_dict=glove_embedding, dimension=300)
vocab_size = embedding_matrix.shape[0]
vector_size = embedding_matrix.shape[1]
logger.info("the size of vocab must be equal to the size of the number of relation %s", vocab_size)
logger.info("the dimension of the embedding vector %s", vector_size)
print('###########word2vec embeddings##################')
word2vec = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
embedding_texts = [
    'concession', 'contrast', 'cause', 'pragmatic', 'alternative', 'conjunction', 'instantiation', 'list',
    'restatement', 'asynchronous', 'synchrony']
# print("The original list : " + str(embedding_texts))
dict_word_index = {val: idx for idx, val in enumerate(embedding_texts)}
embedding_matrix = np.zeros((len(dict_word_index), 300))
# print(dict_word_index)
for word in dict_word_index:
    embedding_matrix[dict_word_index[word]] = word2vec.word_vec(word)
vocab_size = embedding_matrix.shape[0]
vector_size = embedding_matrix.shape[1]
# logger.info("the size of vocab must be equal to the size of the number of relation %s", vocab_size)
# logger.info("the dimension of the embedding vector %s", vector_size)
'''

##################################code for vector space addition #######################

embedding_matrix = np.zeros((11, 300))

embedding_matrix[0] = glove_embedding['despite'] + glove_embedding['even'] + glove_embedding['though'] + \
                      glove_embedding['however']  # concession

embedding_matrix[1] = glove_embedding['contrast'] + glove_embedding['comparison'] + glove_embedding['but']  # contrast

embedding_matrix[2] = glove_embedding['because'] + glove_embedding['result'] + glove_embedding['therefore']  # Cause

embedding_matrix[3] = glove_embedding['considering'] + glove_embedding['accordingly']  # Pragmatic cause

embedding_matrix[4] = glove_embedding['alternatively'] + glove_embedding['instead'] + glove_embedding[
    'rather']  # Alternative

embedding_matrix[5] = glove_embedding['addition'] + glove_embedding['also'] + glove_embedding[
    'furthermore']  # Conjunction

embedding_matrix[6] = glove_embedding['example'] + glove_embedding['instance']  # Instantiation

embedding_matrix[7] = glove_embedding['firstly'] + glove_embedding['secondly'] + glove_embedding['thirdly']  # List

embedding_matrix[8] = glove_embedding['other'] + glove_embedding['words']  + glove_embedding[
    'means']  # Restatement

embedding_matrix[9] = glove_embedding['subsequently'] + glove_embedding['afterwards'] + glove_embedding[
    'previously']  # Asynchronous

embedding_matrix[10] = glove_embedding['same'] + glove_embedding['time'] + glove_embedding['simultaneously'] + \
                       glove_embedding['meanwhile']  # Synchrony

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
