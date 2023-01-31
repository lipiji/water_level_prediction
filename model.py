import sys
import torch
from torch import nn
import torch.nn.functional as F
import random
import os
import argparse
import numpy as np
from transformer import gelu, LayerNorm, TransformerLayer, Embedding, SinusoidalPositionalEmbedding, SelfAttentionMask


class mTransformer(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, layers, batch_size, dropout, device):
        super(mTransformer, self).__init__()
        self.embedding_dim = embedding_dim
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.device = device

        self.dropout = dropout
        self.pos_embed = SinusoidalPositionalEmbedding(self.hidden_dim, device=self.device)

        self.mlayers = nn.ModuleList()
        for i in range(self.layers):
            self.mlayers.append(TransformerLayer(self.hidden_dim, self.hidden_dim, 8, self.dropout))
        self.emb_layer_norm = LayerNorm(self.hidden_dim)
        self.fc = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.one_more = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.one_more_layer_norm = LayerNorm(self.hidden_dim)
        self.attn_mask = SelfAttentionMask(device=self.device)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.fc.bias, 0.)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.constant_(self.one_more.bias, 0.)
        nn.init.normal_(self.one_more.weight, std=0.02)

    def forward(self, inp):
        seq_len, bsz, _ = inp.size()
        self_attn_mask = self.attn_mask(seq_len)
        x = self.fc(inp) + self.pos_embed(inp)
        x = self.emb_layer_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        for layer in self.mlayers:
            x, _ ,_ = layer(x, self_attn_mask = self_attn_mask)
        x = self.one_more_layer_norm(gelu(self.one_more(x)))
        return x

class mMLP(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, layers, batch_size, dropout, device):
        super(mMLP, self).__init__()
        self.embedding_dim = embedding_dim
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.device = device
        self.dropout = nn.Dropout(dropout)

        self.input_fc = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.mlayers = nn.ModuleList()
        for i in range(self.layers):
            self.mlayers.append(nn.Linear(self.hidden_dim, self.hidden_dim))

    def forward(self, x):
        h = F.relu(self.input_fc(x))
        for mlayer in self.mlayers:
            h = F.relu(mlayer(h))
        return h


class mLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, layers, batch_size, dropout, device):
        super(mLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.device = device

        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=self.layers)

    def init_hidden(self, batch_size):
        return (torch.randn(self.layers, batch_size, self.hidden_dim).to(self.device),
                torch.randn(self.layers, batch_size, self.hidden_dim).to(self.device))

    def forward(self, x):
        outputs, (ht, ct) = self.lstm(x, self.init_hidden(x.size(1)))
        #x = self.dropout_layer(ht[-1])
        return outputs #(seq_len x batch_size x hidden_dim)


class Model(nn.Module):
    def __init__(self, model_name, num_class, embedding_dim, hidden_dim, layers, batch_size, dropout, device):
        super(Model, self).__init__()
        self.dropout = dropout
        self.device = device
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.num_class = num_class
        self.criterion = nn.MSELoss()

        if "lstm" in model_name.lower():
            print(model_name)
            self.model = mLSTM(self.embedding_dim, self.hidden_dim, self.layers, self.batch_size, self.dropout, self.device)
        elif "transformer" in model_name.lower():
            print(model_name)
            self.model = mTransformer(self.embedding_dim, self.hidden_dim, self.layers, self.batch_size, self.dropout, self.device)
        else:
            print(model_name)
            self.model = mMLP(self.embedding_dim, self.hidden_dim, self.layers, self.batch_size, self.dropout, self.device)
        
        self.fc = nn.Linear(self.hidden_dim, self.num_class)
    
    def inference(self, x):
        # size of data is [batch_max_len, batch_size]
        y = self.model(x) #(batch_size x hidden_dim)
        y = self.fc(y)
        return y

    def forward(self, x, y):
        y_ = self.inference(x)
        loss = self.criterion(y_, y)
        return y_, loss


