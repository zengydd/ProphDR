import os, sys
sys.path.append("..")
import numpy as np
import pandas as pd 
import math
import random
import pickle
import rdkit
import os, sys
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.autograd import Variable
from torch.autograd import Variable
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, roc_curve
from utils.load import load_pickle
from d2l import torch as d2l
from scipy.stats import pearsonr,spearmanr
from utils.mydata import mydata, dataset_split
# from .drug_encode import drug_pretrain_kbert
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import SequentialSampler
from prettytable import PrettyTable
from subword_nmt.apply_bpe import BPE


TCGA_list = ['LUAD', 'BRCA', 'COREAD', 'SCLC', 'SKCM', 'ESCA', 'OV',
            'PAAD', 'DLBC', 'GBM', 'HNSC', 'LAML', 'STAD', 'NB', 'KIRC', 'BLCA',
            'MM', 'LIHC', 'LUSC', 'ALL', 'THCA', 'CESC', 'LGG', 'UCEC', 'LCML',
            'MESO', 'PRAD', 'MB', 'CLL', 'ACC']

#  文件路径
root = os.getcwd()
data_dir = os.path.join(root, 'data_collect/')
unify_dir = os.path.join(root, 'data_collect/unify/')
omics_std_dir = os.path.join(unify_dir, 'omics_std/')
drug_std_dir =  os.path.join(unify_dir, 'drug_std/')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数 num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数, num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)
    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    # 返回X的形状是（batch_size，num_heads，查询的个数，num_hiddens/num_heads）
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    # permute后的形状是（batch_size，查询的个数，num_heads，num_hiddens/num_heads）
    X = X.permute(0, 2, 1, 3)
    # X.shape[0]：batch_size，X.shape[1]：查询的个数
    # 返回的X的形状是（batch_size，查询的个数，num_hiddens）
    return X.reshape(X.shape[0], X.shape[1], -1)


def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis."""
    # `X`: 3D tensor, `valid_lens`: 1D or 2D tensor
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        valid_lens = valid_lens.to(device)
        a = X.reshape(-1, shape[-1]).to(device)
        X = d2l.sequence_mask(a, valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

# 残差连接和层规范化
class AddNorm_Q(nn.Module):
    """残差连接后进行层规范化"""
    def __init__(self, normalized_shape, query_size, num_hiddens, dropout, **kwargs):
        super(AddNorm_Q, self).__init__(**kwargs)
        self.dropout = nn.Dropout(0.1)
        # self.dropout2 = nn.Dropout(0.2)
        # Y 是attn score， x是q
        self.q_l = nn.Linear(query_size, num_hiddens)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, q, attn):
        q = self.q_l(q)
        return self.ln(self.dropout(attn) + q)

# Q:G, K,V: D  
class DotProductAttention_G(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention_G, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        attn_w = self.attention_weights
        attn_score = torch.bmm(self.dropout(self.attention_weights), values)
        return attn_score, attn_w

class cross_MultiHeadAttention_G(nn.Module):
    def __init__(self, query_size, key_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(cross_MultiHeadAttention_G, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention_G(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
    def forward(self, q, k, v, valid_lens):
        # print('q', q.shape)
        # print('k', k.shape)
        # print('v', v.shape)
        Q = self.W_q(q)
        K = self.W_k(k)
        V = self.W_v(v)
        # print('Q', Q.shape)
        # print('K', K.shape)
        # print('V', V.shape)
        queries = transpose_qkv(Q, self.num_heads)
        keys = transpose_qkv(K, self.num_heads)
        values = transpose_qkv(V, self.num_heads)

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
        # output的形状:(batch_size*num_heads，查询的个数, num_hiddens/num_heads)
        attn_score, attn_w = self.attention(queries, keys, values, valid_lens)
        # print('attn_score', attn_score.shape)
        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(attn_score, self.num_heads)
        # attn_w = transpose_output(attn_w, self.num_heads)
        attn_output = self.W_o(output_concat)
        # print("attn_output", attn_output)
        return attn_output, attn_w

class cross_EncoderBlock_G(nn.Module):
    """Transformer编码器块"""
    def __init__(self, query_size, key_size, value_size, num_hiddens,
                 num_heads, norm_shape,
                 dropout=0.1, bias=False, **kwargs):
        super(cross_EncoderBlock_G, self).__init__(**kwargs)

        self.cross_attention = cross_MultiHeadAttention_G(
            query_size, key_size, value_size, num_hiddens, num_heads, dropout, bias)
        self.addnorm_q = AddNorm_Q(norm_shape, query_size, num_hiddens, dropout)
        self.linear = nn.Linear(num_hiddens, num_hiddens)
    def forward(self, q, k, v, valid_lens):
        attn_output, attn_w = self.cross_attention(q, k, v, valid_lens)

        out = self.addnorm_q(q, attn_output)
        return out, attn_w
     
# Q:D, K,V: G     
class DotProductAttention_D(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention_D, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        scores = scores.transpose(1,2)
        attention_weights = masked_softmax(scores, valid_lens)
        self.attention_weights = attention_weights.transpose(1,2)
        attn_w = self.attention_weights
        attn_score = torch.bmm(self.dropout(self.attention_weights), values)
        return attn_score, attn_w
    
class cross_MultiHeadAttention_D(nn.Module):
    def __init__(self, query_size, key_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(cross_MultiHeadAttention_D, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention_D(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
    def forward(self, q, k, v, valid_lens):
        # print('q', q.shape)
        # print('k', k.shape)
        # print('v', v.shape)
        Q = self.W_q(q)
        K = self.W_k(k)
        V = self.W_v(v)
        # print('Q', Q.shape)
        # print('K', K.shape)
        # print('V', V.shape)
        queries = transpose_qkv(Q, self.num_heads)
        keys = transpose_qkv(K, self.num_heads)
        values = transpose_qkv(V, self.num_heads)

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
        # output的形状:(batch_size*num_heads，查询的个数, num_hiddens/num_heads)
        attn_score, attn_w = self.attention(queries, keys, values, valid_lens)
        # print('attn_score', attn_score.shape)
        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(attn_score, self.num_heads)
        attn_output = self.W_o(output_concat)
        # attn_w = transpose_output(attn_w, self.num_heads)
        # print("attn_output", attn_output)
        return attn_output, attn_w

class cross_EncoderBlock_D(nn.Module):
    """Transformer编码器块"""
    def __init__(self, query_size, key_size, value_size, num_hiddens,
                    num_heads, norm_shape,
                    dropout=0.1, bias=False, **kwargs):
        super(cross_EncoderBlock_D, self).__init__(**kwargs)
        # print('query_size', query_size)
        self.cross_attention = cross_MultiHeadAttention_D(
            query_size, key_size, value_size, num_hiddens, num_heads, dropout, bias)
        # self.norm_shape = [self.len_q, self.h_dim]
        self.addnorm_q = AddNorm_Q(norm_shape, query_size, num_hiddens, dropout)
        # self.addnorm = AddNorm(norm_shape, dropout)
        self.linear = nn.Linear(num_hiddens, num_hiddens)
    def forward(self, q, k, v, valid_lens):
        attn_output, attn_w = self.cross_attention(q, k, v, valid_lens)
        # print('attn_output', attn_output.shape)
        # print('attn_w', attn_w.shape)
        out = self.addnorm_q(q, attn_output)
        return out, attn_w


class self_MultiHeadAttn(nn.Module):
    def __init__(self, query_size, key_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(self_MultiHeadAttn, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention_G(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
    def forward(self, q, k, v, valid_lens):
        Q = self.W_q(q)
        K = self.W_k(k)
        V = self.W_v(v)
        queries = transpose_qkv(Q, self.num_heads)
        keys = transpose_qkv(K, self.num_heads)
        values = transpose_qkv(V, self.num_heads)

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
        # output的形状:(batch_size*num_heads，查询的个数, num_hiddens/num_heads)
        attn_score, attn_w = self.attention(queries, keys, values, valid_lens)
        # print('attn_score', attn_score.shape)
        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(attn_score, self.num_heads)
        attn_output = self.W_o(output_concat)
        # print("attn_output", attn_output)
        return attn_output, attn_w

class self_attn(nn.Module):
    def __init__(self, query_size, key_size, value_size, num_hiddens,
                    num_heads, norm_shape,
                    dropout=0.1, bias=False, **kwargs):
        super(self_attn, self).__init__(**kwargs)
        self.self_attention = self_MultiHeadAttn(
            query_size, key_size, value_size, num_hiddens, num_heads, dropout, bias)
        # self.norm_shape = [self.len_q, self.h_dim]
        self.addnorm_q = AddNorm_Q(norm_shape, query_size, num_hiddens, dropout)
        self.linear = nn.Linear(num_hiddens, num_hiddens)
    def forward(self, q, k, v, valid_lens):
        attn_output, attn_w = self.self_attention(q, k, v, valid_lens)
        out = self.addnorm_q(q, attn_output)
        return out, attn_w
    
     
if __name__ == '__main__':
    model_G = cross_EncoderBlock_G(128, 128, 128, 128, 4, [714, 128], dropout=0.1, bias=False)
    model_G = model_G.to(device)
    model_D = cross_EncoderBlock_D(128, 128, 128, 128, 4, [43, 128], dropout=0.1, bias=False)
    model_D = model_D.to(device)
    q = torch.randn(20, 714, 128).to(device)
    k = torch.randn(20, 43, 128).to(device)
    valid_lenD_list = [38, 33, 29, 39, 27, 22, 21, 27, 37, 35, 22, 14, 28, 27, 38, 14, 43, 33, 34, 24]
    valid_lens =  valid_lens = torch.tensor(valid_lenD_list).to(device)
    output_G = model_G(q, k, k, valid_lens)
    output_D = model_D(k, q, q, valid_lens)
    
    output = torch.cat([output_G, output_G], dim=1)
    print('output_G', output_G.shape)
    print('output_D', output_D.shape)
    print('output', output.shape)