import os, sys
import pandas as pd
import numpy as np 
import torch
from utils.mydata import load_pickle
from Models.k_bert.atom_embedding_generator import bert_atom_embedding

root = os.getcwd()
data_dir = os.path.join(root, 'data_collect/')
drug_std_dir =  os.path.join(data_dir, 'unify/drug_std/')
atom_pad_dict = load_pickle(data_dir + 'unify/drug_std/atom_pad.pkl')

def drug_id2encoding(smiles_list):
    drug_features_list = []
    for i, smiles in enumerate(smiles_list):
        # print("{}/{}".format(i+1, len(smiles_list)))
        try:
            h_global, g_atom = bert_atom_embedding(smiles, pretrain_model='pretrain_k_bert_epoch_7.pth')
            # drug_features_list.append(h_global)
            drug_features_list.append(g_atom)
        except:
            drug_features_list.append(['NaN' for x in range(768)])
            print('drug_encoding_failed:{}'.format(i+1))
    return drug_features_list

def unreg_atomf_list2tensor(f_list):
    # print('f_list', f_list)
    max_len = max(len(sub_list) for sub_list in f_list)
    f_list_pad = []
    valid_len_list = []
    for sub_list in f_list:
        valid_l = len(sub_list)
        valid_len_list.append(valid_l)
        sub_pad = np.pad(sub_list,((0, max_len-sub_list.shape[0]),(0,0)), constant_values=0)
        f_list_pad.append(sub_pad)

    f_bs = torch.stack([torch.tensor(arr) for arr in f_list_pad])
    return f_bs, max_len, valid_len_list


def unreg_atomf_list2tensor_pred(f, max_len=96):
    # 进行padding操作，是对一个batchlist里的数据
    valid_l = len(f)
    f_pad = np.pad(f,((0, max_len-f.shape[0]),(0,0)), constant_values=0)
    return f_pad, max_len, valid_l

def kbert_file(smiles_list):
    drug_df = pd.read_csv(drug_std_dir + 'drug_smiles_k_bert.csv', index_col='drug_id')
    # 一定要drop_dup！！！
    drug_encode_df = drug_df.drop_duplicates(subset='smiles')
    global_feature_list = []
    for i, smiles in enumerate(smiles_list):
        # print('encoding:{}/{}'.format(i, len(smiles_list)))
        features = drug_encode_df.loc[drug_encode_df['smiles']== smiles, 'pretrain_feature_1':'pretrain_feature_768'].values.ravel()
        # print(features.ravel())
        global_feature_list.append(features)
    # print('feature_list{}'.format(len(feature_list)))
    return global_feature_list
    
def kbert_atom_file(smiles_list):
    drug_atom_f_dict = load_pickle(drug_std_dir + 'atom_features_dict.pkl')
    feature_list = []
    for i, smiles in enumerate(smiles_list):
        drug_a_f = drug_atom_f_dict[smiles]
        feature_list.append(drug_a_f)
    return feature_list

def pad_atom_file(drug_id_list):
    atom_feature_list = []
    for i, drug_id in enumerate(drug_id_list):
        # print(drug_id)
        drug_id = drug_id.item()
        drug_a_f = atom_pad_dict[drug_id]
        atom_feature_list.append(drug_a_f)
    return atom_feature_list

def  kbert(drug_id_list):
    drug_encode_df = pd.read_csv(drug_std_dir + 'drug_smiles_k_bert.csv')
    feature_list = []
    for i, drug_id in enumerate(drug_id_list):
        drug_id = drug_id.item()
        drug_global_f = drug_encode_df.loc[drug_encode_df['drug_id']==drug_id, 'pretrain_feature_1':'pretrain_feature_768'].values.ravel()
        drug_global_f =drug_global_f.reshape(-1, drug_global_f.shape[0])
        drug_a_f = atom_pad_dict[drug_id]
        # 第0位是global token
        f = np.vstack((drug_global_f, drug_a_f))
        # print(f.shape)
        feature_list.append(f)
    return feature_list

def encoder_D_pred(smiles):
    h_global, h_atom = bert_atom_embedding(smiles)
    # print('h_global', h_global)
    f_pad, max_len, valid_lens = unreg_atomf_list2tensor_pred(h_atom)
    valid_lenD_list = [valid_lens]*4
    valid_lens = torch.tensor(valid_lenD_list)
    encode_D_pred = np.vstack((h_global, f_pad))
    encode_D_pred_list = [encode_D_pred]*4
    encode_D_pred = torch.stack([torch.tensor(arr) for arr in list(encode_D_pred_list)])
    return encode_D_pred, valid_lens