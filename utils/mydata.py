import os, sys
import numpy as np
import pandas as pd 
import math
import torch
import rdkit
import os, sys
import pickle
import random
import torch
from torch.utils import data
from torch.autograd import Variable
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from utils.load import load_pickle

TCGA_list = ['LUAD', 'BRCA', 'COREAD', 'SCLC', 'SKCM', 'ESCA', 'OV',
            'PAAD', 'DLBC', 'GBM', 'HNSC', 'LAML', 'STAD', 'NB', 'KIRC', 'BLCA',
            'MM', 'LIHC', 'LUSC', 'ALL', 'THCA', 'CESC', 'LGG', 'UCEC', 'LCML',
            'MESO', 'PRAD', 'MB', 'CLL', 'ACC']

# 1： stat_cancer
def _stat_cancer(drug_cell_df):
    print("#" * 50)
    cancer_num = drug_cell_df['TCGA_DESC'].value_counts().shape[0]
    print('#\t 癌症类型一共有：{}'.format(cancer_num))
    min_cancer_drug = min(drug_cell_df.value_counts())
    max_cancer_drug = max(drug_cell_df['TCGA_DESC'].value_counts())
    mean_cancer_drug = np.mean(drug_cell_df['TCGA_DESC'].value_counts())
    print('#\t Cancer at least match{} drugs,\n\t at most{} drugs,\n\t AVG{}drugs'.format(
        min_cancer_drug, max_cancer_drug, mean_cancer_drug))
    return 

# 2: stat drug
def _stat_drug(drug_cell_df):
    print("#" * 50)
    # test_path = ('/mnt/d/desktop/drug_test2_df.csv')
    # drug_cell_df.to_csv(test_path)
    drug_num = drug_cell_df['DRUG_ID'].value_counts().shape[0]
    print('#\t Drugs:{}'.format(drug_num))
    min_cell = min(drug_cell_df['DRUG_ID'].value_counts())
    max_cell = max(drug_cell_df['DRUG_ID'].value_counts())
    mean_cell = np.mean(drug_cell_df['DRUG_ID'].value_counts())
    print('#\t drug at least match {}cell lines,\n\t at most{}cell lines, \n\t AVG{}cell lines'.format(
        min_cell, max_cell, mean_cell))
    return

# 3: stat_cell
def _stat_cell(drug_cell_df):
    print("#" * 50)
    cell_num = drug_cell_df['COSMIC_ID'].value_counts().shape[0]
    print('#\t Cell lines:{}'.format(cell_num))
    min_drug = min(drug_cell_df['COSMIC_ID'].value_counts())
    max_drug = max(drug_cell_df['COSMIC_ID'].value_counts())
    mean_drug = np.mean(drug_cell_df['COSMIC_ID'].value_counts())
    print('#\t Drug at least match {}cell lines,\n\t at most{}cell lines, \n\t AVG{}cell lines'.format(
        min_drug, max_drug, mean_drug))
        
def drop_unclassified(path1, path2):
    # 获取数据
    unified_res_df = pd.read_csv(path1)
    # 去除unclassified 的数据
    unified_res_df_drop_unclassified = unified_res_df[unified_res_df['TCGA_DESC'].isin(TCGA_list)]
    # unified_res_df_drop_unclassified.to_csv(path2)
    res_df = unified_res_df_drop_unclassified.reset_index(drop=True)

    print('Remove unclassified Data')
    _stat_cancer(res_df)
    _stat_drug(res_df)
    _stat_cell(res_df)
    return res_df

def dataset_split(res_df, random=4, stratify=None):
    if stratify == None:
        train_set, val_test_set = train_test_split(res_df, test_size=0.2, random_state=random)
        val_set, test_set = train_test_split(val_test_set, test_size=0.5, random_state=random)
    else:
        train_set, val_test_set = train_test_split(res_df, test_size=0.2, random_state=random, stratify=res_df[stratify])
        # print('ct', val_test_set['binary'].tolist())
        val_set, test_set = train_test_split(val_test_set, test_size=0.5, random_state=random, stratify=val_test_set[stratify])
    print('Responses:{}'.format(res_df.shape[0]))
    print('Train:{}'.format(train_set.shape[0]))
    print('Val:{}'.format(val_set.shape[0]))
    print('Test:{}'.format(test_set.shape[0]))
    print('train_DRUG:{}, val_DRUG:{}, test_DRUG:{}'.format(len(train_set['DRUG_ID'].value_counts()), len(set(val_set['DRUG_ID'])), len(set(test_set['DRUG_ID']))))
    print('train_cell:{}, val_cell:{}, test_cell:{}'.format(len(set(train_set['COSMIC_ID'])), len(set(val_set['COSMIC_ID'])), len(set(test_set['COSMIC_ID']))))
    return train_set, val_set, test_set

# 数据集划分
def dataset_split_pred_miss(res_df, random, stratify=None):
    if stratify == None:
        train, val = train_test_split(res_df, test_size=0.2, random_state=random)
        # val_set, test_set = train_test_split(val_test_set, test_size=0.5, random_state=random)
    # else:
    #     train_set, val_test_set = train_test_split(res_df, test_size=0.2, random_state=random, stratify=res_df[stratify])
    #     # print('ct', val_test_set['binary'].tolist())
    #     val_set, test_set = train_test_split(val_test_set, test_size=0.5, random_state=random, stratify=val_test_set[stratify])
    print('Responses:{}'.format(res_df.shape[0]))
    print('Train:{}'.format(train.shape[0]))
    print('Val:{}'.format(val.shape[0]))
    print('train_DRUG:{}, val_DRUG:{}'.format(len(train['DRUG_ID'].value_counts()), len(set(val['DRUG_ID']))))
    # print('train_cell:{}, val_cell:{}, test_cell:{}'.format(len(set(train_set['COSMIC_ID'])), len(set(val_set['COSMIC_ID'])), len(set(test_set['COSMIC_ID']))))
    return train, val



def set_random_seed(seed=4):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    # dgl.random.seed(seed)
    # dgl.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        
    
# 数据集划分
def dataset_split_pred_miss(res_df, random, stratify=None):
    if stratify == None:
        train, val = train_test_split(res_df, test_size=0.2, random_state=random)
        # val_set, test_set = train_test_split(val_test_set, test_size=0.5, random_state=random)
    # else:
    #     train_set, val_test_set = train_test_split(res_df, test_size=0.2, random_state=random, stratify=res_df[stratify])
    #     # print('ct', val_test_set['binary'].tolist())
    #     val_set, test_set = train_test_split(val_test_set, test_size=0.5, random_state=random, stratify=val_test_set[stratify])
    print('Responses{}'.format(res_df.shape[0]))
    print('Train:{}'.format(train.shape[0]))
    print('Val:{}'.format(val.shape[0]))
    print('train_DRUG:{}, val_DRUG:{}'.format(len(train['DRUG_ID'].value_counts()), len(set(val['DRUG_ID']))))
    # print('train_cell:{}, val_cell:{}, test_cell:{}'.format(len(set(train_set['COSMIC_ID'])), len(set(val_set['COSMIC_ID'])), len(set(test_set['COSMIC_ID']))))
    return train, val



class kfold_split(object):
    def __init__(self, res_df):
        self.res_df = res_df
        return

    # k折 Random
    def random_kfold(self, n_splits=5, shuffle=False, random=42):
        if shuffle==True:
            random_state = random
        else:
            random_state = None
        split_idx_lst = []
        kf = KFold(n_splits=5, shuffle=shuffle,  random_state= random_state)
        for train_val_index, test_index in kf.split(self.res_df.index):
            split_idx = {}
            train_index, val_index = train_test_split(train_val_index, test_size=1 / (n_splits - 1), random_state= random_state)
            split_idx['train'] = list(train_index)
            split_idx['val'] = list(val_index)
            split_idx['test'] = list(test_index)
            
            train_set = self.res_df.iloc[train_index]
            val_set = self.res_df.iloc[val_index]
            test_set = self.res_df.iloc[test_index]
            # print(train_set['DRUG_ID'].value_counts())
            print('train_size:{}, val_size:{}, test_size:{}'.format(len(train_index), len(val_index), len(test_index)))
            print('train_DRUG:{}, val_DRUG:{}, test_DRUG:{}'.format(len(train_set['DRUG_ID'].value_counts()), len(set(val_set['DRUG_ID'])), len(set(test_set['DRUG_ID']))))
            print('train_cell:{}, val_cell:{}, test_cell:{}'.format(len(set(train_set['COSMIC_ID'])), len(set(val_set['COSMIC_ID'])), len(set(test_set['COSMIC_ID']))))
            split_idx_lst.append(split_idx)
        return split_idx_lst
    
        
    # strat: tissue, tcga, pathway
    def strat_kfold(self, n_splits=5, shuffle=False, random=42, strat='TCGA_DESC'):
        if shuffle==True:
            random_state = random
        else:
            random_state = None       
        split_idx_lst = []
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        for train_val_index, test_index in kfold.split(self.res_df.index, self.res_df[strat]):
            # train_set = self.res_df.iloc[train_val_index]
            split_idx = {}
            train_index, val_index = train_test_split(train_val_index, stratify=self.res_df.iloc[train_val_index][strat], test_size=1 / (n_splits - 1), random_state=random_state)
            # print(train_index.size)
            split_idx['train'] = list(train_index)
            split_idx['val'] = list(val_index)
            split_idx['test'] = list(test_index)
            
            train_set = self.res_df.iloc[train_index]
            val_set = self.res_df.iloc[val_index]
            test_set = self.res_df.iloc[test_index]
            print('train_size:{}, val_size:{}, test_size:{}'.format(len(train_index), len(val_index), len(test_index)))
            print('train_DRUG:{}, val_DRUG:{}, test_DRUG:{}'.format(len(set(train_set['DRUG_ID'])), len(set(val_set['DRUG_ID'])), len(set(test_set['DRUG_ID']))))
            print('train_cell:{}, val_cell:{}, test_cell:{}'.format(len(set(train_set['COSMIC_ID'])), len(set(val_set['COSMIC_ID'])), len(set(test_set['COSMIC_ID']))))
            split_idx_lst.append(split_idx)
        return split_idx_lst
    
    # LOO
    def Gkfold_split(self,  out, n_splits=5, random=42):
        split_idx_lst = []
        # n_splits = len(set(self.res_df[out]))
        gfold = GroupKFold(n_splits=n_splits)
        for train_val_index, test_index in gfold.split(self.res_df.index, self.res_df[out], groups=self.res_df[out]):
            split_idx = {}
            train_index, val_index = train_test_split(train_val_index, stratify=self.res_df.iloc[train_val_index][out], test_size=1 / (n_splits - 1), random_state=random)
            # print(train_index.size)
            split_idx['train'] = list(train_index)
            split_idx['val'] = list(val_index)
            split_idx['test'] = list(test_index)
            
            train_set = self.res_df.iloc[train_index]
            val_set = self.res_df.iloc[val_index]
            test_set = self.res_df.iloc[test_index]
            print('train_size:{}, val_size:{}, test_size:{}'.format(len(train_index), len(val_index), len(test_index)))
            print('train_DRUG:{}, val_DRUG:{}, test_DRUG:{}'.format(len(set(train_set['DRUG_ID'])), len(set(val_set['DRUG_ID'])), len(set(test_set['DRUG_ID']))))
            print('train_cell:{}, val_cell:{}, test_cell:{}'.format(len(set(train_set['COSMIC_ID'])), len(set(val_set['COSMIC_ID'])), len(set(test_set['COSMIC_ID']))))
            split_idx_lst.append(split_idx)
        return split_idx_lst



class mydata(data.Dataset):
    def __init__(self, list_ID, label, res_df, drug_smiles_df, omic_encode_dict):
        'Initialization'
        self.list_ID = list_ID
        self.label = label
        self.res_df = res_df 
        self.drug_smiles_df = drug_smiles_df
        self.omic_encode_dict = omic_encode_dict

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_ID)

    def __getitem__(self, index):
        label = self.label[index]
        ID = self.list_ID[index]
        drug_id = self.res_df.iloc[ID]['DRUG_ID']
        cosmic_id = self.res_df.iloc[ID]['COSMIC_ID']
        drug_f = self.drug_smiles_df.loc[drug_id]['smiles']
        omic_f = self.omic_encode_dict[str(cosmic_id)]
        
        return drug_id, cosmic_id, drug_f, omic_f, label
