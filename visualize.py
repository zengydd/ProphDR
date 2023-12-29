import pandas as pd
import numpy as np
import math
import matplotlib
import seaborn
import rdkit.Chem
from matplotlib import pyplot as plt
import pickle
import torch
import os
from mydata import mydata, dataset_split
from utils.load import load_pickle, save_pickle
from torch.utils import data
import random
import pickle
from collections import defaultdict
import seaborn as sn
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps


def nested_dict_factory():
    return defaultdict(nested_dict_factory)

def process_attn_D(w_m, valid_lens):
    w_m = torch.mean(w_m, dim=0)
    w_D = w_m[1:valid_lens, 0]
    # w_D = w_m[0, :valid_lens]
    w_D = w_D.cpu().numpy().tolist()
    return w_D

def process_attn_G(w_m):
    w_m = torch.mean(w_m, dim=0)
    w_G = w_m[1:715, 0]
    # w_D = w_m[0, :valid_lens]
    w_G.numpy().tolist()
    return w_G

def top_genes(top_k, weights, gene_list):
    sample_weights = np.sum(weights, axis=1)
    sorted_indices = np.argsort(weights)[::-1]
    top_genes = gene_list[sorted_indices][:top_k]
    return top_genes



def highlight_mol(w_matrix, smiles):
    mol = Chem.MolFromSmiles(smiles)
    # print('w_matrix', w_matrix)
    # weights = w_matrix.numpy().tolist()  # 根据实际情况替换为权重列表
    fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, weights=w_matrix)
    # fig.show()
    return fig


def visualize_mol(attn_D, drug_id, cosmic_id, save_path, num):
    attn_D0 = attn_D['['+str(drug_id)+']']['['+str(cosmic_id)+']'][0]
    attn_D1 = attn_D['['+str(drug_id)+']']['['+str(cosmic_id)+']'][1]
    attn_D2 = attn_D['['+str(drug_id)+']']['['+str(cosmic_id)+']'][2]
    attn_D3 = attn_D['['+str(drug_id)+']']['['+str(cosmic_id)+']'][3]
    attn_D_list = [attn_D0, attn_D1, attn_D2, attn_D3]
    # print('list@@', attn_D_list[num])

    # 药物分子可视化
    valid_lenD = drug_smiles_df.loc[drug_id]['valid_lens']
    smiles = drug_smiles_df.loc[drug_id]['smiles']
    # print(smiles)
    # cls_token +1
    attn_D_w = process_attn_D(attn_D_list[num], valid_lenD+1)
    print('drug_id', drug_id)
    print('cosmic_id', cosmic_id)
    print(attn_D_w)
    fig = highlight_mol(attn_D_w, smiles)
    fig.savefig(save_path,  bbox_inches='tight')
    plt.close()
    return 


    
