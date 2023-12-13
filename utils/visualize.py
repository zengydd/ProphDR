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

# # 打开文件以写入模式
# with open('gene_list.txt', 'w') as file:
#     # 遍历列表中的每个元素
#     for item in gene_list:
#         # 将元素写入文件，并在末尾添加换行符
#         file.write(item + '\n')
        

def nested_dict_factory():
    return defaultdict(nested_dict_factory)

def process_attn_D(w_m, valid_lens):
    w_m = torch.mean(w_m, dim=0)
    w_D = w_m[1:valid_lens, 0]
    # w_D = w_m[0, :valid_lens]
    w_D.numpy().tolist()
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
    weights = w_matrix.numpy().tolist()  # 根据实际情况替换为权重列表
    fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, weights=weights)
    # fig.show()
    return fig

def visualize_mol(attn_D, drug_id, cosmic_id, num):
    attn_D0 = attn_D['['+str(drug_id)+']']['['+str(cosmic_id)+']'][0]
    attn_D1 = attn_D['['+str(drug_id)+']']['['+str(cosmic_id)+']'][1]
    attn_D2 = attn_D['['+str(drug_id)+']']['['+str(cosmic_id)+']'][2]
    attn_D3 = attn_D['['+str(drug_id)+']']['['+str(cosmic_id)+']'][3]
    attn_D_list = [attn_D0, attn_D1, attn_D2, attn_D3]
    # 药物分子可视化 sens
    
    valid_lenD = drug_smiles_df.loc[drug_id]['valid_lens']
    smiles = drug_smiles_df.loc[drug_id]['smiles']
    print(smiles)
    # cls_token +1
    attn_D_w = process_attn_D(attn_D_list[num], valid_lenD+1)
    print(attn_D_w)
    fig = highlight_mol(attn_D_w, smiles)
    save_path = fig_dir + str(drug_id) + '_' + str(cosmic_id) +'_'+ str(num) +'mol.png'
    fig.savefig(save_path,  bbox_inches='tight')
    # fig.save(save_path)
    return 


if __name__ == '__main__':
    # Drugs NSCLC
    Foretinib_2040 = 'COc1cc2c(Oc3ccc(NC(=O)C4(C(=O)Nc5ccc(F)cc5)CC4)cc3F)ccnc2cc1OCCCN1CCOCC1'
    Lapatinib_1558 = 'CS(=O)(=O)CCNCc1ccc(-c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2)o1'
    Erlotinib_1168 = 'C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1'
    Sapitinib_1549 = 'CNC(=O)CN1CCC(Oc2cc3c(Nc4cccc(Cl)c4F)ncnc3cc2OC)CC1'
    Crizotinib_1083 = 'C[C@@H](Oc1cc(-c2cnn(C3CCNCC3)c2)cnc1N)c1c(Cl)ccc(F)c1Cl'
    Afatinib_1032 = 'CN(C)C/C=C/C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1O[C@H]1CCOC1'
    Gefitinib_1010 = 'COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1'
   
    root ='/home/yundian/gbz/gbz/'
    fig_dir = os.path.join(root, 'model_dual_ca/attn_fig/')
    data_dir = os.path.join(root, 'data_collect/')
    m_dir = os.path.join(root, 'model_dual_ca/')
    # drug_smiles
    drug_smiles_file = os.path.join(data_dir, 'drug_smiles_atom_pad.csv')
    drug_smiles_df = pd.read_csv(drug_smiles_file, index_col='drug_id')
    # panel
    panel = pd.read_csv(m_dir + 'attention_analysis_data/panel_case7.csv')
    # GENE_list
    omic_file = pd.read_csv(data_dir + 'unify/res_omic_mut.csv')
    gene_list = omic_file.iloc[:, 0].tolist()
    # D\C info
    drug_list = [1010, 1032, 1083, 1549, 1168, 1558, 2040]
    cell_list = []

    attn_D = load_pickle(m_dir + 'attention_analysis_data/attn_D_0801.pkl')
    attn_G = load_pickle(m_dir + 'attention_analysis_data/attn_G_0801.pkl')
    # attn_CCN = load_pickle(m_dir + 'attention_analysis_data/attn_G_0801.pkl')
    
    # VISUALIZE MOL, SAV FIG
    set_attn_D_num = 3
    for index, row in panel.iterrows():
        drug_id = panel['DRUG_ID']
        cosmic_id = panel['COSMIC_ID']
        print('drug_id', drug_id)
        print('cosmic_id', cosmic_id)
        
        num = set_attn_D_num
        visualize_mol(attn_D, drug_id, cosmic_id)
    
    
    # VISUALIZE GENE, SAV RANK
    
