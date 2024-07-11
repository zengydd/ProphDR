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
    print(w_m[0])
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
    # weights = w_matrix.numpy().tolist() 
    fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, weights=w_matrix)
    return fig


def visualize_mol(attn_D, drug_id, cosmic_id, save_path, num):
    attn_D0 = attn_D['['+str(drug_id)+']']['['+str(cosmic_id)+']'][0]
    attn_D1 = attn_D['['+str(drug_id)+']']['['+str(cosmic_id)+']'][1]
    attn_D2 = attn_D['['+str(drug_id)+']']['['+str(cosmic_id)+']'][2]
    attn_D3 = attn_D['['+str(drug_id)+']']['['+str(cosmic_id)+']'][3]
    attn_D_list = [attn_D0, attn_D1, attn_D2, attn_D3]

    valid_lenD = drug_smiles_df.loc[drug_id]['valid_lens']
    smiles = drug_smiles_df.loc[drug_id]['smiles']

    attn_D_w = process_attn_D(attn_D_list[num], valid_lenD+1)
    fig = highlight_mol(attn_D_w, smiles)
    fig.savefig(save_path,  bbox_inches='tight')
    print('Drug visualization fig saved at {}'.format(save_path))
    plt.close()
    return 


if __name__ == '__main__':
    # Drugs NSCLC
    # Foretinib_2040 = 'COc1cc2c(Oc3ccc(NC(=O)C4(C(=O)Nc5ccc(F)cc5)CC4)cc3F)ccnc2cc1OCCCN1CCOCC1'
    # Lapatinib_1558 = 'CS(=O)(=O)CCNCc1ccc(-c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2)o1'
    # Erlotinib_1168 = 'C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1'
    # Sapitinib_1549 = 'CNC(=O)CN1CCC(Oc2cc3c(Nc4cccc(Cl)c4F)ncnc3cc2OC)CC1'
    # Crizotinib_1083 = 'C[C@@H](Oc1cc(-c2cnn(C3CCNCC3)c2)cnc1N)c1c(Cl)ccc(F)c1Cl'
    # Afatinib_1032 = 'CN(C)C/C=C/C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1O[C@H]1CCOC1'
    # Gefitinib_1010 = 'COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1'
   
    root =os.getcwd()
    data_dir = os.path.join(root, 'data_collect/')
    m_dir = os.path.join(data_dir, 'attention_analysis_data/')
   
    fig_dir = os.path.join(root, 'VIS_fig/')
    if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

    # Prepare files
    # drug_smiles_file = os.path.join(data_dir, 'drug_smiles_atom_pad.csv')
    # drug_smiles_df = pd.read_csv(drug_smiles_file, index_col='drug_id')
    # panel = pd.read_csv(m_dir + 'panel_case7.csv')
    # GENE_list
    gene_list = pd.read_csv(os.path.join(m_dir, 'Gene_list.csv'))['Gene'].to_list()

    # Load attention weights file
    attn_D = load_pickle(m_dir + 'attn_D_Gefi_Afa.pkl')

    # VISUALIZE MOL, SAV FIG
    # eg: Gefitinib (drug id: 1010) on NSCLC cell line HCC-827 (COSMIC ID: 1240146)
    drug_id = 1010
    cosmic_id = 1240146
    set_attn_D_num = 3
    save_path = fig_dir + str(drug_id) + '_' + str(cosmic_id) +'_'+ str(set_attn_D_num) +'mol.png'
    visualize_mol(attn_D, drug_id, cosmic_id, save_path, num=set_attn_D_num)
    
    
