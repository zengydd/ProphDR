import os, sys
os.environ['NUMEXPR_MAX_THREADS'] = '32'
sys.path.append("..")
import pandas as pd
import numpy as np
import random
import copy
import time
import datetime
import math
import pickle
import optuna
import yaml
import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
from torch.nn.parallel import DataParallel
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import SequentialSampler
from prettytable import PrettyTable
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, roc_curve, f1_score, precision_recall_curve
from lifelines.utils import concordance_index
from scipy.stats import pearsonr,spearmanr
from utils.optimizer import load_config, get_optimizer, get_scheduler
from easydict import EasyDict
from collections import defaultdict
from utils.load import load_pickle, save_pickle, set_file, set_random_seed, nested_dict_factory
from utils.mydata import mydata, dataset_split
from Models.RCCA_ca import CCNet
from Models.cross_attention_dual import cross_EncoderBlock_G, cross_EncoderBlock_D
from Models.Proph_DR import gbz_main_cross

torch.set_default_dtype(torch.float32)
config = './utils/train_res.yml'
config = load_config(config)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
float2str = lambda x: '%0.4f' % x

# FILEs
root = os.getcwd()
data_dir = os.path.join(root, 'data_collect/')
unify_dir = os.path.join(root, 'data_collect/unify/')
# drug data
drug_std_dir =  os.path.join(unify_dir, 'drug_std/')
drug_smiles_file = os.path.join(data_dir, 'drug_smiles_atom_pad.csv')
drug_smiles_df = pd.read_csv(drug_smiles_file, index_col='drug_id')
atom_pad_dict = load_pickle(data_dir + 'unify/drug_std/atom_pad.pkl')
# omics data
omic_encode = os.path.join(unify_dir, 'omics_std/omics_stk_dict.pkl')
mut_encode = os.path.join(unify_dir, 'omics_std/mut_dict.pkl')
cnv_encode = os.path.join(unify_dir, 'omics_std/cnv_dict.pkl')
exp_encode = os.path.join(unify_dir, 'omics_std/exp_dict.pkl')
mut_cnv = os.path.join(unify_dir, 'omics_std/omics_stk_dict_mut_cnv.pkl')
mut_exp = os.path.join(unify_dir, 'omics_std/omics_stk_dict_mut_exp.pkl')
exp_cnv = os.path.join(unify_dir, 'omics_std/omics_stk_dict_exp_cnv.pkl')


if __name__ == '__main__':
    set_random_seed()
    model_dir = os.path.join(root, 'Models/')

    task = 'IC50'
    method = 'only2'
    # setting parameters, here is a example of using all 3 types of omixs data, regression task of IC50
    exp_params = {
        'task': task,
        'omic_dim': 3,
        'method': method,
        'down_sample': False,
        'strat': None,
        'omic_f':omic_encode,
    }
    
    # data file
    omic_encode_dict = load_pickle(exp_params['omic_f'])   
    response = set_file(root_path =unify_dir ,task=exp_params['task'], method=exp_params['method'], down_sample=exp_params['down_sample'])
    train_set, val_set, test_set = dataset_split(response, stratify=exp_params['strat'])
    test_set = test_set.head(10)
    
    # init model
    net  = gbz_main_cross(task=exp_params['task'], omic_dim=exp_params['omic_dim'], res_df=response, omic_encode_dict=omic_encode_dict, model_dir=model_dir)
    
    # test
    metric_test, loss_test = net.test(test_set)
    print('test_result:{}'.format(metric_test))
   
    torch.cuda.empty_cache()

         