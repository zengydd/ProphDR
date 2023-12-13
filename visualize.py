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
from collections import defaultdict
from utils.load import nested_dict_factory, load_pickle, save_pickle, EarlyStopping, FocalLoss
from utils.mydata import mydata, dataset_split
from utils.Drug_encode import kbert
from Models.RCCA_ca import CCNet
from Models.cross_attention_dual import cross_EncoderBlock_G, cross_EncoderBlock_D
from Models.cross_attention_dual import cross_EncoderBlock_G, cross_EncoderBlock_D
from Models.Proph_DR import gbz_main_cross

root = os.getcwd()
unify_dir = os.path.join(root, 'data_collect/unify/')
omic_encode = os.path.join(unify_dir, 'omics_std/omics_stk_dict.pkl')
mut_encode = os.path.join(unify_dir, 'omics_std/mut_dict.pkl')
cnv_encode = os.path.join(unify_dir, 'omics_std/cnv_dict.pkl')
exp_encode = os.path.join(unify_dir, 'omics_std/exp_dict.pkl')
mut_cnv = os.path.join(unify_dir, 'omics_std/omics_stk_dict_mut_cnv.pkl')
mut_exp = os.path.join(unify_dir, 'omics_std/omics_stk_dict_mut_exp.pkl')
exp_cnv = os.path.join(unify_dir, 'omics_std/omics_stk_dict_exp_cnv.pkl')

if __name__ == '__main__':
   
    attn_G_dict = nested_dict_factory()
    attn_D_dict = nested_dict_factory()
    CCN_dict = nested_dict_factory()

    pt_path = os.path.join(root, 'Models/')
    exp_params = {
        'task': 'IC50',
        'omic_dim': 3,
        'method': 'only2',
        'down_sample': False,
        'strat': None,
        'omic_f':omic_encode,
    }
    print('file_set:', exp_params)

    test_set = pd.read_csv('/home/yundian/gbz/gbz/model_dual_ca/attention_analysis_data/pnael.csv', index_col=0)
    print('test_set.shape', test_set.shape)

    # load_model
    pt_path = '/home/yundian/gbz/gbz/model_dual_ca/attention_analysis_data/cycle_20230509_15_5647/checkpoint.pt'
    omic_encode = load_pickle(exp_params['omic_f'])            
    net  = gbz_main_cross(task=exp_params['task'], omic_dim=exp_params['omic_dim'], res_df=test_set, omic_encode_dict=omic_encode, model_dir=model_dir)
    # testing
    metric_test, loss_test = net.test(test_set, pt_path)
    print('test_metric:', metric_test)

# save_dict
    # save_pickle(attn_D_dict, 'attn_D_0801_test.pkl')
    # save_pickle(attn_G_dict, 'attn_G_0801_test.pkl')
    # save_pickle(CCN_dict, 'ccn_attn_0801_test.pkl')

# 核苷类似物
    # save_pickle(attn_D_dict, 'attn_D_nucle.pkl')
    # save_pickle(attn_G_dict, 'attn_G_nucle.pkl')
    # save_pickle(CCN_dict, 'ccn_attn_nucle.pkl')

# panel_3
    # save_pickle(attn_D_dict, 'attn_D_panel_3.pkl')
    # save_pickle(attn_G_dict, 'attn_G_panel_3.pkl')
    # save_pickle(CCN_dict, 'ccn_attn_panel_3.pkl')
    # print('finish================================')


    # Gefi_afa
    save_pickle(attn_D_dict, 'attn_D_Afa.pkl')
    save_pickle(attn_G_dict, 'attn_G_Afa.pkl')
    # save_pickle(CCN_dict, 'ccn_attn_Gefi_Afa.pkl')
    print('finish================================')
