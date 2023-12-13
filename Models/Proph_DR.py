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
from utils.Drug_encode import encoder_D_pred, kbert
from Models.RCCA_ca import CCNet
from Models.cross_attention_dual import cross_EncoderBlock_G, cross_EncoderBlock_D
from k_bert.atom_embedding_generator import bert_atom_embedding

torch.set_default_dtype(torch.float32)
config = './utils/train_res.yml'
config = load_config(config)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
float2str = lambda x: '%0.4f' % x

root = os.getcwd()
drug_smiles_df = pd.read_csv(os.path.join(root,'data_collect/drug_smiles_atom_pad.csv'), index_col='drug_id')

def unreg_atomf_list2tensor_pred(f, max_len=96):
    # 进行padding操作，是对一个batchlist里的数据
    valid_l = len(f)
    f_pad = np.pad(f,((0, max_len-f.shape[0]),(0,0)), constant_values=0)
    return f_pad, max_len, valid_l


def encoder_D_pred(smiles):
    h_global, h_atom = bert_atom_embedding(smiles)
    # print('h_global', h_global)
    f_pad, max_len, valid_lens = unreg_atomf_list2tensor_pred(h_atom)
    valid_lenD_list = [valid_lens]
    valid_lens = torch.tensor(valid_lenD_list)
    encode_D_pred = np.vstack((h_global, f_pad))
    encode_D_pred_list = [encode_D_pred]
    encode_D_pred = torch.stack([torch.tensor(arr) for arr in list(encode_D_pred_list)])
    return encode_D_pred, valid_lens


def encoder_D(drug_id):
    drug_id_list = list(drug_id.cpu())
    valid_lenD_list = drug_smiles_df.loc[drug_id_list]['valid_lens'].to_list()
    valid_lenD_list = [i+1 for i in valid_lenD_list]
    valid_lens = torch.tensor(valid_lenD_list)
    encode_D_list = kbert(drug_id_list)
    # print('+++', encode_D_list[0].shape)
    encode_D = torch.stack([torch.tensor(arr) for arr in encode_D_list])
    # print(encode_D.shape)
    return encode_D, valid_lens

class MLP(nn.Module):
    def __init__(self, h_dim, dropoutrate=0.2):
        super(MLP, self).__init__()
        self.h_dim = h_dim
        self.dropout = nn.Dropout(dropoutrate)
        self.linear = nn.Linear(self.h_dim , self.h_dim)
        self.linear3 = nn.Linear(self.h_dim , 1)
        self.regression = nn.Sequential(
            nn.Linear(812, 1024),
            nn.ReLU(),
            nn.Dropout(dropoutrate),
            nn.Linear(1024, 1024),
            nn.ELU(),
            nn.Dropout(dropoutrate),
            nn.Linear(1024, 1)
        )
    def forward(self, x):
        x = self.dropout(self.linear(x))
        x =  self.dropout(self.linear3(x))
        x = x.squeeze(-1)
        x = self.regression(x)
        return x
    
class Predictor(nn.Module):
    def __init__(self, h_dim, num_heads, omic_dim):
        super(Predictor, self).__init__()
        self.device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        # q:Cell   k:drug
        self.len_g = 715
        # padding max_len==96 + 1(global)
        self.len_d = 97
        self.dim_g = omic_dim
        self.dim_d = 768
        # self.h_dim = 128
        self.h_dim = h_dim
        self.num_heads = num_heads
        self.norm_shape_G = [self.len_g, self.h_dim]
        self.norm_shape_D = [self.len_d, self.h_dim]

        self.model_g = CCNet(self.dim_g, recurrence=2)
        self.l_G = nn.Linear(self.dim_g, self.h_dim)
        self.l_D = nn.Linear(self.dim_d, self.h_dim)
        self.l_common = nn.Linear(self.h_dim, self.h_dim)
        
        self.cls_token = nn.Parameter(torch.ones(1, 1, self.dim_g))
        self.cross_encoder_G = cross_EncoderBlock_G(self.h_dim, self.h_dim, self.h_dim, 
                                                self.h_dim, self.num_heads, 
                                                norm_shape=self.norm_shape_G, 
                                                bias=False)
        
        self.cross_encoder_D = cross_EncoderBlock_D(self.h_dim, self.h_dim, self.h_dim, 
                                                self.h_dim, self.num_heads, 
                                                norm_shape=self.norm_shape_D, 
                                                bias=False)
        self.mlp = MLP(self.h_dim)

    # def forward(self, D_ids, f_G, cosmic_id, VIS=False):
    def forward(self, drug_id, encode_D, f_G, valid_lens, cosmic_id):
        torch.cuda.empty_cache()
        # cosmic_id_list = cosmic_id.cpu().numpy().tolist()
        # drug_id_list = list(D_ids.cpu())

        v_D = encode_D.detach().to(self.device).float()
        f_G = f_G.detach().to(self.device)
        cls_token = self.cls_token.repeat(f_G.shape[0], 1, 1).to(self.device)
        f_G = torch.cat((cls_token, f_G), dim=1)
        encode_G = f_G.reshape(-1, 1, self.len_g, self.dim_g)

        # OMIC_EMBEDDING
        self.model_g = self.model_g.to(self.device)
        v_G, attn_list = self.model_g(encode_G)
        # ALL
        v_G_cross = self.l_G(v_G)
        v_G_cross = self.l_common(v_G_cross)
        v_D_cross = self.l_D(v_D)
        v_D_cross = self.l_common(v_D_cross)
        
        # CROSS_ATTN
        out_G, attn_G0 = self.cross_encoder_G(v_G_cross, v_D_cross, v_D_cross, valid_lens)
        out_G, attn_G1 = self.cross_encoder_G(out_G, v_D_cross, v_D_cross, valid_lens)
        out_G, attn_G2 = self.cross_encoder_G(out_G, v_D_cross, v_D_cross, valid_lens)
        out_G, attn_G3 = self.cross_encoder_G(out_G, v_D_cross, v_D_cross, valid_lens)
        
        out_D, attn_D0 = self.cross_encoder_D(v_D_cross, v_G_cross, v_G_cross, valid_lens)
        out_D, attn_D1 = self.cross_encoder_D(out_D, v_G_cross, v_G_cross, valid_lens)
        out_D, attn_D2 = self.cross_encoder_D(out_D, v_G_cross, v_G_cross, valid_lens)
        out_D, attn_D3 = self.cross_encoder_D(out_D, v_G_cross, v_G_cross, valid_lens)

        output = torch.cat([out_D, out_G], dim=1)
        output = self.mlp(output)
        
        # if VIS== True:
        #     cosmic_id = str(cosmic_id.numpy())
        #     drug_id = str(drug_id.numpy())
        #     attn_D_dict[drug_id][cosmic_id]= [attn_D0, attn_D1, attn_D2, attn_D3]
        #     attn_G_dict[drug_id][cosmic_id]= [attn_G0, attn_G1, attn_G2, attn_G3]

        return output

class gbz_main_cross(object):
    def __init__(self, task, omic_dim, res_df, omic_encode_dict, model_dir):
        # self.model_drug = bert_atom_embedding
        self.task = task
        self.model_dir = model_dir
        self.model = Predictor(h_dim=128, num_heads=4, omic_dim=omic_dim)
        self.device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.omic_encode_dict = omic_encode_dict
        self.record_file = os.path.join(self.model_dir, "valid_markdowntable.txt")
        self.pkl_file = os.path.join(self.model_dir, "loss_curve_iter.pkl")
        self.res_df = res_df
        if self.task=='binary':
            self.label = 'binary'  
            self.loss_fct = FocalLoss(logits=True)
        elif self.task=='IC50':
            self.label = 'LN_IC50'
            self.loss_fct = torch.nn.MSELoss() 
        elif self.task=='AUC':
            self.label='AUC'
            self.loss_fct = torch.nn.MSELoss()

    def validate(self, generator, model):
        torch.cuda.empty_cache()
        loss_fct = self.loss_fct
        model.eval()
        y_label = []
        y_pred = []
        with torch.no_grad():
            for i, (drug_id, cosmic_id, drug_fs, omic_f, label) in enumerate(generator):
                torch.cuda.empty_cache()
                label = Variable(torch.from_numpy(np.array(label)).float()).to(self.device)
                # score = model(drug_id, omic_f, cosmic_id)
                encode_D, valid_lens = encoder_D(drug_id)
                score = model(drug_id, encode_D, omic_f, valid_lens, cosmic_id)
                
                score_flatten = score.flatten().to(self.device)
                loss = loss_fct(score_flatten, label).to(self.device)
                y_label.append(label.view(-1,1))
                y_pred.append(score_flatten.view(-1, 1))
            y_label = torch.cat(y_label, dim=0).cpu().numpy().flatten()
            y_pred = torch.cat(y_pred, dim=0).cpu().numpy().flatten()

            # Metrics
        if self.task=='binary':
            metric = {}
            y_pred = torch.sigmoid(torch.tensor(y_pred)).tolist()
            # print('y_label:{},\ny_pred:{}'.format(y_label, y_pred))
            metric['AUC'] = roc_auc_score(y_label, y_pred)
            metric['pr_score'] = average_precision_score(y_label, y_pred)
            false_positive_rate,true_positive_rate,thresholds = roc_curve(y_label, y_pred)
            recall, precision, thresholds = precision_recall_curve(y_label, y_pred)
            print('roc_curve data:', [false_positive_rate,true_positive_rate,thresholds])
            print('PR_curve data:', [recall, precision])
            to_binary = lambda x: 1 if x > 0.5 else 0 
            y_pred_cls = list(map(to_binary, y_pred))
            metric['acc'] = accuracy_score(y_label, y_pred_cls)
            metric['F1'] = f1_score(y_label, y_pred_cls, average='binary')
            print('metric_resut_{}{}:'.format(self.task, metric))
        else:
            metric = {}
            metric['r2'] = r2_score(y_label, y_pred)
            metric['MAE'] = mean_absolute_error(y_label, y_pred)
            metric['mse'] = mean_squared_error(y_label, y_pred)
            metric['rmse'] = torch.sqrt(torch.tensor(metric['mse']))
            metric['spearman'] = spearmanr(y_label, y_pred)[0]
            metric['pearson'] = pearsonr(y_label, y_pred)[0]
            metric['ci'] = concordance_index(y_label, y_pred)
            print('metric_resut_{}{}:'.format(self.task, metric))
        
        model.train()
        return metric, loss

    def train(self, train_set, val_set, **param):
        torch.cuda.empty_cache()
        self.model = self.model.to(self.device)

        print(self.model)
        label = self.label
        loss_fct = self.loss_fct
        BATCH_SIZE = param['bs']
        train_epoch = param['te']
        patience = param['pt']

        opt = getattr(torch.optim, param['optimizer'])(self.model.parameters(), 
                                                        lr=param['lr'], 
                                                        weight_decay=param['decay'])

        params = {'batch_size': BATCH_SIZE,
                  'shuffle': True,
                  'num_workers': 0,
                  'drop_last': False}
        # loader
        train_generator = data.DataLoader(
            mydata(
            train_set.index.values,
            train_set[label].values, 
            self.res_df, 
            drug_smiles_df, 
            self.omic_encode_dict
            ), 
            **params)
        val_generator = data.DataLoader(
            mydata(
            val_set.index.values,
            val_set[label].values, 
            self.res_df, 
            drug_smiles_df, 
            self.omic_encode_dict
            ), 
            **params)
        
        max_MSE = 10000
        model_max = copy.deepcopy(self.model)
        writer = SummaryWriter(self.model_dir)
        table = PrettyTable()
        table.title = 'valid'
        
        t_start = time.time()
        loss_history = []
        early_stopping = EarlyStopping(patience=patience, verbose=False)
        for epo in range(train_epoch):
            torch.cuda.empty_cache()
            for i, (drug_id, cosmic_id, drug_fs, omic_f, label) in enumerate(train_generator):
                torch.backends.cudnn.enabled = False
                # score = self.model(drug_id, omic_f, cosmic_id)
                encode_D, valid_lens = encoder_D(drug_id)
                score = self.model(drug_id, encode_D, omic_f, valid_lens, cosmic_id)
                # print('score:'.format(type(score), score))
                label = Variable(torch.from_numpy(np.array(label))).float().to(self.device)
                n = torch.squeeze(score, 1).float()
                n = n.squeeze(-1)

                loss = loss_fct(n, label)
                loss_history.append(loss.item())
                writer.add_scalar("Loss/train", loss.item(), epo)
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                if (i % 1000 == 0):
                    t_now = time.time()
                    print('Training at Epoch ' + str(epo + 1) +
                            ' iteration ' + str(i) + \
                            ' with loss ' + str(loss.cpu().detach().numpy())[:7] + \
                            ' with lr ' + str(opt.param_groups[0]['lr']) + \
                            ". Total time " + str(int(t_now - t_start) / 3600)[:7] + " hours")

            metric_result, loss_val = self.validate(val_generator, self.model)
            print('Validation at Epoch:{} \nMetric_result:{}'.format(str(epo + 1), metric_result))
            # mark
            table.field_names =  ['# epoch'] + list(metric_result.keys()) + ['loss']
            valid_content_lst = ['epo'+str(epo)]+list(map(float2str, metric_result.values()))+[str(loss_val)]
            table.add_row(valid_content_lst)
            # tensorboard
            for k, v in metric_result.items():
                writer.add_scalar("valid/{}".format(k), v, epo)
            writer.add_scalar("Loss/valid", loss_val.item(), epo)

            # early_stop
            early_stopping(loss, self.model, self.model_dir)
            if early_stopping.early_stop:
                print("Early stopping at epoch{}".format(epo))
                break
            
            lowest_val = 1e9
            if loss_val < lowest_val:
                lowest_val = lowest_val
                self.save_model(self.model, self.model_dir)
            print(f'Val Loss: {loss_val}')

        # self.model = model_max
        with open(self.record_file, 'w') as fp:
            fp.write(table.get_string())
        with open(self.pkl_file, 'wb') as pck:
            pickle.dump(loss_history, pck)

        print('--- Training Finished ---')
        writer.flush()
        writer.close()
        return metric_result, loss_val

    def test(self, test_set):
        self.model = self.model.to(self.device)
        label = self.label
        params = {'batch_size': 200,
                  'shuffle': True,
                  'num_workers': 0,
                  'drop_last': False}
        # loader
        test_generator = data.DataLoader(
            mydata(
            test_set.index.values,
            test_set[label].values, 
            self.res_df, 
            drug_smiles_df, 
            self.omic_encode_dict
            ), 
            **params)       
        print("=====testing...")
        self.model.load_state_dict(torch.load(self.model_dir + '/checkpoint.pt')['model_state_dict'])
        metric_result, loss = self.validate(test_generator, self.model)
        return metric_result, loss

    def pred(self, smiles_list, cosmic_id_list, pt_path=os.path.join(root, 'ckpt/checkpoint.pt'), drug_id=0):
        with torch.no_grad():
            score_list = []
            smi_list = []
            cell_list = []
            for smiles in smiles_list:
                smi_list.append(smiles)
                for cosmic_id in cosmic_id_list:
                    cell_list.append(str(cosmic_id))
                    self.model = self.model.to(self.device)
                    omic_f = self.omic_encode_dict[str(cosmic_id)]
                    omic_f = omic_f.unsqueeze(0)
                    self.model.load_state_dict(torch.load(pt_path, map_location='cpu')['model_state_dict'])
                    encode_D_pred, valid_lens = encoder_D_pred(smiles)
                    score = self.model(drug_id, encode_D_pred, omic_f, valid_lens, cosmic_id)
                    score = score.flatten().to(self.device).cpu().numpy().item()
                    score_list.append(score)
            res = pd.DataFrame()
            res['LN(IC50)'] = pd.Series(score_list)
            res['smiles'] = smi_list
            res['cosmic'] = cell_list
        return res
    
    def save_model(self, model, model_dir):
        torch.save({'model_state_dict': model.state_dict()}, model_dir + '/checkpoint.pt')
        print('model_saved:{}'.format(model_dir))
        


