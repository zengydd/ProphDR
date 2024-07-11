import sys, os
import pickle
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from easydict import EasyDict
from collections import defaultdict

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: 
            return super().find_class(module, name)  

def nested_dict_factory():
    return defaultdict(nested_dict_factory)

def save_pickle(data, path):
	f = open(path, "wb")
	pickle.dump(data, f)
	f.close()
 
def load_pickle(path):
	if torch.cuda.is_available():
		with open(path, 'rb') as f:
			# Load the model onto the GPU
			data = pickle.load(f)
	else:
		with open(path, 'rb') as f:
			# Load the model on the CPU
			data = CPU_Unpickler(f).load()
		# 加载pickle文件并将其映射到CPU上
	f.close()
	return data


def set_file(root_path, task, method, down_sample):
    if task=='binary':   
        if method =='orio':
            res_df = pd.read_csv(root_path + 'unify_thred_Iorio.csv')
        elif method =='only2':
            res_df = pd.read_csv(root_path + 'unify_thred_only2.csv')
    else:
        res_df = pd.read_csv(root_path + 'unify_thred_only2.csv')
    return res_df


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
        

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, model_dir):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_dir)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            print('best_score:',self.best_score, 'now:',score)
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_dir)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, model_dir):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), 'checkpoint.pt')
        print('now best_score:', self.best_score)
        torch.save({'model_state_dict': model.state_dict()}, model_dir + '/checkpoint.pt')
        self.val_loss_min = val_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduction=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction:
            return torch.mean(F_loss)
        else:
            return F_loss


def markdown(title, header_lst, content_lst, file_dir):
    x = PrettyTable()
    x.title = title
    x.field_names = header_lst
    x.add_row(content_lst)
    record_path = os.path.join(file_dir, "{}_markdowntable.txt".format(title))      
    with open(record_path, 'w') as fp:
        fp.write(x.get_string())