import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import datetime
import random
import os
import csv
import pickle
import math
from models.TGDRP import TGDRP
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from preprocess_gene import get_STRING_graph, get_predefine_cluster
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr
from tqdm import tqdm


dict_dir = '/data/ouyangzhenqiu/project/cloud_ecg/TGSA/data/similarity_augment/dict/'
with open(dict_dir + "cell_id2idx_dict", 'rb') as f:
    cell_id2idx_dict = pickle.load(f)
with open(dict_dir + "drug_name2idx_dict", 'rb') as f:
    drug_name2idx_dict = pickle.load(f)
with open(dict_dir + "cell_idx2id_dict", 'rb') as f:
    cell_idx2id_dict = pickle.load(f)
with open(dict_dir + "drug_idx2name_dict", 'rb') as f:
    drug_idx2name_dict = pickle.load(f)


def train(model, loader, criterion, opt, device):
    model.train()

    for idx, data in enumerate(tqdm(loader, desc='Iteration')):
        drug, cell, label = data
        if isinstance(cell, list):
            drug, cell, label = drug.to(device), [feat.to(device) for feat in cell], label.to(device)
        else:
            drug, cell, label = drug.to(device), cell.to(device), label.to(device)
        output = model(drug, cell)
        loss = criterion(output, label.view(-1, 1).float())
        opt.zero_grad()
        loss.backward()
        opt.step()

    print('Train Loss:{}'.format(loss))
    return loss


def validate(model, loader, device):
    model.eval()

    y_true = []
    y_pred = []

    total_loss = 0
    with torch.no_grad():
        for data in tqdm(loader, desc='Iteration'):
            drug, cell, label = data
            if isinstance(cell, list):
                drug, cell, label = drug.to(device), [feat.to(device) for feat in cell], label.to(device)
            else:
                drug, cell, label = drug.to(device), cell.to(device), label.to(device)
            output = model(drug, cell)
            total_loss += F.mse_loss(output, label.view(-1, 1).float(), reduction='sum')
            y_true.append(label.view(-1, 1))
            y_pred.append(output)

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    rmse = torch.sqrt(total_loss / len(loader.dataset))
    MAE = mean_absolute_error(y_true.cpu(), y_pred.cpu())
    r2 = r2_score(y_true.cpu(), y_pred.cpu())
    r = pearsonr(y_true.cpu().numpy().flatten(), y_pred.cpu().numpy().flatten())[0]

    return rmse, MAE, r2, r

def gradient(model, drug_name, cell_name, drug_dict, cell_dict, edge_index, args):
    cell_dict[cell_name].edge_index = torch.tensor(edge_index, dtype=torch.long)
    drug = Batch.from_data_list([drug_dict[drug_name]]).to(args.device)
    cell = Batch.from_data_list([cell_dict[cell_name]]).to(args.device)
    
    model.eval()
    drug_representation = model.GNN_drug(drug)
    drug_representation = model.drug_emb(drug_representation)

    cell_node, cell_representation = model.GNN_cell.grad_cam(cell)
    cell_representation = model.cell_emb(cell_representation)

    # combine drug feature and cell line feature
    x = torch.cat([drug_representation, cell_representation], -1)
    ic50 = model.regression(x)
    ic50.backward()
    # cell_node_importance = torch.relu((cell_node*torch.mean(cell_node.grad, dim=0)).sum(dim=1))
    cell_node_importance = torch.abs((cell_node*torch.mean(cell_node.grad, dim=0)).sum(dim=1))
    cell_node_importance = cell_node_importance/cell_node_importance.sum()
    sorted, indices = torch.sort(cell_node_importance, descending=True)
    return ic50, indices.cpu()

def inference(model, drug_dict, cell_dict, edge_index, save_name, args):
    model.eval()
    IC = pd.read_csv("/data/ouyangzhenqiu/project/cloud_ecg/TGSA/data/IC50_GDSC/PANCANCER_IC_82833_580_170.csv")
    if args.setup == 'known':
        train_set, val_test_set = train_test_split(IC, test_size=0.2, random_state=42, stratify=IC['Cell line name'])
        val_set, test_set = train_test_split(val_test_set, test_size=0.5, random_state=42,
                                             stratify=val_test_set['Cell line name'])
        
    cell_table = IC[["DepMap_ID", "stripped_cell_line_name"]].drop_duplicates(keep='first')
    drug_table = IC["Drug name"].drop_duplicates(keep='first').to_frame()
    cell_table['value'] = 1
    drug_table['value'] = 1
    drug_cell_table = drug_table.merge(cell_table, how='left', on='value')
    del drug_cell_table['value']
    unknown_set = drug_cell_table.append(IC[["Drug name", "DepMap_ID", "stripped_cell_line_name"]])
    unknown_set.drop_duplicates(keep=False, inplace=True)
    dataset = {'train':train_set, 'val':val_set, 'test':test_set, 'unknown':unknown_set}
    writer = pd.ExcelWriter(save_name)
    for dataset_name, data in dataset.items():
        data.reset_index(drop=True, inplace=True)
        IC50_pred = []
        with torch.no_grad():
            drug_name, cell_ID, cell_line_name = data['Drug name'], data["DepMap_ID"], data["stripped_cell_line_name"]
            for cell in cell_ID:
                cell_dict[cell].edge_index = torch.tensor(edge_index, dtype=torch.long)
            drug_list = [drug_dict[name] for name in drug_name]
            cell_list = [cell_dict[name] for name in cell_ID]
            batch_size = 2048
            batch_num = math.ceil(len(drug_list)/batch_size)
            for index in range(batch_num):
                drug = Batch.from_data_list(drug_list[index*batch_size:(index+1)*batch_size]).to(args.device)
                cell = Batch.from_data_list(cell_list[index*batch_size:(index+1)*batch_size]).to(args.device)
                y_pred = model(drug, cell)
                IC50_pred.append(y_pred)
            IC50_pred = torch.cat(IC50_pred, dim=0)
        table = pd.concat([drug_name, cell_ID, cell_line_name], axis=1)
        if dataset_name != 'unknown':
            table["IC50"] = data["IC50"]
        table["IC50_Pred"] = IC50_pred.cpu().numpy()
        if dataset_name != 'unknown':
            table["Abs_error"] = np.abs(IC50_pred.cpu().numpy()-np.array(table["IC50"]).reshape(-1,1))
        table.to_excel(writer, sheet_name=dataset_name, index=False)
        torch.cuda.empty_cache()
    writer.close()
        
        
class MyDataset(Dataset):
    def __init__(self, drug_dict, cell_dict, IC, edge_index):
        super(MyDataset, self).__init__()
        self.drug, self.cell = drug_dict, cell_dict
        IC.reset_index(drop=True, inplace=True)  # train_test_split之后，数据集的index混乱，需要reset
        self.drug_name = IC['Drug name']
        self.Cell_line_name = IC['DepMap_ID']
        self.value = IC['IC50']
        self.edge_index = torch.tensor(edge_index, dtype=torch.long)

    def __len__(self):
        return len(self.value)

    def __getitem__(self, index):
        self.cell[self.Cell_line_name[index]].edge_index = self.edge_index
        # self.cell[self.Cell_line_name[index]].adj_t = SparseTensor(row=self.edge_index[0], col=self.edge_index[1])
        return (self.drug[self.drug_name[index]], self.cell[self.Cell_line_name[index]], self.value[index])


class MyDataset_CDR(Dataset):
    def __init__(self, drug_dict, cell_dict, IC):
        super().__init__()
        self.drug, self.cell = drug_dict, cell_dict
        IC.reset_index(drop=True, inplace=True)  # train_test_split之后，数据集的index混乱，需要reset
        self.drug_name = IC['Drug name']
        self.Cell_line_name = IC['DepMap_ID']
        self.value = IC['IC50']

    def __len__(self):
        return len(self.value)

    def __getitem__(self, index):
        return (self.drug[self.drug_name[index]], self.cell[self.Cell_line_name[index]], self.value[index])


class MyDataset_name(Dataset):
    def __init__(self, drug_dict, cell_dict, IC):
        super().__init__()
        self.drug, self.cell = drug_dict, cell_dict
        IC.reset_index(drop=True, inplace=True)
        self.drug_name = IC['Drug name']
        self.Cell_line_name = IC['Cell line name']
        self.value = IC['IC50']

    def __len__(self):
        return len(self.value)

    def __getitem__(self, index):
        return (self.drug[self.drug_name[index]], self.cell[self.Cell_line_name[index]], self.value[index])


def _collate(samples):
    drugs, cells, labels = map(list, zip(*samples))
    batched_drug = Batch.from_data_list(drugs)
    batched_cell = Batch.from_data_list(cells)
    return batched_drug, batched_cell, torch.tensor(labels)


def _collate_drp(samples):
    drugs, cells, labels = map(list, zip(*samples))
    batched_graph = Batch.from_data_list(drugs)
    cells = [torch.tensor(cell) for cell in cells]
    return batched_graph, torch.stack(cells, 0), torch.tensor(labels)


def _collate_CDR(samples):
    drugs, cells, labels = map(list, zip(*samples))
    batched_graph = Batch.from_data_list(drugs)
    exp = [torch.tensor(cell[0]) for cell in cells]
    cn = [torch.tensor(cell[1]) for cell in cells]
    mu = [torch.tensor(cell[2]) for cell in cells]
    return batched_graph, [torch.stack(exp, 0), torch.stack(cn, 0), torch.stack(mu, 0)], torch.tensor(labels)


def load_data(IC, drug_dict, cell_dict, edge_index, args):
    if args.setup == 'known':
        train_set, val_test_set = train_test_split(IC, test_size=0.2, random_state=42, stratify=IC['Cell line name'])
        val_set, test_set = train_test_split(val_test_set, test_size=0.5, random_state=42,
                                             stratify=val_test_set['Cell line name'])

    elif args.setup == 'leave_drug_out':
        ## scaffold
        smiles_list = pd.read_csv('./data/IC50_GDSC/drug_smiles.csv')[
            ['CanonicalSMILES', 'drug_name']]
        train_set, val_set, test_set = scaffold_split(IC, smiles_list, seed=42)

    elif args.setup == 'leave_cell_out':
        ## stratify
        cell_info = IC[['Tissue', 'Cell line name']].drop_duplicates()
        train_cell, val_test_cell = train_test_split(cell_info, stratify=cell_info['Tissue'], test_size=0.4,
                                                     random_state=42)
        val_cell, test_cell = train_test_split(val_test_cell, stratify=val_test_cell['Tissue'], test_size=0.5,
                                               random_state=42)

        train_set = IC[IC['Cell line name'].isin(train_cell['Cell line name'])]
        val_set = IC[IC['Cell line name'].isin(val_cell['Cell line name'])]
        test_set = IC[IC['Cell line name'].isin(test_cell['Cell line name'])]

    else:
        raise ValueError

    if args.model == 'TCNN':
        Dataset = MyDataset_name
        collate_fn = None
        train_dataset = Dataset(drug_dict, cell_dict, train_set)
        val_dataset = Dataset(drug_dict, cell_dict, val_set)
        test_dataset = Dataset(drug_dict, cell_dict, test_set)

    elif args.model == 'GraphDRP':
        Dataset = MyDataset_name
        collate_fn = _collate_drp
        train_dataset = Dataset(drug_dict, cell_dict, train_set)
        val_dataset = Dataset(drug_dict, cell_dict, val_set)
        test_dataset = Dataset(drug_dict, cell_dict, test_set)

    elif args.model == 'DeepCDR':
        Dataset = MyDataset_CDR
        collate_fn = _collate_CDR
        train_dataset = Dataset(drug_dict, cell_dict, train_set)
        val_dataset = Dataset(drug_dict, cell_dict, val_set)
        test_dataset = Dataset(drug_dict, cell_dict, test_set)

    else:
        Dataset = MyDataset
        collate_fn = _collate
        train_dataset = Dataset(drug_dict, cell_dict, train_set, edge_index=edge_index)
        val_dataset = Dataset(drug_dict, cell_dict, val_set, edge_index=edge_index)
        test_dataset = Dataset(drug_dict, cell_dict, test_set, edge_index=edge_index)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                              num_workers=4
                              )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                            num_workers=4
                            )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                             num_workers=4)

    return train_loader, val_loader, test_loader


def prepare_val_data(IC, drug_dict, cell_dict, edge_index, split_idx, fold, model, args):
    train_set = IC.iloc[split_idx['train'][fold], :]
    val_set = IC.iloc[split_idx['val'][fold], :]
    test_set = IC.iloc[split_idx['test'][fold], :]

    if model == 'TCNN':
        Dataset = MyDataset_name
        collate_fn = None
        train_dataset = Dataset(drug_dict, cell_dict, train_set)
        val_dataset = Dataset(drug_dict, cell_dict, val_set)
        test_dataset = Dataset(drug_dict, cell_dict, test_set)

    elif model == 'GraphDRP':
        Dataset = MyDataset_name
        collate_fn = _collate_drp
        train_dataset = Dataset(drug_dict, cell_dict, train_set)
        val_dataset = Dataset(drug_dict, cell_dict, val_set)
        test_dataset = Dataset(drug_dict, cell_dict, test_set)

    elif model == 'DeepCDR':
        Dataset = MyDataset_CDR
        collate_fn = _collate_CDR
        train_dataset = Dataset(drug_dict, cell_dict, train_set)
        val_dataset = Dataset(drug_dict, cell_dict, val_set)
        test_dataset = Dataset(drug_dict, cell_dict, test_set)

    else:
        Dataset = MyDataset
        collate_fn = _collate
        train_dataset = Dataset(drug_dict, cell_dict, train_set, edge_index=edge_index)
        val_dataset = Dataset(drug_dict, cell_dict, val_set, edge_index=edge_index)
        test_dataset = Dataset(drug_dict, cell_dict, test_set, edge_index=edge_index)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                              num_workers=4
                              )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                            num_workers=4
                            )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                             num_workers=4)

    return train_loader, val_loader, test_loader


def get_idx_split_cell(dataset, k_splits=5):
    split_idx = {}

    cell_info = dataset[['Tissue', 'DepMap_ID']].drop_duplicates()
    cell_info.reset_index(drop=True, inplace=True)

    root_idx_dir = './data_split/{}_fold/cell'.format(k_splits)
    if not os.path.exists(root_idx_dir):
        os.makedirs(root_idx_dir)
        cross_val_fold = StratifiedKFold(n_splits=k_splits, shuffle=True, random_state=42)
        for train_val_name, test_name in cross_val_fold.split(cell_info, cell_info['Tissue']):
            train_name, val_name = train_test_split(cell_info.iloc[train_val_name]['DepMap_ID'],
                                                    stratify=cell_info.iloc[train_val_name]['Tissue'],
                                                    test_size=1 / (k_splits - 1))
            test_name = cell_info.iloc[test_name]['DepMap_ID']

            train_index = dataset[dataset['DepMap_ID'].isin(train_name)].index.tolist()
            val_index = dataset[dataset['DepMap_ID'].isin(val_name)].index.tolist()
            test_index = dataset[dataset['DepMap_ID'].isin(test_name)].index.tolist()

            f_train_w = csv.writer(open(root_idx_dir + '/train.index', 'a+'))
            f_val_w = csv.writer(open(root_idx_dir + '/val.index', 'a+'))
            f_test_w = csv.writer(open(root_idx_dir + '/test.index', 'a+'))

            f_train_w.writerow(train_index)
            f_val_w.writerow(val_index)
            f_test_w.writerow(test_index)

            print("[!] Splitting done!")

    for section in ['train', 'val', 'test']:
        with open(root_idx_dir + '/{}.index'.format(section), 'r') as f:
            reader = csv.reader(f)
            split_idx[section] = [list(map(int, idx)) for idx in reader]

    return split_idx


def get_idx_split_drug(dataset, k_splits=5):
    split_idx = {}
    smiles_name_list = pd.read_csv('./data/IC50_GDSC/drug_smiles.csv')[['CanonicalSMILES', 'drug_name']]
    pointer = np.array(list(set(dataset['Drug name'])))

    root_idx_dir = './data_split/{}_fold/drug'.format(k_splits)

    scaffolds = defaultdict(list)
    for i in range(len(smiles_name_list)):
        smiles = smiles_name_list.iloc[i, 0]
        name = smiles_name_list.iloc[i, 1]
        scaffold = generate_scaffold(smiles, include_chirality=True)
        scaffolds[scaffold].append(name)

    if not os.path.exists(root_idx_dir):
        os.makedirs(root_idx_dir)
        cross_val_fold = KFold(n_splits=k_splits, shuffle=True, random_state=0)
        for remain_name, test_name in cross_val_fold.split(scaffolds):
            train_name, val_name = train_test_split(remain_name, test_size=1 / (k_splits - 1))
            train_name, val_name, test_name = pointer[train_name], pointer[val_name], pointer[test_name]

            train_index = dataset[dataset['Drug name'].isin(train_name)].index.tolist()
            val_index = dataset[dataset['Drug name'].isin(val_name)].index.tolist()
            test_index = dataset[dataset['Drug name'].isin(test_name)].index.tolist()

            f_train_w = csv.writer(open(root_idx_dir + '/train.index', 'a+'))
            f_val_w = csv.writer(open(root_idx_dir + '/val.index', 'a+'))
            f_test_w = csv.writer(open(root_idx_dir + '/test.index', 'a+'))

            f_train_w.writerow(train_index)
            f_val_w.writerow(val_index)
            f_test_w.writerow(test_index)

            print("[!] Splitting done!")

    for section in ['train', 'val', 'test']:
        with open(root_idx_dir + '/{}.index'.format(section), 'r') as f:
            reader = csv.reader(f)
            split_idx[section] = [list(map(int, idx)) for idx in reader]

    return split_idx


class EarlyStopping():
    """
    Parameters
    ----------
    mode : str
        * 'higher': Higher metric suggests a better model
        * 'lower': Lower metric suggests a better model
        If ``metric`` is not None, then mode will be determined
        automatically from that.
    patience : int
        The early stopping will happen if we do not observe performance
        improvement for ``patience`` consecutive epochs.
    filename : str or None
        Filename for storing the model checkpoint. If not specified,
        we will automatically generate a file starting with ``early_stop``
        based on the current time.
    metric : str or None
        A metric name that can be used to identify if a higher value is
        better, or vice versa. Default to None. Valid options include:
        ``'r2'``, ``'mae'``, ``'rmse'``, ``'roc_auc_score'``.
    """

    def __init__(self, mode='higher', patience=10, filename=None, metric=None):
        if filename is None:
            dt = datetime.datetime.now()
            folder = os.path.join(os.getcwd(), 'results')
            if not os.path.exists(folder):
                os.makedirs(folder)
            filename = os.path.join(folder, 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
                dt.date(), dt.hour, dt.minute, dt.second))

        if metric is not None:
            assert metric in ['r2', 'mae', 'rmse', 'roc_auc_score', 'pr_auc_score'], \
                "Expect metric to be 'r2' or 'mae' or " \
                "'rmse' or 'roc_auc_score', got {}".format(metric)
            if metric in ['r2', 'roc_auc_score', 'pr_auc_score']:
                print('For metric {}, the higher the better'.format(metric))
                mode = 'higher'
            if metric in ['mae', 'rmse']:
                print('For metric {}, the lower the better'.format(metric))
                mode = 'lower'

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False

    def _check_higher(self, score, prev_best_score):
        """Check if the new score is higher than the previous best score.
        Parameters
        ----------
        score : float
            New score.
        prev_best_score : float
            Previous best score.
        Returns
        -------
        bool
            Whether the new score is higher than the previous best score.
        """
        return score > prev_best_score

    def _check_lower(self, score, prev_best_score):
        """Check if the new score is lower than the previous best score.
        Parameters
        ----------
        score : float
            New score.
        prev_best_score : float
            Previous best score.
        Returns
        -------
        bool
            Whether the new score is lower than the previous best score.
        """
        return score < prev_best_score

    def step(self, score, model):
        """Update based on a new score.
        The new score is typically model performance on the validation set
        for a new epoch.
        Parameters
        ----------
        score : float
            New score.
        model : nn.Module
            Model instance.
        Returns
        -------
        bool
            Whether an early stop should be performed.
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when the metric on the validation set gets improved.
        Parameters
        ----------
        model : nn.Module
            Model instance.
        '''
        torch.save({'model_state_dict': model.state_dict()}, self.filename)

    def load_checkpoint(self, model):
        '''Load the latest checkpoint
        Parameters
        ----------
        model : nn.Module
            Model instance.
        '''
        model.load_state_dict(torch.load(self.filename)['model_state_dict'])


def set_random_seed(seed, deterministic=True):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def init_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, a=0, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            # nn.init.normal_(m.weight, mean=0, std=1e-3)
            nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
            # nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles
    :param smiles:
    :param include_chirality:
    :return: smiles of scaffold
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold


# # test generate_scaffold
# s = 'Cc1cc(Oc2nccc(CCC)c2)ccc1'
# scaffold = generate_scaffold(s)
# assert scaffold == 'c1ccc(Oc2ccccn2)cc1'

def scaffold_split(dataset, smiles_name_list, frac_train=0.6, frac_valid=0.2, frac_test=0.2, seed=42):
    """
    Adapted from https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py
    Split dataset by Bemis-Murcko scaffolds
    This function can also ignore examples containing null values for a
    selected task when splitting. Deterministic split
    :param dataset: pytorch geometric dataset obj
    :param smiles_list: list of smiles corresponding to the dataset obj
    :param task_idx: column idx of the data.y tensor. Will filter out
    examples with null value in specified task column of the data.y tensor
    prior to splitting. If None, then no filtering
    :param frac_train:
    :param frac_valid:
    :param frac_test:
    :return: train, valid, test slices of the input dataset obj. If
    return_smiles = True, also returns ([train_smiles_list],
    [valid_smiles_list], [test_smiles_list])
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    num_drug = len(smiles_name_list)

    np.random.seed(seed)

    scaffolds = defaultdict(list)
    for i in range(num_drug):
        smiles = smiles_name_list.iloc[i, 0]
        name = smiles_name_list.iloc[i, 1]
        scaffold = generate_scaffold(smiles, include_chirality=True)
        scaffolds[scaffold].append(name)

    scaffold_sets = np.random.permutation(list(scaffolds.values()))

    # get train, valid test indices
    train_cutoff = int(frac_train * num_drug)
    valid_cutoff = int((frac_train + frac_valid) * num_drug)
    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0

    train_dataset = dataset[dataset['Drug name'].isin(train_idx)]
    valid_dataset = dataset[dataset['Drug name'].isin(valid_idx)]
    test_dataset = dataset[dataset['Drug name'].isin(test_idx)]

    return train_dataset, valid_dataset, test_dataset

def train_SA(train_loader, model, loss_fn, optimizer, args):
    model.train()
    for step, data in enumerate(tqdm(train_loader, desc='Iteration')):
        drug, cell, ic50 = data
        ic50 = ic50.to(args.device)
        prediction = model(drug, cell)
        loss = loss_fn(prediction, ic50.view(-1, 1).float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Train Loss:{}'.format(loss))
    return loss


def validate_SA(loader, model, args):
    model.eval()
    y_true = []
    y_pred = []
    total_loss = 0
    with torch.no_grad():
        for step, data in enumerate(tqdm(loader, desc='Iteration')):
            drug, cell, ic50 = data
            prediction = model(drug, cell)
            ic50 = ic50.to(args.device)
            total_loss += F.mse_loss(prediction, ic50.view(-1, 1).float(), reduction='sum')
            y_true.append(ic50.view(-1, 1))
            y_pred.append(prediction)

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    rmse = torch.sqrt(total_loss / len(loader.dataset))
    MAE = mean_absolute_error(y_true.cpu(), y_pred.cpu())
    r2 = r2_score(y_true.cpu(), y_pred.cpu())
    r, _= pearsonr(y_true.cpu().numpy().flatten(), y_pred.cpu().numpy().flatten())

    return rmse, MAE, r2, r 

class MyDataset_SA(Dataset):
    def __init__(self, IC):
        super(MyDataset_SA, self).__init__()
        IC.reset_index(drop=True, inplace=True)
        self.drug_name = IC['Drug name']
        self.Cell_line_name = IC['DepMap_ID']
        self.value = IC['IC50']

    def __len__(self):
        return len(self.value)

    def __getitem__(self, index):
        return (drug_name2idx_dict[self.drug_name[index]], cell_id2idx_dict[self.Cell_line_name[index]], self.value[index])
    
    
def load_data_SA(args):
    IC = pd.read_csv('./data/PANCANCER_IC_82833_580_170.csv')
    train_set, val_test_set = train_test_split(IC, test_size=0.2, random_state=42, stratify=IC['Cell line name'])
    val_set, test_set = train_test_split(val_test_set, test_size=0.5, random_state=42,
                                         stratify=val_test_set['Cell line name'])
    train_data, val_data, test_data = MyDataset_SA(train_set), MyDataset_SA(val_set), MyDataset_SA(test_set)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def load_graph_data_SA(args):
    drug_id2graph_dict = np.load('./data/Drugs/drug_feature_graph.npy', allow_pickle=True).item()
    cell_name2feature_dict = np.load('./data/CellLines_DepMap/CCLE_580_18281/census_706/cell_feature_all.npy',
                                        allow_pickle=True).item()
    drug_name = pd.read_csv("./data/Drugs/drug_smiles.csv").iloc[:, 0]
    cell_idx2feature_dict = {u: cell_name2feature_dict[v] for u, v in cell_idx2id_dict.items()}
    drug_idx2graph_dict = {u: drug_id2graph_dict[v] for u, v in enumerate(drug_name)}
    drug_graph = [u for _, u in drug_idx2graph_dict.items()]
    cell_graph = [u for _, u in cell_idx2feature_dict.items()]
    cell_feature_edge_index = np.load(
        './data/CellLines_DepMap/CCLE_580_18281/census_706/edge_index_PPI_{}.npy'.format(args.edge))
    cell_feature_edge_index = torch.tensor(cell_feature_edge_index, dtype=torch.long)
    for u in cell_graph:
        u.edge_index = cell_feature_edge_index
    set_random_seed(args.seed)
    genes_path = './data/CellLines_DepMap/CCLE_580_18281/census_706'
    weight = "TGDRP_pre" if args.pretrain else "TGDRP"
    edge_index = get_STRING_graph(genes_path, args.edge)
    cluster_predefine = get_predefine_cluster(edge_index, genes_path, args.edge, args.device)
    args.num_feature = 3
    model = TGDRP(cluster_predefine, args).to(args.device)
    model.load_state_dict(torch.load('./weights/{}.pth'.format(weight), map_location=args.device)['model_state_dict'])
    parameter = {'drug_emb':model.drug_emb, 'cell_emb':model.cell_emb, 'regression':model.regression}
    drug_nodes = model.GNN_drug(Batch.from_data_list(drug_graph).to(args.device)).detach()
    cell_nodes = model.GNN_cell(Batch.from_data_list(cell_graph).to(args.device)).detach()


    with open("./data/similarity_augment/edge/drug_cell_edges_{}_knn".format(args.knn), 'rb') as f:
        drug_edges, cell_edges = pickle.load(f)
    drug_edges = torch.tensor(drug_edges, dtype=torch.long).t()
    cell_edges = torch.tensor(cell_edges, dtype=torch.long).t()
    return drug_nodes, cell_nodes, drug_edges, cell_edges, parameter

    
