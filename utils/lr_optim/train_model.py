import torch as th
import numpy as np
import dgl
#from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import sys
sys.path.append("/home/shenchao/sdegen")
from sdegen.data.data import PDBbindDataset
from sdegen.model.model import InteractNet, LigandNet, TargetNet
from sdegen.model.utils import EarlyStopping, set_random_seed, run_a_train_epoch, run_an_eval_epoch
#import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')

args={}
args["datadir"] = "/home/shenchao/sdegen/dataset"
args["data_prefix"] = "v2020_train"
args["valnum"] = 1500
args['seeds'] = 126
args["batch_size"] = 16
args["num_workers"] = 10
args['device'] = 'cuda' if th.cuda.is_available() else 'cpu'
args["hidden_dim0"] = 128
args["residual_layers"] = 10
args["hidden_dim"] = 128
args["dropout_rate"] = 0.15


data = PDBbindDataset(ids="%s/%s_ids.npy"%(args["datadir"], args["data_prefix"]),
					  ligs="%s/%s_lig.pt"%(args["datadir"], args["data_prefix"]),
					  prots="%s/%s_prot.pt"%(args["datadir"], args["data_prefix"])
					  )

train_inds, val_inds = data.train_and_test_split(valnum=args["valnum"], seed=args['seeds'])
train_data = PDBbindDataset(ids=data.pdbids[train_inds],
					  ligs=data.gls[train_inds],
					  prots=data.gps[train_inds]
					  )
val_data = PDBbindDataset(ids=data.pdbids[val_inds],
					  ligs=data.gls[val_inds],
					  prots=data.gps[val_inds]
					  )


train_loader = DataLoader(train_data, 
							batch_size=args["batch_size"], 
							shuffle=True, 
							num_workers=args["num_workers"]
							)

#batch_id, batch_data = next(enumerate(train_loader))
#pdbids, bgp, bgl = batch_data


ligmodel = LigandNet(in_channels=41, 
					edge_features=11, 
					hidden_dim=args["hidden_dim0"], 
					residual_layers=args["residual_layers"], 
					dropout_rate=0.15)

protmodel = TargetNet(in_channels=4,
					edge_features=1, 
					hidden_dim=args["hidden_dim0"], 
					residual_layers=args["residual_layers"], 
					dropout_rate=0.15)

model = InteractNet(ligmodel, protmodel, 
				in_channels=args["hidden_dim0"], 
				hidden_dim=args["hidden_dim"], 
				dropout_rate=args["dropout_rate"]).to(args['device'])







