import argparse
import torch
from torch_geometric.data import DataLoader
from gin import GIN
import pandas as pd
from fmnet import FMNet
import rdkit.Chem as Chem
#from molgraphdataset import *
from golden_dataset import *
import pickle
import os
import pickle
device = 'cuda' if torch.cuda.is_available() else 'cpu'

silly_smiles = "O=O"
silly_mol = Chem.MolFromSmiles(silly_smiles)
n_node_features = len(get_atom_features(silly_mol.GetAtomWithIdx(0)))
n_edge_features = len(get_bond_features(silly_mol.GetBondBetweenAtoms(0,1)))

parser = argparse.ArgumentParser()
parser.add_argument('--num_wl_iterations', type = int, default = 3)
parser.add_argument('--node_features_dim', type = int, default = n_node_features)
parser.add_argument('--edge_feature_dim', type = int, default =n_edge_features)
parser.add_argument('--santitize', type = bool, default = False)
parser.add_argument('--embedding_dim', type = int, default=512)
parser.add_argument('--num_layers', type = int, default = 3)
parser.add_argument('--lr', type=float, default = 0.001)
parser.add_argument('--batch_size', type = int, default = 1)

args = parser.parse_args()
test_data = pd.read_csv('MoleculeDataset/test.csv')
test_dataset = MolGraphDataset(test_data, args.num_wl_iterations, santitize=args.santitize)
test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False,follow_batch=['x_r', 'x_p'])


gnn = GIN(args.node_features_dim, args.embedding_dim, num_layers=args.num_layers, cat=True )
model = FMNet(gnn)

print(device)
print(model)
print(args)

gnn =  gnn.to(device)
model = model.to(device)
path = 'experiment1/05_12'

model.load_state_dict(torch.load(f'{path}/model.pth',map_location=torch.device('cpu')))
print(20*'*')

model.eval()
all_h1 = []
all_h3 = []
all_h5 = []
all_h10 = []
all_acc = []

preds = []

total_loss = total_nodes = total_correct = 0
with torch.no_grad():
    for i, data in enumerate(test_loader):
        M0_hat  = model( data.x_r,data.edge_index_r,data.edge_feat_r ,
                                data.x_p, data.edge_index_p,data.edge_feat_p,
                                 data.batch_size) 
                
        M_0 = model.symmetrywise_correspondence_matrix(M0_hat, data.eq_as[0],data.rp_mapper)
        index_r = range(len(data.y_r))
        index_r = torch.tensor(index_r)
        pred = M_0[index_r].argmax(dim=-1)

        preds.append(pred)
        correct = model.acc(M_0, data.y_r, data.rp_mapper, reduction='sum')
        total_nodes = data.y_r.size(0)
        all_acc.append(correct / total_nodes) 
        h1 = model.hits_at_k( 1, M_0, data.y_r,   data.rp_mapper,reduction='mean')
        all_h1.append(h1)
        h3 = model.hits_at_k( 3, M_0, data.y_r,   data.rp_mapper,reduction='mean')
        all_h3.append(h3)
        h5 = model.hits_at_k( 5, M_0, data.y_r,  data.rp_mapper,reduction='mean')
        all_h5.append(h5)
        h10 = model.hits_at_k( 10, M_0, data.y_r,  data.rp_mapper,reduction='mean')
        all_h10.append(h10)

if not os.path.exists(path):
    os.makedirs(path)

with open(f'{path}/all_test_acc.txt', 'wb') as file:
        pickle.dump(all_acc, file)

with open(f'{path}/all_test_h1.txt', 'wb') as file:
        pickle.dump(all_h1 , file)

with open(f'{path}/all_test_h3.txt', 'wb') as file:
        pickle.dump(all_h3 , file)

with open(f'{path}/all_test_h5.txt', 'wb') as file:
        pickle.dump(all_h5 , file)

with open(f'{path}/all_test_h10.txt', 'wb') as file:
        pickle.dump(all_h10 , file)

with open(f'{path}/preds.txt', 'wb') as file:
        pickle.dump(preds , file)
