import argparse
import torch
from torch_geometric.data import DataLoader
from gin import GIN
import pandas as pd
from amnet import AMNet
import rdkit.Chem as Chem
from molgraphdataset import *
#from golden_test import *
from plots import *
import os
import pickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'

silly_smiles = "O=O"
silly_mol = Chem.MolFromSmiles(silly_smiles)
n_node_features = len(get_atom_features(silly_mol.GetAtomWithIdx(0)))
n_edge_features = len(get_bond_features(silly_mol.GetBondBetweenAtoms(0,1)))
print(n_node_features)
parser = argparse.ArgumentParser()
parser.add_argument('--num_wl_iterations', type = int, default = 3)
parser.add_argument('--node_features_dim', type = int, default = n_node_features)
parser.add_argument('--edge_feature_dim', type = int, default = None)#n_edge_features
parser.add_argument('--santitize', type = int, default = False)
parser.add_argument('--embedding_dim', type = int, default=512)
parser.add_argument('--second_embedding_dim', type=int, default=256)
parser.add_argument('--num_layers', type = int, default = 3)
parser.add_argument('--lr', type=float, default = 0.0001)
parser.add_argument('--num_concensus_iter', type=int, default=10)
parser.add_argument('--n_epochs', type = int, default = 35)
parser.add_argument('--batch_size', type = int, default = 1)


args = parser.parse_args()


test_data = pd.read_csv('MoleculeDataset/vcna_train.csv')
#test_data = pd.read_csv('golden_dataset/processed_golden2.csv', header=None)
test_dataset = MolGraphDataset(test_data.iloc[:306], args.num_wl_iterations, santitize=args.santitize)
test_loader = DataLoader(test_dataset, args.batch_size, shuffle=True,follow_batch=['x_r', 'x_p'])

gnn_1 = GIN(args.node_features_dim, args.embedding_dim, num_layers=args.num_layers, cat=False )
gnn_2 = GIN(args.second_embedding_dim, args.second_embedding_dim, num_layers=args.num_layers, cat=True)
model = AMNet(gnn_1, gnn_2, num_concensus_iter=args.num_concensus_iter)

print(device)
print(model)
print(args)



model.load_state_dict(torch.load('experiment2/28_09_t5/model.pth',map_location=torch.device('cpu')))
model.eval()

correct = num_examples = 0



print(20*'*')

model.eval()
all_h1 = []
all_h3 = []
all_h5 = []
all_h10 = []
all_acc = []
total_loss = total_nodes = total_correct = 0
with torch.no_grad():
    for data in test_loader:
        M0_hat, Mt_hat  = model( data.x_r,data.edge_index_r,None,#data.edge_feat_r,
                                data.x_p, data.edge_index_p,None, #data.edge_feat_p,
                                 data.batch_size)#
                
        #M_0 = model.adjust_correspondence_matrix_for_symmetry(M0_hat, data.eq_as[0])
        M_T = model.symmetrywise_correspondence_matrix(Mt_hat, data.eq_as[0],data.rp_mapper)
        correct = model.acc(M_T,  data.y_r, data.rp_mapper, reduction='none')
        num_examples = data.y_r.size(0)  
        all_acc.append(correct / num_examples) 
        h1 = model.hits_at_k( 1, M_T, data.y_r,  data.rp_mapper,reduction='mean')
        all_h1.append(h1)
        h3 = model.hits_at_k( 3, M_T, data.y_r, data.rp_mapper,reduction='mean')
        all_h3.append(h3)
        h5 = model.hits_at_k( 5, M_T, data.y_r, data.rp_mapper,reduction='mean')
        all_h5.append(h5)
        h10 = model.hits_at_k( 10, M_T, data.y_r, data.rp_mapper,reduction='mean')
        all_h10.append(h10)


path = 'experiment2/28_09_t5/t10'

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
