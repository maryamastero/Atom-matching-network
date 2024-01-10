import argparse
import torch
from torch_geometric.data import DataLoader
from gin import GIN
import pandas as pd
from amnet import AMNet
import rdkit.Chem as Chem
from molgraphdataset import *
import pickle
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

silly_smiles = "O=O"
silly_mol = Chem.MolFromSmiles(silly_smiles)
n_node_features = len(get_atom_features(silly_mol.GetAtomWithIdx(0)))
n_edge_features = len(get_bond_features(silly_mol.GetBondBetweenAtoms(0,1)))

parser = argparse.ArgumentParser()
parser.add_argument('--num_wl_iterations', type = int, default = 3)
parser.add_argument('--node_features_dim', type = int, default = n_node_features)
parser.add_argument('--edge_feature_dim', type = int, default = None)#n_edge_features
parser.add_argument('--santitize', type = int, default = False)
parser.add_argument('--embedding_dim', type = int, default=128)
parser.add_argument('--second_embedding_dim', type=int, default=64)
parser.add_argument('--num_layers', type = int, default = 3)
parser.add_argument('--lr', type=float, default = 0.0001)
parser.add_argument('--num_concensus_iter', type=int, default=200)
parser.add_argument('--n_epochs', type = int, default = 5)
parser.add_argument('--batch_size', type = int, default = 1)

args = parser.parse_args()


train_data = pd.read_csv('MoleculeDataset/vcna_train.csv', names= ['reactions', 'edits'])
valid_data =pd.read_csv('MoleculeDataset/vcna_valid.csv', names= ['reactions', 'edits'])
whole_data = pd.concat([train_data, valid_data])
train_dataset = MolGraphDataset(whole_data , args.num_wl_iterations, santitize=args.santitize)
train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True,follow_batch=['x_r', 'x_p'])

validation_dataset = MolGraphDataset(valid_data, args.num_wl_iterations, santitize=args.santitize)
validation_loader = DataLoader(validation_dataset, args.batch_size, shuffle=False,follow_batch=['x_r', 'x_p'])


gnn_1 = GIN(args.node_features_dim, args.embedding_dim, num_layers=args.num_layers, cat=False )
gnn_2 = GIN(args.second_embedding_dim, args.second_embedding_dim, num_layers=args.num_layers, cat=True)
model = AMNet(gnn_1, gnn_2, num_concensus_iter=args.num_concensus_iter)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.95, 0.999))
print(model)
print(args)

gnn_1 =  gnn_1.to(device)
gnn_2 =  gnn_2.to(device)
model = model.to(device)

best_valid_loss = float('inf')
patience = 20  # Number of epochs to wait for improvement
counter = 0  # Counter to keep track of epochs without improvement




def train(train_loader):
    model.train()
    total_loss = total_nodes = total_correct = 0
    for i, data in enumerate(train_loader):   
        optimizer.zero_grad()
        data = data.to(device)
        M0_hat, Mt_hat  = model( data.x_r,data.edge_index_r,None,  #data.edge_feat_r
                                data.x_p, data.edge_index_p, None,# data.edge_feat_p
                                data.batch_size)

        M_0 = model.adjust_correspondence_matrix_for_symmetry(M0_hat, data.eq_as[0])
        M_T = model.adjust_correspondence_matrix_for_symmetry(Mt_hat, data.eq_as[0])

        loss = model.loss(M_0, data.y_r, data.y_p)
        loss = model.loss(M_T, data.y_r, data.y_p) + loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        total_correct += model.acc(M_T , data.y_r, data.y_p, reduction='sum')
        total_nodes += data.y_r.size(0)
        
        
    return total_loss/len(train_loader), total_correct /total_nodes


def validation_loss(loader):
    model.eval()
    total_loss = total_nodes = total_correct = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            M0_hat, Mt_hat  = model( data.x_r,data.edge_index_r,None,  #data.edge_feat_r
                        data.x_p, data.edge_index_p, None, data.batch_size)# data.edge_feat_p
        
            M_0 = model.adjust_correspondence_matrix_for_symmetry(M0_hat, data.eq_as[0])
            M_T = model.adjust_correspondence_matrix_for_symmetry(Mt_hat, data.eq_as[0])

            loss = model.loss(M_0, data.y_r, data.y_p)
            loss = model.loss(M_T, data.y_r, data.y_p) + loss

            total_loss += loss.item()

            total_correct += model.acc(M_T, data.y_r, data.y_p, reduction='mean')
            total_nodes += data.y_r.size(0)
            
            
    return total_loss/len(loader), total_correct/total_nodes

all_train_loss = []
all_train_acc = []

all_valid_loss = []
all_valid_acc = []

for epoch in range(1, args.n_epochs+1):
    print(f'Epoch: {epoch:02d}', 5*'*')
    train_loss , train_acc = train(train_loader)
    #print(train_loss , train_acc)
    valid_loss , valid_acc  = validation_loss(validation_loader)

    all_train_loss.append(train_loss)
    all_train_acc.append(train_acc)

    all_valid_loss.append(valid_loss)
    all_valid_acc.append(valid_acc)
   

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        counter = 0  
    else:
        counter += 1  
        
    if counter >= patience:
        print(f'Early stopping. No improvement in {epoch} epochs.')
        break
    
path = 'experiment2/25_09_t5'

if not os.path.exists(path):
    os.makedirs(path)


torch.save(model.state_dict(), f'{path}/model.pth')


with open(f'{path}/losses_train.txt', 'wb') as file:
       pickle.dump(all_train_loss, file)
       
with open(f'{path}/acces_train.txt', 'wb') as file:
       pickle.dump(all_train_acc, file)


with open(f'{path}/losses_valid.txt', 'wb') as file:
       pickle.dump(all_valid_loss, file)

with open(f'{path}/acces_valid.txt', 'wb') as file:
       pickle.dump(all_valid_acc, file)



        
