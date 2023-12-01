from rdkit import Chem
import os
pt = Chem.GetPeriodicTable()
import pandas as pd
from utils import *


def is_valid_molecule(mol):
    if mol is None:
        return False
    try:                        
        Chem.SanitizeMol(mol)
        return True
    except Exception as e:
        return False

def get_invalid_molecule_index(df):
    invalid_indices = []
    for index, row in df.iterrows():
        reaction_smiles, graph_edits = get_reaction_info(df, index)
        reactantes_mol, products_mol = get_reaction_mols(reaction_smiles, santitize=False)
        gp_mol = product_by_editing_subsrate(reaction_smiles,graph_edits, santitize=False)
        if is_valid_molecule(gp_mol):
            continue
        else:
            invalid_indices.append(index)   
    return invalid_indices


def validity_checked(test_data, name):
    path = 'MolGraphDataset'
    if not os.path.exists(path):
        os.makedirs(path)
    problematic_i = get_invalid_molecule_index(test_data)
    processed_data =  test_data.drop(index=problematic_i)  
    processed_data = processed_data.reset_index(drop=True)  
    print('# of reactons in ', name, len(processed_data))
    processed_data.to_csv(f'{path}/vc_{name}.csv', index=False, header=None)


def get_not_equal_num_atoms(df):
    invalid_indices = []
    for index, row in df.iterrows():
        reaction_smiles, graph_edits = get_reaction_info(df, index)
        reactantes_mol, products_mol = get_reaction_mols(reaction_smiles, santitize=False)
        gp_mol = product_by_editing_subsrate(reaction_smiles,graph_edits, santitize=False)
        if reactantes_mol.GetNumAtoms() !=gp_mol.GetNumAtoms():
            invalid_indices.append(index)
    return invalid_indices

def atom_number_checked(test_data, name):
    path = 'MolGraphDataset'
    if not os.path.exists(path):
        os.makedirs(path)
    problematic_i = get_not_equal_num_atoms(test_data)
    processed_data =  test_data.drop(index=problematic_i)  
    processed_data = processed_data.reset_index(drop=True)  
    print('# of reactons in ', name, len(processed_data))

    processed_data.to_csv(f'{path}/vanc_{name}.csv', index=False, header=None)


def get_problematic_data(df, santitize):
    '''
    Returns index of problematic reactions
    '''
    problematic_i = []
    for i in range(len(test_data)):
        try:
            reaction_smiles, graph_edits = get_reaction_info(df, i)
            mw =  product_by_editing_subsrate(reaction_smiles,graph_edits, santitize)
        except: 
            problematic_i.append(i)
            continue
    return problematic_i



def preprocess_data(test_data, name, santitize=True):
    path = 'MolGraphDataset'

    if not os.path.exists(path):
        os.makedirs(path)
    problematic_i = get_problematic_data(test_data, santitize)
    processed_data =  test_data.drop(problematic_i)    
    processed_data.to_csv(f'{path}/{name}.csv', index=False, header=None)


if __name__ =='__main__':

    print('USPTO-15k dataset ...' )     
    train_data = pd.read_csv("../dataset/USPTO-15k/train.txt", header=None, sep=' ')
    test_data = pd.read_csv("../dataset/USPTO-15k/test.txt", header=None, sep=' ')
    valid_data = pd.read_csv("../dataset/USPTO-15k/valid.txt", header=None, sep=' ')

    print('Filtering invalid mols ...' )

    validity_checked(train_data, 'train')
    validity_checked(test_data, 'test')
    validity_checked(valid_data, 'valid')

    print('Filtering not equals atom numbers ...' )
    train_data = pd.read_csv("MolGraphDataset/vc_train.csv") #5628
    test_data = pd.read_csv("MolGraphDataset/vc_test.csv") #+[455, 1318, 1379])  
    valid_data = pd.read_csv("MolGraphDataset/vc_valid.csv")

    atom_number_checked(train_data, 'train')
    atom_number_checked(test_data, 'test')
    atom_number_checked(valid_data, 'valid')
    print('done...')


