
import networkx as nx
import random 
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
import pickle
import pandas as pd
import torch
from rdkit.Chem import Draw
from plots import *
from rdkit.Chem import AllChem
import os
pt = Chem.GetPeriodicTable()
import argparse
import torch
from torch_geometric.loader import DataLoader
import pandas as pd
from plots import *
from rdkit import RDLogger        
RDLogger.DisableLog('rdApp.*')      
from rdkit.Chem.Draw import IPythonConsole
from networkx.algorithms import isomorphism



def wl_atom_similarity(mol, num_wl_iterations):
    """
    Get updated atom labels by Weisfeiler-Lehman test.
    Args:
        mol (rdkit.Chem.Mol): The RDKit molecule object.
        num_wl_iterations (int): Number of Weisfeiler-Lehman iterations to perform.
    Returns:
        dict: A dictionary mapping atom indices to their updated labels.
    """
    label_dict = dict()
    for atom in mol.GetAtoms():
        label_dict[atom.GetIdx()]= atom.GetSymbol()

    for _ in range(num_wl_iterations):
        label_dict = update_atom_labels(mol, label_dict)

    return label_dict

def update_atom_labels(mol, label_dict):
    """
    Update atom labels based on the neighbors' labels.
    Args:
        mol (rdkit.Chem.Mol): The RDKit molecule object.
        label_dict (dict): A dictionary mapping atom indices to their current labels.
    Returns:
        dict: A dictionary mapping atom indices to their updated labels.
    """
    new_label_dict = {}

    for atom in mol.GetAtoms():
        neighbors_index = [n.GetIdx() for n in atom.GetNeighbors()]
        neighbors_index.sort()
        label_string = label_dict[atom.GetIdx()]
        for neighbor in neighbors_index:
            label_string += label_dict[neighbor]

        new_label_dict[atom.GetIdx()] = label_string

    return new_label_dict


def get_equivalent_atoms(mol, num_wl_iterations):
    """
    Create a list containing sets of equivalent atoms based on similarity in neighborhood.
    Args:
        mol (rdkit.Chem.Mol): The RDKit molecule object.
        num_wl_iterations (int): Number of Weisfeiler-Lehman iterations to perform.
    Returns:
        list: A list of sets where each set contains atom indices of equivalent atoms.
    """    
    node_similarity = wl_atom_similarity(mol, num_wl_iterations)
    n_h_dict = dict()
    for atom in mol.GetAtoms():
        n_h_dict[atom.GetIdx()]= atom.GetTotalNumHs()
    degree_dict = dict()
    for atom in mol.GetAtoms():
        degree_dict[atom.GetIdx()] = atom.GetDegree()
    neighbor_dict = dict()
    for atom in mol.GetAtoms():
        neighbor_dict[atom.GetIdx()]= [nbr.GetSymbol() for nbr in atom.GetNeighbors()]
        
    atom_equiv_classes = []
    visited_atoms = set()
    for centralnode_indx, centralnodelabel in node_similarity.items():
        equivalence_class = set()

        if centralnode_indx not in visited_atoms:
            visited_atoms.add(centralnode_indx) 
            equivalence_class.add(centralnode_indx)

        for firstneighbor_indx, firstneighborlabel in node_similarity.items():
            if firstneighbor_indx not in visited_atoms and centralnodelabel[0] == firstneighborlabel[0] and \
                    set(centralnodelabel[1:]) == set(firstneighborlabel[1:]) and \
                    degree_dict[centralnode_indx] == degree_dict[firstneighbor_indx]  and \
                    len(centralnodelabel)== len(firstneighborlabel) and \
                    set(neighbor_dict[centralnode_indx]) == set(neighbor_dict[firstneighbor_indx]) and \
                    n_h_dict[centralnode_indx] == n_h_dict[firstneighbor_indx]:
                    equivalence_class.add(firstneighbor_indx)
                    visited_atoms.add(firstneighbor_indx)
        if equivalence_class :
            atom_equiv_classes.append(equivalence_class)
          
    return atom_equiv_classes


def get_atom_matcher(mol1, mol2, num_wl_iterations = 3):

    node_similarity1 = wl_atom_similarity(mol1, num_wl_iterations)
    n_h_dict1 = dict()
    degree_dict1 = dict()
    neighbor_dict1 = dict()
    for atom in mol1.GetAtoms():
        n_h_dict1[atom.GetIdx()]= atom.GetTotalNumHs()
        degree_dict1[atom.GetIdx()] = atom.GetDegree()
        neighbor_dict1[atom.GetIdx()]= [nbr.GetSymbol() for nbr in atom.GetNeighbors()]

    node_similarity2 = wl_atom_similarity(mol2, num_wl_iterations)
    n_h_dict2 = dict()
    degree_dict2 = dict()
    neighbor_dict2 = dict()
    for atom in mol2.GetAtoms():
        n_h_dict2[atom.GetIdx()]= atom.GetTotalNumHs()
        degree_dict2[atom.GetIdx()] = atom.GetDegree()
        neighbor_dict2[atom.GetIdx()]= [nbr.GetSymbol() for nbr in atom.GetNeighbors()]
        
    mapper = dict()
    visited_atoms1 = []
    visited_atoms2 = []

    for centralnode_indx1, centralnodelabel1 in node_similarity1.items():
        for centralnode_indx2, centralnodelabel2 in node_similarity2.items():
            if centralnode_indx1 not in visited_atoms1 and centralnode_indx2 not in visited_atoms2:

                if centralnodelabel1[0] == centralnodelabel2[0] and \
                        set(centralnodelabel1) == set(centralnodelabel2) and \
                        len(centralnodelabel1)== len(centralnodelabel2)and \
                        set(neighbor_dict1[centralnode_indx1]) == set(neighbor_dict2[centralnode_indx2]) :

                        visited_atoms1.append(centralnode_indx1) 
                        visited_atoms2.append(centralnode_indx2) 


                        mapper[centralnode_indx1]= centralnode_indx2

    return mapper


def get_reaction_mols(reaction_smiles, santitize=False):
    """
    Returns reactant and product molecules from a reaction SMILES string.
    Args:
        reaction_smiles: Reaction SMILES string.
    Returns:
        A tuple containing reactant and product RDKit molecule objects.
    """
    reactantes_smiles, products_smiles = reaction_smiles.split('>>')
    reactantes_mol = Chem.MolFromSmiles(reactantes_smiles)
    products_mol = Chem.MolFromSmiles(products_smiles)
    if santitize:
        Chem.SanitizeMol(reactantes_mol)
        Chem.SanitizeMol(products_mol)
    return reactantes_mol, products_mol 
  
def get_reaction_componant(rxn):
    r, p = rxn.split('>>')
    return r,p

def get_map(mol):
     return[atom.GetAtomMapNum()-1 for atom in mol.GetAtoms()]

def get_atomidx2mapidx(mol):
    atomidx2mapidx = {}
    for atom in mol.GetAtoms():
        atomidx2mapidx[atom.GetIdx()] = atom.GetAtomMapNum()-1
    return atomidx2mapidx


def get_mol_from_golden_dataset(gt_dataset,reaction_ind ):
    r = gt_dataset.iloc[reaction_ind][0]
    r_mol, p_mol = get_reaction_mols(r)
    
    return  r_mol, p_mol

def get_mol_from_rxnmapper(pred_data_set,reaction_ind):
    rxn = pred_data_set.mapped_rxn.iloc[reaction_ind]
    reactants_rxn, products_rxn = get_reaction_componant(rxn)
    r_mol_rxn = Chem.MolFromSmiles(reactants_rxn)
    p_mol_rxn = Chem.MolFromSmiles(products_rxn) 
    
    return  r_mol_rxn, p_mol_rxn


def get_atom_correspondance_rxnmapper(gt_dataset, pred_data_set, reaction_ind):
    r_mol, p_mol = get_mol_from_golden_dataset(gt_dataset,reaction_ind )
    r_mol_rxn, p_mol_rxn =  get_mol_from_rxnmapper(pred_data_set,reaction_ind)

    y_r = get_map(r_mol)
    y_p = get_map(p_mol)

    eq_as =  get_equivalent_atoms(p_mol_rxn, num_wl_iterations=3)

    gt =[y_p.index(element) for element in y_r]

    m_r_rprim = get_atom_matcher(r_mol, r_mol_rxn, num_wl_iterations = 3)
    m_p_pprim = get_atom_matcher(p_mol, p_mol_rxn, num_wl_iterations = 3)

    correspondace = [ (m_r_rprim[i],m_p_pprim[gt[i]]) for i in y_r]
    
    return correspondace, eq_as



def get_symmetry_gt_rxnmapper(gt_dataset, pred_data_set, reaction_ind):

    correspondace ,eq_as= get_atom_correspondance_rxnmapper(gt_dataset, pred_data_set, reaction_ind)
    symmetry_gt = []
    for pair in correspondace:
        r, p = pair
        symmetry_gt.append(pair)
        for group in eq_as:
            if len(group) > 1:
                indices = list(group)
                for p_ind in indices:
                    if r == p_ind:
                        if p in indices:
                            if (r,p_ind) not in symmetry_gt:
                                    symmetry_gt.append((r,p_ind))

    return(symmetry_gt)

def get_predictions_react2prod_rxnmapper(pred_data_set, reaction_ind):
    r_mol_rxn, p_mol_rxn =  get_mol_from_rxnmapper(pred_data_set,reaction_ind)

    y_r_rxn = get_map(r_mol_rxn)
    y_p_rxn= get_map(p_mol_rxn)
    predictions =[(element,y_p_rxn.index(element)) for element in y_r_rxn]
    return predictions


def get_symmetry_gt_amnet(gt_dataset, reaction_ind):
    r_mol, p_mol =  get_mol_from_golden_dataset(gt_dataset,reaction_ind)
    y_r = get_map(r_mol)
    y_p = get_map(p_mol)
    eq_as =  get_equivalent_atoms(p_mol, num_wl_iterations=3)

    gt =[y_p.index(element) for element in y_r]
    gt_correspondace = [ (i,gt[i]) for i in range(len(y_r))]
    symmetry_gt = []
    for pair in gt_correspondace:
        r, p = pair
        symmetry_gt.append(pair)
        for group in eq_as:
            if len(group) > 1:
                indices = list(group)
                for p_ind in indices:
                    if r == p_ind:
                        if p in indices:
                            if (r,p_ind) not in symmetry_gt:
                                    symmetry_gt.append((r,p_ind))

    return(symmetry_gt)

def get_predictions_react2prod_amnet(preds, reaction_ind):
    product_pred = preds[reaction_ind]
    p_np = [p.item() for p in product_pred]
    predictions =[(element,p_np[element]) for element in range(len(p_np))]
    return predictions


def num_missing_values(symmetry_gt, predictions):
    missing_values = 0

    for item in predictions:
        if item not in symmetry_gt:
            missing_values += 1

    return missing_values

def get_missing_values(symmetry_gt, predictions):
    missing_values = []

    for item in predictions:
        if item not in symmetry_gt:
            missing_values.append(item)

    return missing_values


def calculate_accuracy(symmetry_gt, predictions):
    missing_values = num_missing_values(symmetry_gt, predictions)
    shorter_length = min(len(symmetry_gt), len(predictions))
    accuracy = (shorter_length - missing_values) / shorter_length * 100.0

    return accuracy








if __name__ == '__main__':
    test_data = pd.read_csv('golden_dataset/processed_golden4.csv', header=None)

    rxn_csv =  pd.read_csv('rxnmapper_prediction_golden_dataset.csv')
    path = 'experiment1/05_10_edege_feature_golden_real'

    with open(f"{path}/preds.txt", "rb") as fp:   
                preds = pickle.load(fp)

    all_acc_rxnmapper = []
    all_acc_amnet = []

    for reaction_ind in range(185):
            predictions_rxnmapper =get_predictions_react2prod_rxnmapper(rxn_csv, reaction_ind)
            symmetry_gt_rxnmapper = get_symmetry_gt_rxnmapper(test_data, rxn_csv, reaction_ind)
            acc_rxnmapper = calculate_accuracy(symmetry_gt_rxnmapper, predictions_rxnmapper)
            all_acc_rxnmapper.append(acc_rxnmapper)
            symmetry_gt_amnet = get_symmetry_gt_amnet(test_data, reaction_ind)
            predictions_amnet = get_predictions_react2prod_amnet(preds, reaction_ind)
            acc_amnet = calculate_accuracy(symmetry_gt_amnet, predictions_amnet)
            all_acc_amnet.append(acc_amnet)


    mean_accuracy_rxnmapper = np.mean(all_acc_rxnmapper)
    print(f"Mean Accuracy of RXNMapper: {mean_accuracy_rxnmapper}")
    mean_accuracy_amnet = np.mean(all_acc_amnet)
    print(f"Mean Accuracy of RXNMapper: {mean_accuracy_amnet}")