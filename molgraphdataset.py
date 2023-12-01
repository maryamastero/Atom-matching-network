import torch
import numpy as np
from torch_geometric.data import Data
import collections
import rdkit.Chem as Chem
pt = Chem.GetPeriodicTable()
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')    
import pandas as pd 
from torch_geometric.data import DataLoader
import random
from utils import *
from wl import *

"""
'GraphData' is used to encapsulate information about a molecular graph, making it easier to work with graph data. It includes the following fields:
- n_nodes: The total number of nodes in the graph.
- node_features: A tensor containing node features.
- edge_features: A tensor containing edge features.
- edge_index: A tensor representing the edge indices.
"""

GraphData = collections.namedtuple('GraphData', [
                                    'n_nodes',
                                    'node_features',
                                    'edge_features',
                                    'edge_index'])


class PairData(Data):
    """
    Adapted from https://pytorch-geometric.readthedocs.io/en/latest/advanced/batching.html#pairs-of-graphs

    Args:
        edge_index_r: Edge indices for the reactant graph.
        x_r: Node features for the reactant graph.
        edge_index_p: Edge indices for the product graph.
        x_p: Node features for the product graph.
        edge_feat_r: Edge features for the reactant graph.
        edge_feat_p: Edge features for the product graph.
        y_r: A list of atom mapping value based on graph traverse (atom indices) for the reactant graph.
        y_p: A list of atom mapping value based on graph traverse (atom indices) for the product graph.
        rp_mapper: A mapper function to maps atoms in reactant to products. It is important after permuting the product graphs to be sure the model just not learn diagonal!
        eq_as: Equivalent atoms to consider molecule symmetry for product graph.
    """
    def __init__(self, edge_index_r=None, x_r=None, edge_index_p=None, x_p=None,  \
                 edge_feat_r = None, edge_feat_p = None, y_r = None, y_p = None,  rp_mapper = None, eq_as = None):
        super().__init__()
        self.edge_index_r = edge_index_r
        self.x_r = x_r
        self.edge_index_p = edge_index_p
        self.x_p = x_p
        self.y_r = y_r
        self.y_p = y_p
        self.edge_feat_r = edge_feat_r
        self.edge_feat_p = edge_feat_p
        self.rp_mapper = rp_mapper
        self.eq_as = eq_as

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_r':
            return self.x_r.size(0)
        if key == 'edge_index_p':
            return self.x_p.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)



class MolGraphDataset(torch.utils.data.Dataset):
    """
    MolGraphDataset pytorch dataset
    Args:
        reactions: List of reaction SMILES strings.
        num_wl_iterations (int): Number of iterations used for finding chemically equivalent atoms
            in a molecule by WL isomorphism test.
        santitize (bool): Whether to sanitize the molecules during graph creation.
    """

    def __init__(self,reactions, num_wl_iterations, santitize=False):
        self.reactions = reactions
        self.num_wl_iterations = num_wl_iterations
        self.santitize= santitize

    def __len__(self):
        """
        Gets the number of reaction in the dataset.
        Returns:
            int: Number of reactions in dataset.
        """
        return len(self.reactions)

    
    def __getitem__(self, idx):
        """
        Gets an index of a reaction dataset.
        Parameters:
            idx (int): Gets a reaction from the dataset by index.

        Returns:
            PairData: A PairData object containing information about reactant and product molecules,
                    their graph representations, equivalent atoms, and atom mappings.
        """
        reaction_smiles = self.reactions.iloc[idx, 0]
        graph_edits = self.reactions.iloc[idx, 1]

        reactantes_mol, _ = get_reaction_mols(reaction_smiles) # reactant
        generated_product = get_random_order_product(reaction_smiles, graph_edits) # reordered product
        #generated_product = product_by_editing_subsrate(reaction_smiles, graph_edits)
           
        reactant_graph = get_graph_data_tensor(reactantes_mol)
        product_graph = get_graph_data_tensor(generated_product)

        edge_index_r = reactant_graph.edge_index
        x_r = reactant_graph.node_features

        edge_index_p = product_graph.edge_index
        x_p = product_graph.node_features

        edge_feat_r = reactant_graph.edge_features
        edge_feat_p = product_graph.edge_features

        #eq_a_r = get_equivalent_atoms(reactantes_mol, self.num_wl_iterations)
        eq_as = get_equivalent_atoms(generated_product, self.num_wl_iterations)# equivalent atoms in product

        y_r_np = get_mapping_number(reactantes_mol)
        y_p_np = get_mapping_number(generated_product)
        y_p = torch.tensor(y_p_np) 
        y_r = torch.tensor(y_r_np)

        y_p_prim_np =[y_p_np.index(element) for element in y_r_np]
        rp_mapper = torch.tensor(y_p_prim_np)  #reactant to product mapper

        data = PairData(
                        edge_index_r, x_r, edge_index_p, x_p, edge_feat_r=edge_feat_r,
                        edge_feat_p=edge_feat_p, y_r=y_r, y_p=y_p,rp_mapper=rp_mapper, eq_as=eq_as
                        )
        data.batch_size = torch.zeros(x_r.shape[0], dtype=int)
        return data


    
def one_hot_encoding(x, permitted_list):
    """
    Map input elements x, which are not in the permitted list, to the last element of the permitted list.
    Args:
        x (str or int): The input element to one hot encode.
        permitted_list (list): List of permitted elements.
    Returns:
         list: Binary encoding of the input element based on the permitted list.
    """
    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]

    return binary_encoding

def get_atom_features(atom, use_chirality = True, hydrogens_implicit = True): # I change both to False True
    '''
    Gets an RDKit atom object as input and return a 1D numpy array of atom features.
    Args:
        atom: The RDKit atom object to extract features from.
        use_chirality: Whether to include chirality information.
        hydrogens_implicit: Whether to include implicit hydrogen information.
    Returns:
        Array of atom features.
    '''
    #list of permitted atoms
    permitted_list_of_atoms = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', \
                                'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', \
                                'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', \
                                'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', \
                                'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', \
                                'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', \
                                'Os', 'Ir', 'Ce','Gd','Ga','Cs', 'unknown']


    if hydrogens_implicit == False:
        permitted_list_of_atoms = ['H'] + permitted_list_of_atoms
    
    # compute atom features
    atom_type  = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
    n_heavy_neighbors  = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])
    formal_charge  = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
    hybridisation_type  = one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
    ex_valence = one_hot_encoding(int(atom.GetExplicitValence()), list(range(1, 7)))
    imp_valence = one_hot_encoding(int(atom.GetImplicitValence()), list(range(0, 6)))
    is_in_a_ring = [int(atom.IsInRing())]
    is_aromatic = [int(atom.GetIsAromatic())]
    atomic_mass_scaled = [float((atom.GetMass() - 10.812)/116.092)]
    vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6)] 
    covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76)]
    atom_feature_vector =   atom_type  + n_heavy_neighbors + is_in_a_ring  + is_aromatic  \
                          + ex_valence + imp_valence  + atomic_mass_scaled\
                         + vdw_radius_scaled + covalent_radius_scaled  + hybridisation_type + formal_charge                                
    #
    if use_chirality == True:
        chirality_type  = one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        atom_feature_vector += chirality_type
    
    if hydrogens_implicit == True:
        n_hydrogens  = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens
    #atom_feature_vector2 = atom_type + is_aromatic + ex_valence     
    return np.array(atom_feature_vector)


def get_bond_features(bond, use_stereochemistry = True):
    '''
    Gets an RDKit bond object as input and return a 1D numpy array of bond features.
    Args:
        bond: The RDKit bond object to extract bond features.
        use_stereochemistry (bool): Whether to include stereochemistry information.
    Returns:
        Array of bond features.
    '''
    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]

    bond_type  = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
    bond_is_conj  = [int(bond.GetIsConjugated())]
    bond_is_in_ring  = [int(bond.IsInRing())]
    bond_feature_vector = bond_type  + bond_is_conj  + bond_is_in_ring

    if use_stereochemistry == True:
        stereo_type  = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
        bond_feature_vector += stereo_type

    return np.array(bond_feature_vector)

def get_graph_data_tensor(mol):
        """
        Gets RDKit molecule object and return a graph data object containing tensors.
        Args:
            mol: RDKit molecule object.
        Returns:
            GraphData: Graph data object containing molecule information.
        """
        n_nodes = mol.GetNumAtoms()
        n_edges = 2*mol.GetNumBonds()
        #to have the feature dimention we use a silly mol

        silly_smiles = "O=O"
        silly_mol = Chem.MolFromSmiles(silly_smiles)
        n_node_features = len(get_atom_features(silly_mol.GetAtomWithIdx(0)))
        n_edge_features = len(get_bond_features(silly_mol.GetBondBetweenAtoms(0,1)))
        
        X = np.zeros((n_nodes, n_node_features))
        for atom in mol.GetAtoms():
            X[atom.GetIdx(), :] = get_atom_features(atom)
            
        X = torch.tensor(X, dtype = torch.float)
        
        (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        E = torch.stack([torch_rows, torch_cols], dim = 0)
        
        EF = np.zeros((n_edges, n_edge_features))
        for (k, (i,j)) in enumerate(zip(rows, cols)):
            EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))
        EF = torch.tensor(EF, dtype = torch.float)

        return GraphData(
                n_nodes,
                node_features= X,
                edge_features =EF,
                edge_index= E)



if __name__ =='__main__':

    train_data = pd.read_csv('MoleculeDataset/vcna_train.csv', names= ['reactions', 'edits'])
    train_dataset = MolGraphDataset(train_data, 3, False)
    train_loader = DataLoader(train_dataset, 3, shuffle=False,follow_batch=['x_s', 'x_t'])
    for i, data in enumerate(train_loader):
        print(data)
        if i == 2:
            break
