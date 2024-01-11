import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import pickle
from rdkit import Chem
pt = Chem.GetPeriodicTable()
import statistics
import random

def get_reaction(df, i):
    reaction_smiles = df.iloc[i,0]
    graph_edits = df.iloc[i,1]
    precursures_smiles, products_smiles = reaction_smiles.split('>>')
    precursures_mol = Chem.MolFromSmiles(precursures_smiles)
    products_mol = Chem.MolFromSmiles(products_smiles)

    return precursures_mol, products_mol, graph_edits

def product_by_editing_subsrate(df, i, santitize=False):
    'Create product from reactant and edits'
    r, _, graph_edits = get_reaction(df,i )
    mw = Chem.RWMol(r)
    removeh = graph_edits.split(';')[0]
    addh = graph_edits.split(';')[1]
    delbond = graph_edits.split(';')[2]
    newbond = graph_edits.split(';')[3]

    if len(removeh) > 0:
        for a in removeh.split(','):
            atom_index = int(a)-1 
            if mw.GetAtomWithIdx(atom_index).GetNumExplicitHs() > 0:
                mw.GetAtomWithIdx(atom_index).SetNumExplicitHs(mw.GetAtomWithIdx(atom_index).GetNumExplicitHs() - 1)
                mw.GetAtomWithIdx(atom_index).UpdatePropertyCache(strict=False)

    if len(addh) > 0:
        for a in addh.split(','):
            atom_index = int(a)-1 
            if mw.GetAtomWithIdx(atom_index).GetNumExplicitHs() > 0:
                mw.GetAtomWithIdx(atom_index).SetNumExplicitHs(mw.GetAtomWithIdx(atom_index).GetNumExplicitHs() + 1)
                mw.GetAtomWithIdx(atom_index).UpdatePropertyCache(strict=False)
            
    if len(delbond) > 0:
        for s in delbond.split(','):
            a1, a2, change = s.split('-')
            atom1_idx = int(a1)-1 
            atom2_idx = int(a2)-1
            mw.RemoveBond(atom1_idx,atom2_idx)
            
    if len(newbond) > 0:
        for s in newbond.split(','):
                a1, a2, change = s.split('-')

                atom1_idx = int(a1)-1 
                atom2_idx = int(a2)-1 
                change = float(change)
                mw.RemoveBond(atom1_idx,atom2_idx)
                atom1 = mw.GetAtomWithIdx(atom1_idx)
                atom2 = mw.GetAtomWithIdx(atom2_idx)
                if change == 1.0:
                        mw.AddBond(atom1_idx,atom2_idx,Chem.BondType.SINGLE)
                if change == 2.0:
                        mw.AddBond(atom1_idx,atom2_idx,Chem.BondType.DOUBLE)
                if change == 3.0:
                        mw.AddBond(atom1_idx,atom2_idx,Chem.BondType.TRIPLE)

    mw.UpdatePropertyCache(strict=False)
    for atom in mw.GetAtoms():
            for nbr in atom.GetNeighbors():
                    if  nbr.GetExplicitValence()>pt.GetDefaultValence(nbr.GetAtomicNum()) and \
                    mw.GetBondBetweenAtoms(nbr.GetIdx(),atom.GetIdx()).GetBondType() == Chem.BondType.SINGLE:
                            mw.RemoveBond(nbr.GetIdx(),atom.GetIdx())
                            mw.AddBond(nbr.GetIdx(),atom.GetIdx(),Chem.BondType.DATIVE)
            
            
    if santitize:
        Chem.SanitizeMol(mw)

    return mw


def get_atom_number_symbol_map(mol):
    mapping= dict()
    for atom in mol.GetAtoms():
            mapping[atom.GetAtomMapNum()-1]= atom.GetSymbol()
    return mapping

def plot_M(M,i, name):
    fig, ax1 = plt.subplots(ncols=1, figsize=(10,8))
    M = M.detach().numpy()
    heatmap1 = ax1.imshow(M, cmap='coolwarm')
    cbar = fig.colorbar(heatmap1, ax=ax1)#, shrink=0.85
    ax1.xaxis.tick_top()
    ax1.set_ylabel('Reactant atom index', fontsize=14)
    ax1.set_xlabel('Products atom index', fontsize=14)
    xticks = yticks  =  np.arange(0,len(M))
    ax1.set_xticks(xticks)
    ax1.set_yticks(yticks)
    ax1.set_title(name, fontsize=14)
    plt.savefig(f'plots/{i}_{name}.png',dpi=300, bbox_inches='tight' )
    #plt.show()

def plot_M_with_atom_label(df,i, M , name):
    precursures_mol , product_mol, graph_edits = get_reaction(df, i)
    mw = product_by_editing_subsrate(precursures_mol, graph_edits)   
    mapping_precursure= get_atom_number_symbol_map(precursures_mol)
    mapping_product= get_atom_number_symbol_map(mw)
    fig, ax1 = plt.subplots(ncols=1, figsize=(8,8))
    M = M.detach().numpy()
    heatmap1 = ax1.imshow(M, cmap='coolwarm')
    cbar = fig.colorbar(heatmap1, ax=ax1, shrink=0.8)
    ax1.xaxis.tick_top()
    ax1.set_ylabel('Reactant atom index')
    ax1.set_xlabel('Products atom index')
    xticks = yticks  =  np.arange(0,len(M))
    x_label_names = [mapping_precursure.get(val) for val in xticks]
    y_label_names = [mapping_product.get(val) for val in yticks]
    ax1.set_xticks(xticks)
    ax1.set_yticks(yticks)
    ax1.set_xticklabels(x_label_names, fontsize=10)
    ax1.set_title(name)
    ax1.set_yticklabels(y_label_names, fontsize=10)
    plt.savefig(f'plots/{name}_atom_label_{i}.png',dpi=300, bbox_inches='tight' )
    plt.show()

def move_diagonal_value(matrix):
    matrix = matrix.detach().numpy()
    
    new_matrix = np.copy(matrix)
    n = new_matrix.shape[0]  # Number of rows

    for i in range(n):
        diagonal_value = new_matrix[i, i]  # Get the diagonal value
        rand_col = np.random.choice(np.delete(np.arange(n), i))  # Random column index excluding diagonal
        new_matrix[i, i] = new_matrix[i, rand_col]
        new_matrix[i, rand_col] = diagonal_value

    return new_matrix

def plot_moved_values_matrix(M, i, name, symnonsym):
    moved_values_M = move_diagonal_value(M)
    
    fig, ax1 = plt.subplots(ncols=1, figsize=(10, 8))
    heatmap1 = ax1.imshow(moved_values_M, cmap='coolwarm')
    cbar = fig.colorbar(heatmap1, ax=ax1)
    ax1.xaxis.tick_top()
    ax1.set_ylabel('Reactant atom index', fontsize=14)
    ax1.set_xlabel('Products atom index', fontsize=14)
    xticks = yticks = np.arange(0, len(moved_values_M))
    ax1.set_xticks(xticks)
    ax1.set_yticks(yticks)
    ax1.set_title(name, fontsize=14)
    plt.savefig(f'plots/{i}_random_{symnonsym}_{name}_moved.png', dpi=300, bbox_inches='tight')

def plot_bothM(M0 , MT, i):

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
    M0  = M0 .detach().numpy()
    MT = MT.detach().numpy()
    heatmap1 = ax1.imshow(M0, cmap='coolwarm')
    heatmap2 = ax2.imshow(MT, cmap='coolwarm')
    cbar = fig.colorbar(heatmap1, ax=[ax1,ax2], shrink=0.5)
    ax1.set_title('M0')
    ax1.set_ylabel('Reactant atom index')
    ax1.set_xlabel('Products atom index')
    ax2.set_title('MT')
    ax2.set_ylabel('Reactant atom index')
    ax2.set_xlabel('Products atom index')
    ax1.xaxis.tick_top()
    ax2.xaxis.tick_top()

    xticks = yticks  =  np.arange(0,len(M0),1)
    ax1.set_xticks(xticks)
    ax1.set_yticks(yticks)
    ax1.set_xticklabels(xticks)
    ax1.set_yticklabels(yticks)
    ax2.set_xticks(xticks)
    ax2.set_yticks(yticks)
    ax2.set_xticklabels(xticks)
    ax2.set_yticklabels(yticks)
    plt.savefig(f'plots_test/{i}_both.png',dpi=300, bbox_inches='tight' )
    plt.show()

def draw_mol_with_index(smiles):
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        atom.SetProp('molAtomMapNumber', str(atom.GetIdx()))
    img = Draw.MolToImage(mol, size=(300, 300), kekulize=True)
    img.save('plots/mol.png')

   
def draw_whole_reaction(df,i, santitize=False):
    from rdkit.Chem.Draw import IPythonConsole
    random.seed(42)

    IPythonConsole.drawOptions.addAtomIndices = True

    precursures_mol , product_mol, graph_edits = get_reaction(df, i)
    mw = product_by_editing_subsrate(df, i, santitize=False)
    for atom in precursures_mol.GetAtoms():
        atom.SetProp('molAtomMapNumber', str(atom.GetAtomMapNum()-1))

    for atom in mw.GetAtoms():
        atom.SetProp('molAtomMapNumber', str(atom.GetAtomMapNum()-1))

    new_atom_order = list(range(mw.GetNumAtoms()))
    
    #random.shuffle(new_atom_order)
    
    generated_product = Chem.RenumberAtoms(mw, newOrder=new_atom_order)

    img = Draw.MolsToGridImage([precursures_mol,generated_product], molsPerRow=2, returnPNG=False, subImgSize=(500,500))
    img.save(f'plots/{i}_whole_reaction.png')    
     
    #return graph_edits

def draw_mol_without_map(df,i, santitize=False):
    precursures_mol , product_mol, graph_edits = get_reaction(df, i)
    mw = product_by_editing_subsrate(df, i, santitize=False)
    [atom.SetAtomMapNum(0) for atom in precursures_mol.GetAtoms()]
    [atom.SetAtomMapNum(0) for atom in mw.GetAtoms()]
    img = Draw.MolsToGridImage([precursures_mol,mw], molsPerRow=2, returnPNG=False, subImgSize=(500,500))
    img.save(f'plots/{i}_mol_without_mapping.png')

def remapp_mw(df, i, num_wl_iterations, santitize=False):
    precursures_mol , product_mol, graph_edits = get_reaction(df, i)
    mw = product_by_editing_subsrate(df, i, santitize=False)
    n = mw.GetNumAtoms()
    random_key = list(range(n))
    random.shuffle(random_key)

    for atom in mw.GetAtoms():
        atom.SetProp('molAtomMapNumber', str(random_key))

def plot_mol_with_equivalent_atom(df, i, num_wl_iterations, santitize=False):
    precursures_mol , product_mol, graph_edits = get_reaction(df, i)
    mw = product_by_editing_subsrate(df, i, santitize=False)
    atom_equvalent_sets = get_equivalent_atoms(precursures_mol, num_wl_iterations)
    smallest_atoms = [min(set_) for set_ in atom_equvalent_sets]
    for atom in precursures_mol.GetAtoms():
        atom_idx = atom.GetAtomMapNum()-1
        for idx, set_ in enumerate(atom_equvalent_sets):
            if atom_idx in set_:
                #atom.SetAtomMapNum(smallest_atoms[idx])
                atom.SetProp('molAtomMapNumber', str(smallest_atoms[idx]))

    for atom in mw.GetAtoms():
        atom_idx = atom.GetAtomMapNum()-1
        for idx, set_ in enumerate(atom_equvalent_sets):
            if atom_idx in set_:
                #atom.SetAtomMapNum(smallest_atoms[idx])
                atom.SetProp('molAtomMapNumber', str(smallest_atoms[idx]))
   # am = sorted([atom.GetAtomMapNum() for atom in precursures_mol.GetAtoms()])
    img = Draw.MolsToGridImage([precursures_mol,mw, product_mol], molsPerRow=3, returnPNG=False, subImgSize=(500,500))
    img.save(f'plots/{i}_equvalentatoms.png') 
    #return am

def plot_loss(path, name):
    
   with open(f"{path}/losses_train.txt", "rb") as fp:   # Unpickling
      loss_t = pickle.load(fp)
   with open(f"{path}/losses_valid.txt", "rb") as fp:   # Unpickling
      loss_v = pickle.load(fp)

   fig, axs = plt.subplots(1, 1, figsize=(6, 6))

   x = range(len(loss_t))
   axs.plot(x, loss_t, label='Train Loss')
   axs.plot(x, loss_v, label='Validation Loss')
   axs.set_xlabel('Epoch')
   axs.set_ylabel('Loss')
   axs.set_title(name)
   axs.legend()
   plt.tight_layout()
   plt.savefig(f'experiment1/test_loss_{name}.png',dpi=300, bbox_inches='tight' )
   plt.show()


def plot_loss_acc(path):
    
   with open(f"{path}/losses_train.txt", "rb") as fp:   # Unpickling
      loss_t = pickle.load(fp)
   with open(f"{path}/acces_train.txt", "rb") as fp:   # Unpickling
      acc_t = pickle.load(fp)

   with open(f"{path}/losses_valid.txt", "rb") as fp:   # Unpickling
      loss_v = pickle.load(fp)
   with open(f"{path}/acces_valid.txt", "rb") as fp:   # Unpickling
      acc_v = pickle.load(fp)

   fig, axs = plt.subplots(1, 2, figsize=(12, 6))

   x = range(len(acc_t))
   axs[0].plot(x, loss_t, label='Train Loss')
   axs[0].plot(x, loss_v, label='Validation Loss')
   axs[0].set_xlabel('Epoch')
   axs[0].set_ylabel('Loss')
   axs[0].set_title('Loss')
   axs[0].legend()

   axs[1].plot(x, acc_t, label='Train Accuracy')
   axs[1].plot(x, acc_v, label='Validation Accuracy')
   axs[1].set_xlabel('Epoch')
   axs[1].set_ylabel('Accuracy')
   axs[1].set_title('Accuracy')
   axs[1].legend()
   plt.tight_layout()
   plt.show()

def plot_test_acc(path):

   with open(f"{path}/all_test_acc.txt", "rb") as fp:   # Unpickling
      acc = pickle.load(fp)

   fig, axs = plt.subplots(1,1, figsize=(15, 5))

   x = range(len(acc))
   axs.plot(x, acc, label='Acc')
   axs.set_xlabel('Batch')
   axs.set_ylabel('Acc')
   axs.set_title('Accuracy')
   axs.legend()
   plt.tight_layout()
   plt.show()

