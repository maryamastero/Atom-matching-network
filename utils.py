import rdkit.Chem as Chem
pt = Chem.GetPeriodicTable()
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')    
import random


def get_reaction_info(df, i):
    """
    Returns a reaction SMILES string and reaction edits from dataset.
    Four types of reaction edits:
            Atoms lost Hydrogens
            Atoms obtained Hydrogens
            Deleted bonds
            Added bonds
    Args:
        df: dataset contains that each row has info about reactant, products and reaction edits.
    Returns:
        A tuple containing reaction and reaction edits.
    """
    reaction_smiles = df.iloc[i,0]
    graph_edits = df.iloc[i,1]

    return reaction_smiles, graph_edits


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



def product_by_editing_subsrate(sample_reaction,graph_edits, santitize=False):
    """
    Creates atom mapped product molecule from reactant and reaction edits.
    Args:
        sample_reaction : Reaction SMILES string representing the reactant.
        graph_edits: String containing reaction edits (remove H, add H, delete bond, create bond).

    Returns:
        RDKit molecule object representing the product molecule.
    """
    r, _ = get_reaction_mols(sample_reaction)

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
    mw = mw.GetMol()
    return mw

def get_mapping_number(mol):
    """
    Gets the mapping numbers of atoms in the molecule.
    Args:
        mol: RDKit molecule object.
    Returns:
        List of mapping numbers of atoms in the molecule (0-based indices).
    """
    mapping = []
    for atom in mol.GetAtoms():
            mapping.append(atom.GetAtomMapNum()-1)
    return mapping


def get_random_order_product(reaction_smiles, edits):
    """
    Generates a product molecule that randomly reordering atoms.

    Parameters:
    reaction_smiles: SMILES representation of the reaction substrate molecule.
    graph_edits: List of reaction edits to apply to the substrate.

    Returns:
    Chem.Mol: A randomly reordered atoms product molecule.

    """
    mol = product_by_editing_subsrate(reaction_smiles, edits)
    new_atom_order = list(range(mol.GetNumAtoms()))
    random.seed(42)
    random.shuffle(new_atom_order)
    generated_product = Chem.RenumberAtoms(mol, newOrder=new_atom_order)
    
    return generated_product


