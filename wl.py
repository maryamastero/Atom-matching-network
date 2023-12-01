import rdkit.Chem as Chem
pt = Chem.GetPeriodicTable()
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')    

def wl_atom_similarity(mol, num_wl_iterations):
    """
    Args:
        mol: RDKit molecule object.
        num_wl_iterations: Number of Weisfeiler-Lehman iterations to perform.
    Returns:
        dict: A dictionary atom indices to their updated labels.
    """
    label_dict = dict()
    for atom in mol.GetAtoms():
        label_dict[atom.GetIdx()]= atom.GetSymbol()

    for _ in range(num_wl_iterations):
        label_dict = update_atom_labels(mol, label_dict)

    return label_dict

def update_atom_labels(mol, label_dict):
    """
    Updates atom labels based on the neighbors' labels.
    Args:
        mol: The RDKit molecule object.
        label_dict: A dictionary atom indices to their current labels.
    Returns:
        dict: A dictionary atom indices to their updated labels.
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
    Creates a list containing sets of equivalent atoms based on similarity in neighborhood.
    Args:
        mol: RDKit molecule object.
        num_wl_iterations: Number of Weisfeiler-Lehman iterations to perform.
    Returns:
        A list of sets where each set contains atom indices of equivalent atoms.
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

def get_atom_matcher(mol1, mol2, num_wl_iterations):
    """
    Creates a dictionary containing mached atoms in two molecular graphs.
    Args:
        mol1 , mol2: RDKit molecule object.
        num_wl_iterations: Number of Weisfeiler-Lehman iterations to perform.
    Returns:
        A dictionary of mached atoms in two molecular graphss.
    """    
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
        
    atom_equiv_classes = dict()
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


                        atom_equiv_classes[centralnode_indx1]= centralnode_indx2

    return atom_equiv_classes
        
