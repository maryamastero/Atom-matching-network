from rdkit import Chem
import os
pt = Chem.GetPeriodicTable()
import pandas as pd

train_data = pd.read_csv("../../dataset/USPTO-15k/train.txt", header=None, sep=' ')


def get_reaction(df, i):
    reaction_smiles = df.iloc[i,0]
    graph_edits = df.iloc[i,1]
    precursures_smiles, products_smiles = reaction_smiles.split('>>')
    precursures_mol = Chem.MolFromSmiles(precursures_smiles)
    products_mol = Chem.MolFromSmiles(products_smiles)

    return precursures_mol, products_mol, graph_edits

def edit_mol(mol, graph_edits, santitize):
        mw = Chem.RWMol(mol)

        losth = graph_edits.split(';')[0]
        gainh = graph_edits.split(';')[1]
        delbond = graph_edits.split(';')[2]
        newbond = graph_edits.split(';')[3]

        

        if len(delbond) > 0:
                for s in delbond.split(','):
                        a1, a2, change = s.split('-')
                        atom1 = int(a1)-1 
                        atom2 = int(a2)-1
                        mw.RemoveBond(atom1,atom2)


        if len(newbond) > 0:
                for s in newbond.split(','):
                        a1, a2, change = s.split('-')

                        atom1_idx = int(a1)-1 
                        atom2_idx = int(a2)-1 
                        change = float(change)
                        mw.RemoveBond(atom1_idx,atom2_idx)
                        atom1 = mw.GetAtomWithIdx(atom1_idx)
                        atom2 = mw.GetAtomWithIdx(atom2_idx)
                        #atom1.SetNumExplicitHs(0)
                        #atom2.SetNumExplicitHs(0)
                        if change == 1.0:
                                mw.AddBond(atom1_idx,atom2_idx,Chem.BondType.SINGLE)
                        if change == 2.0:
                                mw.AddBond(atom1_idx,atom2_idx,Chem.BondType.DOUBLE)
                        if change == 3.0:
                                mw.AddBond(atom1_idx,atom2_idx,Chem.BondType.TRIPLE)
        '''           
        for atom in mw.GetAtoms():
                explicit_valence = atom.GetExplicitValence()
                valid_implicit_valence = pt.GetDefaultValence(atom.GetAtomicNum())
                if explicit_valence > valid_implicit_valence:
                        atom.SetNumExplicitHs(0)
        
        '''  
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

def get_problematic_data(test_data, santitize ):
    problematic_i = []
    for i in range(len(test_data)):
        try:
            precursures_mol, products_mol, graph_edits= get_reaction(test_data, i)
            mw = edit_mol(precursures_mol, graph_edits, santitize)
        except: 
            problematic_i.append(i)
            continue
    return problematic_i



def preprocess_data(test_data, name, santitize=True):
    if santitize==True:
        path = 'SanitizeMol_data'
    else:
        path = 'unSanitizeMol_data'

    if not os.path.exists(path):
        os.makedirs(path)
    problematic_i = get_problematic_data(test_data, santitize)
    processed_data =  test_data.drop(problematic_i)    
    processed_data.to_csv(f'{path}/{name}.csv', index=False, header=None)


if __name__ =='__main__':
      
    train_data = pd.read_csv("../../dataset/USPTO-15k/train.txt", header=None, sep=' ')
    test_data = pd.read_csv("../../dataset/USPTO-15k/test.txt", header=None, sep=' ')
    valid_data = pd.read_csv("../../dataset/USPTO-15k/valid.txt", header=None, sep=' ')
    #whole_data = pd.concat([train_data, valid_data])


    preprocess_data(train_data, 'train')
    preprocess_data(test_data, 'test')
    preprocess_data(valid_data, 'valid')
    #preprocess_data(whole_data, 'whole_data')

