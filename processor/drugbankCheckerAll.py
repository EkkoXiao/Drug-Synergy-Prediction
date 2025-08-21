import os
import re
import torch
import random
import pickle
import pandas as pd
import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader.dataloader import Collater
from itertools import repeat, chain

# allowable node and edge features
allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)),
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list' : [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs' : [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}

def mol_to_graph_data_obj_simple(smiles):
    # atoms
    mol = Chem.MolFromSmiles(smiles)
    num_atom_features = 2   # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
            'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2   # bond type, bond direction
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())] + [allowable_features[
                                            'possible_bond_dirs'].index(
                bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:   # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data

import pandas as pd
import numpy as np
import os 
import csv

dict_path = "../dicts/"
data_path = "../data/ddi_data/"
dataset_name = "drugbankCheckerAll/"

all_drug = pd.read_csv(dict_path + "drugdict.csv")

dict_0_2 = {row[0]: row[2] for index, row in all_drug.iterrows()}
dict_1_3 = {row[1]: row[3] for index, row in all_drug.iterrows()}

all_drug = pd.read_excel(dict_path + "drugdict.xlsx", sheet_name='drugbank')

all_drug = np.array(all_drug)
drug_dict_new = {row[0]: row[4] for row in all_drug}
drug_dict = dict_0_2.copy()  
drug_dict.update(dict_1_3)  

for key, value in drug_dict.items():
    if key not in drug_dict_new:
        print(key)

# test_data = pd.read_csv(data_path + 'drugbankCheckerAll_test.csv')
# test_data = np.array(test_data)
# for i in range(len(test_data)):
#     print(i)
#     os.makedirs(data_path + dataset_name + "test/graph1/" + str(i))
#     os.makedirs(data_path + dataset_name + "test/graph2/" + str(i))
#     os.makedirs(data_path + dataset_name + "test/smiles1/" + str(i))
#     os.makedirs(data_path + dataset_name + "test/smiles2/" + str(i))
#     os.makedirs(data_path + dataset_name + "test/text/" + str(i))
    
#     data = mol_to_graph_data_obj_simple(drug_dict[test_data[i][0]])
#     torch.save(data, data_path + dataset_name + "test/graph1/" + str(i) + '/graph_data.pt')
#     data = mol_to_graph_data_obj_simple(drug_dict[test_data[i][1]])
#     torch.save(data, data_path + dataset_name + "test/graph2/" + str(i) + '/graph_data.pt')
#     text = test_data[i][2]
#     text = text.replace(test_data[i][5], "#Drug1")
#     text = text.replace(test_data[i][6], "#Drug2")

#     file = open(data_path + dataset_name + "test/text/" + str(i) + "/text.txt","w")
#     file.write(text)
#     file.close()
#     smiles1 = test_data[i][6]
#     smiles2 = test_data[i][7]
#     file = open(data_path + dataset_name + "test/smiles1/" + str(i) + "/text.txt","w")
#     file.write(drug_dict[test_data[i][0]])
#     file.close()
#     file = open(data_path + dataset_name + "test/smiles2/" + str(i) + "/text.txt","w")
#     file.write(drug_dict[test_data[i][1]])
#     file.close()


# train_data = pd.read_csv(data_path + 'drugbankCheckerAll_train.csv')
# train_data = np.array(train_data)
# for i in range(len(train_data)):
#     print(i)
#     os.makedirs(data_path + dataset_name + "train/graph1/" + str(i))
#     os.makedirs(data_path + dataset_name + "train/graph2/" + str(i))
#     os.makedirs(data_path + dataset_name + "train/smiles1/" + str(i))
#     os.makedirs(data_path + dataset_name + "train/smiles2/" + str(i))
#     os.makedirs(data_path + dataset_name + "train/text/" + str(i))
#     data = mol_to_graph_data_obj_simple(drug_dict[train_data[i][0]])
#     torch.save(data, data_path + dataset_name + "train/graph1/" + str(i) + '/graph_data.pt')
#     data = mol_to_graph_data_obj_simple(drug_dict[train_data[i][1]])
#     torch.save(data, data_path + dataset_name + "train/graph2/" + str(i) + '/graph_data.pt')
#     text = train_data[i][2]
#     text = text.replace(train_data[i][5], "#Drug1")
#     text = text.replace(train_data[i][6], "#Drug2")

#     file = open(data_path + dataset_name + "train/text/" + str(i) + "/text.txt","w")
#     file.write(text)
#     file.close()
#     smiles1 = train_data[i][6]
#     smiles2 = train_data[i][7]
#     file = open(data_path + dataset_name + "train/smiles1/" + str(i) + "/text.txt","w")
#     file.write(drug_dict[train_data[i][0]])
#     file.close()
#     file = open(data_path + dataset_name + "train/smiles2/" + str(i) + "/text.txt","w")
#     file.write(drug_dict[train_data[i][1]])
#     file.close()

# valid_data = pd.read_csv(data_path + 'drugbankCheckerAll_validation.csv')
# valid_data = np.array(valid_data)
# for i in range(len(valid_data)):
#     print(i)
#     os.makedirs(data_path + dataset_name + "valid/graph1/" + str(i))
#     os.makedirs(data_path + dataset_name + "valid/graph2/" + str(i))
#     os.makedirs(data_path + dataset_name + "valid/smiles1/" + str(i))
#     os.makedirs(data_path + dataset_name + "valid/smiles2/" + str(i))
#     os.makedirs(data_path + dataset_name + "valid/text/" + str(i))
#     data = mol_to_graph_data_obj_simple(drug_dict[valid_data[i][0]])
#     torch.save(data, data_path + dataset_name + "valid/graph1/" + str(i) + '/graph_data.pt')
#     data = mol_to_graph_data_obj_simple(drug_dict[valid_data[i][1]])
#     torch.save(data, data_path + dataset_name + "valid/graph2/" + str(i) + '/graph_data.pt')
#     text = valid_data[i][2]
#     text = text.replace(valid_data[i][5], "#Drug1")
#     text = text.replace(valid_data[i][6], "#Drug2")

#     file = open(data_path + dataset_name + "valid/text/" + str(i) + "/text.txt","w")
#     file.write(text)
#     file.close()
#     smiles1 = valid_data[i][6]
#     smiles2 = valid_data[i][7]
#     file = open(data_path + dataset_name + "valid/smiles1/" + str(i) + "/text.txt","w")
#     file.write(drug_dict[valid_data[i][0]])
#     file.close()
#     file = open(data_path + dataset_name + "valid/smiles2/" + str(i) + "/text.txt","w")
#     file.write(drug_dict[valid_data[i][1]])
#     file.close()