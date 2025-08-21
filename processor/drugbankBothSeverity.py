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
import shutil

dict_path = "../both/"
data_path = "../../cancerProj/"
dataset_name = "drugbankBothSeverity/"

if os.path.exists(data_path + dataset_name):
    shutil.rmtree(data_path + dataset_name)

invalid_list = []

cancer_drug = pd.read_excel(dict_path + "propertyDrugbank.xlsx")
drug_dict = dict(zip(cancer_drug['DrugBank ID'], cancer_drug['SMILES']))
drug_dict_property = dict(zip(cancer_drug['DrugBank ID'], cancer_drug['description']))

target_df = pd.read_excel(dict_path + "targetDrugbank.xlsx")
target_dict = target_df.groupby(target_df.columns[1])[target_df.columns[2]].agg(list).to_dict()

empty_graph = Data(
    x=torch.empty(0, 0),  # 没有节点特征
    edge_index=torch.empty(2, 0, dtype=torch.long),  # 没有边
    edge_attr=torch.empty(0, 0)  # 没有边特征
)

test_data = pd.read_csv(data_path + 'drugbankBoth_test.csv')
test_data = np.array(test_data)
i = 0
for j in range(len(test_data)): 
    try:
        graph1 = mol_to_graph_data_obj_simple(drug_dict[test_data[j][0]])
        data1 = {"Valid": True, "Graph": graph1, "Drug": test_data[j][0]}
    except Exception as e:
        invalid_list.append(test_data[j][0])
        print(f"Error occurred while processing {test_data[j][0]}:")
        print(str(e))
        data1 = {"Valid": False, "Graph": empty_graph, "Drug": test_data[j][0]}
    try:
        graph2 = mol_to_graph_data_obj_simple(drug_dict[test_data[j][1]])
        data2 = {"Valid": True, "Graph": graph2, "Drug": test_data[j][1]}
    except Exception as e:
        invalid_list.append(test_data[j][1])
        print(f"Error occurred while processing {test_data[j][1]}:")
        print(str(e))
        data2 = {"Valid": False, "Graph": empty_graph, "Drug": test_data[j][1]}
    text = test_data[j][2]
    if test_data[j][5].lower() not in text.lower() or test_data[j][6].lower() not in text.lower():
        print(f"Skipped text: {text}")
        continue
    os.makedirs(data_path + dataset_name + "test/graph1/" + str(i))
    torch.save(data1, data_path + dataset_name + "test/graph1/" + str(i) + '/graph_data.pt')
    os.makedirs(data_path + dataset_name + "test/graph2/" + str(i))
    torch.save(data2, data_path + dataset_name + "test/graph2/" + str(i) + '/graph_data.pt')
    os.makedirs(data_path + dataset_name + "test/smiles1/" + str(i))
    os.makedirs(data_path + dataset_name + "test/smiles2/" + str(i))
    os.makedirs(data_path + dataset_name + "test/property1/" + str(i))
    os.makedirs(data_path + dataset_name + "test/property2/" + str(i))
    os.makedirs(data_path + dataset_name + "test/description/" + str(i))
    os.makedirs(data_path + dataset_name + "test/text/" + str(i))
    print(i)
    if re.search(re.escape(test_data[j][5]), test_data[j][6], flags=re.IGNORECASE):
        text = re.sub(re.escape(test_data[j][6]), '#Drug2', text, flags=re.IGNORECASE)
        text = re.sub(re.escape(test_data[j][5]), '#Drug1', text, flags=re.IGNORECASE)
    else:
        text = re.sub(re.escape(test_data[j][5]), '#Drug1', text, flags=re.IGNORECASE)
        text = re.sub(re.escape(test_data[j][6]), '#Drug2', text, flags=re.IGNORECASE)
    file = open(data_path + dataset_name + "test/description/" + str(i) + "/text.txt","w")
    file.write(text)
    file.close()
    text = test_data[j][4].capitalize() + '.'
    file = open(data_path + dataset_name + "test/text/" + str(i) + "/text.txt","w")
    file.write(text)
    file.close()
    file = open(data_path + dataset_name + "test/smiles1/" + str(i) + "/text.txt","w")
    if pd.notna(drug_dict[test_data[j][0]]) and drug_dict[test_data[j][0]] != "Not Available":
        file.write(drug_dict[test_data[j][0]])
    else:
        file.write("None")
    file.close()
    file = open(data_path + dataset_name + "test/smiles2/" + str(i) + "/text.txt","w")
    if pd.notna(drug_dict[test_data[j][1]]) and drug_dict[test_data[j][1]] != "Not Available":
        file.write(drug_dict[test_data[j][1]])
    else:
        file.write("None")
    file.close()
    file = open(data_path + dataset_name + "test/property1/" + str(i) + "/text.txt","w")
    property1 = "#Drug1 is " + test_data[j][5] + ". [START_PROPERTY]"
    property1 += "" if drug_dict_property[test_data[j][0]].strip().endswith("Overview") else drug_dict_property[test_data[j][0]].strip()
    property1 += "[END_PROPERTY]"
    target1 = target_dict.get(test_data[j][0], None)
    if target1 is not None:
        target1 = ",".join(target1)
        property1 += "[START_TARGET]" + target1[:128] + "[END_TARGET]"
    file.write(property1)
    file.close()
    file = open(data_path + dataset_name + "test/property2/" + str(i) + "/text.txt","w")
    property2 = "#Drug2 is " + test_data[j][6] + ". [START_PROPERTY]"
    property2 += "" if drug_dict_property[test_data[j][1]].strip().endswith("Overview") else drug_dict_property[test_data[j][1]].strip()
    property2 += "[END_PROPERTY]"
    target2 = target_dict.get(test_data[j][1], None)
    if target2 is not None:
        target2 = ",".join(target2)
        property2 += "[START_TARGET]" + target2[:128] + "[END_TARGET]"
    file.write(property2)
    file.close()
    i += 1


train_data = pd.read_csv(data_path + 'drugbankBoth_train.csv')
train_data = np.array(train_data)
i = 0
for j in range(len(train_data)):
    try:
        graph1 = mol_to_graph_data_obj_simple(drug_dict[train_data[j][0]])
        data1 = {"Valid": True, "Graph": graph1, "Drug": train_data[j][0]}
    except Exception as e:
        invalid_list.append(train_data[j][0])
        print(f"Error occurred while processing {train_data[j][0]}:")
        print(str(e))
        data1 = {"Valid": False, "Graph": empty_graph, "Drug": train_data[j][0]}
    try:
        graph2 = mol_to_graph_data_obj_simple(drug_dict[train_data[j][1]])
        data2 = {"Valid": True, "Graph": graph2, "Drug": train_data[j][1]}
    except Exception as e:
        invalid_list.append(train_data[j][1])
        print(f"Error occurred while processing {train_data[j][1]}:")
        print(str(e))
        data2 = {"Valid": False, "Graph": empty_graph, "Drug": train_data[j][1]}
    text = train_data[j][2]
    if train_data[j][5].lower() not in text.lower() or train_data[j][6].lower() not in text.lower():
        print(f"Skipped text: {text}")
        continue
    os.makedirs(data_path + dataset_name + "train/graph1/" + str(i))
    torch.save(data1, data_path + dataset_name + "train/graph1/" + str(i) + '/graph_data.pt')
    os.makedirs(data_path + dataset_name + "train/graph2/" + str(i))
    torch.save(data2, data_path + dataset_name + "train/graph2/" + str(i) + '/graph_data.pt')
    os.makedirs(data_path + dataset_name + "train/smiles1/" + str(i))
    os.makedirs(data_path + dataset_name + "train/smiles2/" + str(i))
    os.makedirs(data_path + dataset_name + "train/property1/" + str(i))
    os.makedirs(data_path + dataset_name + "train/property2/" + str(i))
    os.makedirs(data_path + dataset_name + "train/description/" + str(i))
    os.makedirs(data_path + dataset_name + "train/text/" + str(i))
    print(i)
    if re.search(re.escape(train_data[j][5]), train_data[j][6], flags=re.IGNORECASE):
        text = re.sub(re.escape(train_data[j][6]), '#Drug2', text, flags=re.IGNORECASE)
        text = re.sub(re.escape(train_data[j][5]), '#Drug1', text, flags=re.IGNORECASE)
    else:
        text = re.sub(re.escape(train_data[j][5]), '#Drug1', text, flags=re.IGNORECASE)
        text = re.sub(re.escape(train_data[j][6]), '#Drug2', text, flags=re.IGNORECASE)
    file = open(data_path + dataset_name + "train/description/" + str(i) + "/text.txt","w")
    file.write(text)
    file.close()
    text = train_data[j][4].capitalize() + '.'
    file = open(data_path + dataset_name + "train/text/" + str(i) + "/text.txt","w")
    file.write(text)
    file.close()
    file = open(data_path + dataset_name + "train/smiles1/" + str(i) + "/text.txt","w")
    if pd.notna(drug_dict[train_data[j][0]]) and drug_dict[train_data[j][0]] != "Not Available":
        file.write(drug_dict[train_data[j][0]])
    else:
        file.write("None")
    file.close()
    file = open(data_path + dataset_name + "train/smiles2/" + str(i) + "/text.txt","w")
    if pd.notna(drug_dict[train_data[j][1]]) and drug_dict[train_data[j][1]] != "Not Available":
        file.write(drug_dict[train_data[j][1]])
    else:
        file.write("None")
    file.close()
    file = open(data_path + dataset_name + "train/property1/" + str(i) + "/text.txt","w")
    property1 = "#Drug1 is " + train_data[j][5] + ". [START_PROPERTY]"
    property1 += "" if drug_dict_property[train_data[j][0]].strip().endswith("Overview") else drug_dict_property[train_data[j][0]].strip()
    property1 += "[END_PROPERTY]"
    target1 = target_dict.get(train_data[j][0], None)
    if target1 is not None:
        target1 = ",".join(target1)
        property1 += "[START_TARGET]" + target1[:128] + "[END_TARGET]"
    file.write(property1)
    file.close()
    file = open(data_path + dataset_name + "train/property2/" + str(i) + "/text.txt","w")
    property2 = "#Drug2 is " + train_data[j][6] + ". [START_PROPERTY]"
    property2 += "" if drug_dict_property[train_data[j][1]].strip().endswith("Overview") else drug_dict_property[train_data[j][1]].strip()
    property2 += "[END_PROPERTY]"
    target2 = target_dict.get(train_data[j][1], None)
    if target2 is not None:
        target2 = ",".join(target2)
        property2 += "[START_TARGET]" + target2[:128] + "[END_TARGET]"
    file.write(property2)
    file.close()
    i += 1



train_data = pd.read_csv(data_path + 'drugbankBoth_all.csv')
train_data = np.array(train_data)
i = 0
for j in range(len(train_data)):
    try:
        graph1 = mol_to_graph_data_obj_simple(drug_dict[train_data[j][0]])
        data1 = {"Valid": True, "Graph": graph1, "Drug": train_data[j][0]}
    except Exception as e:
        invalid_list.append(train_data[j][0])
        print(f"Error occurred while processing {train_data[j][0]}:")
        print(str(e))
        data1 = {"Valid": False, "Graph": empty_graph, "Drug": train_data[j][0]}
    try:
        graph2 = mol_to_graph_data_obj_simple(drug_dict[train_data[j][1]])
        data2 = {"Valid": True, "Graph": graph2, "Drug": train_data[j][1]}
    except Exception as e:
        invalid_list.append(train_data[j][1])
        print(f"Error occurred while processing {train_data[j][1]}:")
        print(str(e))
        data2 = {"Valid": False, "Graph": empty_graph, "Drug": train_data[j][1]}
    text = train_data[j][2]
    if train_data[j][5].lower() not in text.lower() or train_data[j][6].lower() not in text.lower():
        print(f"Skipped text: {text}")
        continue
    os.makedirs(data_path + dataset_name + "all/graph1/" + str(i))
    torch.save(data1, data_path + dataset_name + "all/graph1/" + str(i) + '/graph_data.pt')
    os.makedirs(data_path + dataset_name + "all/graph2/" + str(i))
    torch.save(data2, data_path + dataset_name + "all/graph2/" + str(i) + '/graph_data.pt')
    os.makedirs(data_path + dataset_name + "all/smiles1/" + str(i))
    os.makedirs(data_path + dataset_name + "all/smiles2/" + str(i))
    os.makedirs(data_path + dataset_name + "all/property1/" + str(i))
    os.makedirs(data_path + dataset_name + "all/property2/" + str(i))
    os.makedirs(data_path + dataset_name + "all/description/" + str(i))
    os.makedirs(data_path + dataset_name + "all/text/" + str(i))
    print(i)
    if re.search(re.escape(train_data[j][5]), train_data[j][6], flags=re.IGNORECASE):
        text = re.sub(re.escape(train_data[j][6]), '#Drug2', text, flags=re.IGNORECASE)
        text = re.sub(re.escape(train_data[j][5]), '#Drug1', text, flags=re.IGNORECASE)
    else:
        text = re.sub(re.escape(train_data[j][5]), '#Drug1', text, flags=re.IGNORECASE)
        text = re.sub(re.escape(train_data[j][6]), '#Drug2', text, flags=re.IGNORECASE)
    file = open(data_path + dataset_name + "all/description/" + str(i) + "/text.txt","w")
    file.write(text)
    file.close()
    text = train_data[j][4].capitalize() + '.'
    file = open(data_path + dataset_name + "all/text/" + str(i) + "/text.txt","w")
    file.write(text)
    file.close()
    file = open(data_path + dataset_name + "all/smiles1/" + str(i) + "/text.txt","w")
    if pd.notna(drug_dict[train_data[j][0]]) and drug_dict[train_data[j][0]] != "Not Available":
        file.write(drug_dict[train_data[j][0]])
    else:
        file.write("None")
    file.close()
    file = open(data_path + dataset_name + "all/smiles2/" + str(i) + "/text.txt","w")
    if pd.notna(drug_dict[train_data[j][1]]) and drug_dict[train_data[j][1]] != "Not Available":
        file.write(drug_dict[train_data[j][1]])
    else:
        file.write("None")
    file.close()
    file = open(data_path + dataset_name + "all/property1/" + str(i) + "/text.txt","w")
    property1 = "#Drug1 is " + train_data[j][5] + ". [START_PROPERTY]"
    property1 += "" if drug_dict_property[train_data[j][0]].strip().endswith("Overview") else drug_dict_property[train_data[j][0]].strip()
    property1 += "[END_PROPERTY]"
    target1 = target_dict.get(train_data[j][0], None)
    if target1 is not None:
        target1 = ",".join(target1)
        property1 += "[START_TARGET]" + target1[:128] + "[END_TARGET]"
    file.write(property1)
    file.close()
    file = open(data_path + dataset_name + "all/property2/" + str(i) + "/text.txt","w")
    property2 = "#Drug2 is " + train_data[j][6] + ". [START_PROPERTY]"
    property2 += "" if drug_dict_property[train_data[j][1]].strip().endswith("Overview") else drug_dict_property[train_data[j][1]].strip()
    property2 += "[END_PROPERTY]"
    target2 = target_dict.get(train_data[j][1], None)
    if target2 is not None:
        target2 = ",".join(target2)
        property2 += "[START_TARGET]" + target2[:128] + "[END_TARGET]"
    file.write(property2)
    file.close()
    i += 1