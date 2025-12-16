import argparse
from collections import Counter
import datetime
import itertools
import json
import pandas as pd
import warnings
import os
from tqdm import tqdm

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

from model.moltc import MolTC
from transformers import AutoTokenizer, AutoModelForCausalLM

import math
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

CUSTOM_SEQ_RE = re.compile(r"(\[START_(DNA|SMILES|I_SMILES|AMINO)])(.*?)(\[END_\2])")
SPLIT_MARKER = f"|"

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

def _insert_split_marker(m: re.Match):
    start_token, _, sequence, end_token = m.groups()
    sequence = re.sub(r"(.)", fr"{SPLIT_MARKER}\1", sequence, flags=re.DOTALL)
    return f"{start_token}{sequence}{SPLIT_MARKER}{end_token}"


def smiles_handler(text, mol_ph, mode="val"):
    smiles_list = []
    for match in CUSTOM_SEQ_RE.finditer(text):
        smiles = match.group(3)
        smiles_list.append(smiles)
    text = CUSTOM_SEQ_RE.sub(r'\1\3\4%s' % (mol_ph), text)
    text = escape_custom_split_sequence(text)
    return text, smiles_list

def cell_handler(text, cell_ph, mode="val"):
    text = re.sub(r'\[START_CELL\]\[END_CELL\]', f'[START_CELL]{cell_ph}[END_CELL]', text)
    return text

def escape_custom_split_sequence(text):
    return CUSTOM_SEQ_RE.sub(_insert_split_marker, text)


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

empty_graph = Data(
    x=torch.empty(0, 0),
    edge_index=torch.empty(2, 0, dtype=torch.long),
    edge_attr=torch.empty(0, 0)
)

genes = pd.read_csv("./dicts/genes.txt", sep='\t')

def process(drugs, cell):
    try:
        graph1 = mol_to_graph_data_obj_simple(drugs[0]["smiles"])
        data1 = {"Valid": True, "Graph": graph1, "Drug": drugs[0]["name"]}
    except Exception as e:
        print(f"Error occurred while processing {drugs[0]['name']}:")
        print(str(e))
        data1 = {"Valid": False, "Graph": empty_graph, "Drug": drugs[0]["name"]}
    try:
        graph2 = mol_to_graph_data_obj_simple(drugs[1]["smiles"])
        data2 = {"Valid": True, "Graph": graph2, "Drug": drugs[1]["name"]}
    except Exception as e:
        print(f"Error occurred while processing {drugs[1]['name']}:")
        print(str(e))
        data2 = {"Valid": False, "Graph": empty_graph, "Drug": drugs[1]["name"]}
    cell_line = cell
    tensor = torch.tensor(genes[cell_line].astype(float).values)
    smiles1 = drugs[0]["smiles"]
    smiles2 = drugs[1]["smiles"]
    if pd.notna(smiles1) and smiles1 != "Not Available":
        if not data1["Valid"]:
            print("ERROR!")
            print(smiles1)
            print(drugs[0]["name"]) 
    else:
        smiles1 = None
    if pd.notna(smiles2) and smiles2 != "Not Available":
        if not data2["Valid"]:
            print("ERROR!")
            print(smiles2)
            print(drugs[1]["name"]) 
    else:
        smiles2 = None
    property1 = "#Drug1 is " + drugs[0]["name"]
    target1 = drugs[0]["target"]
    if target1 is not None:
        property1 += "[START_TARGET]" + target1[:128] + "[END_TARGET]"
    property2 = "#Drug2 is " + drugs[1]["name"]
    target2 = drugs[1]["target"]
    if target2 is not None:
        property2 += "[START_TARGET]" + target2[:128] + "[END_TARGET]"

    return data1, data2, smiles1, smiles2, property1, property2, tensor

def generate(data1, data2, smiles1, smiles2, property1, property2, genes, model, tokenizer, device):
    smiles_prompt1 = f"[START_SMILES]{smiles1}[END_SMILES]" if smiles1 else ""
    smiles_prompt2 = f"[START_SMILES]{smiles2}[END_SMILES]" if smiles1 else ""

    smiles_prompt1 = property1 + ("" if smiles1 is None else smiles_prompt1)
    smiles_prompt2 = property2 + ("" if smiles2 is None else smiles_prompt2)

    cell_line_prompt = "The cell line of this drug pair is [START_CELL][END_CELL]. "
    smiles_prompt = '</s> '+ cell_line_prompt + '< /s>' + smiles_prompt1 + ' </s>'+' </s>'+smiles_prompt2+' </s> .' + " " + "Do the two drugs exhibit synergy effects? What is their bliss synergy score?"

    collator = Collater([], [])
    graphs1 = {
        'Valid': collator((data1['Valid'],)),
        'Graph': collator((data1['Graph'],)).to(device),
    }
    graphs2 = {
        'Valid': collator((data2['Valid'],)),
        'Graph': collator((data2['Graph'],)).to(device),
    }
    samples = {
        'graphs1': graphs1,
        'graphs2': graphs2,
        'prompts': collator((smiles_prompt,)),
        'genes': collator((genes,)).to(device),
    }

    result = model.blip2opt.cellline_qa(samples, device=device)

    return result

def get_args():
    parser = argparse.ArgumentParser()
    parser = MolTC.add_model_specific_args(parser)
    parser.add_argument('--prompt', type=str, default='[START_SMILES]{}[END_SMILES]')
    parser.add_argument('--cancer', type=bool, default=True)
    parser.add_argument('--NAS', type=bool, default=False)
    parser.add_argument('--cell', type=bool, default=True)
    parser.add_argument('--string', type=bool, default=False)
    parser.add_argument('--category', type=int, default=-1)
    parser.add_argument('--device', type=int, default=2)
    parser.add_argument('--llm_device', type=int, default=0)
    args = parser.parse_args()
    args.tune_gnn = True
    args.llm_tune = "lora"
    return args

def main(args):
    device = f"cuda:{args.device}"
    model = MolTC(args)
    ckpt = torch.load("all_checkpoints/ft_SynergyCancerCombs/epoch=00.ckpt", map_location=device)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model = model.to(device)

    tokenizer = model.blip2opt.opt_tokenizer

    df = pd.read_csv("./cancer_outputs/new_comb/drug_pairs_dedup.csv")

    output_file = "./cancer_outputs/new_comb/results_new.txt"

    with open(output_file, "w") as f:
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            drugs = [
                {"name": row["drug1"], "smiles": row["smiles1"], "target": ""},
                {"name": row["drug2"], "smiles": row["smiles2"], "target": ""}
            ]
            cell = row["cell"]

            # 调用你的处理函数
            try:
                data1, data2, smiles1, smiles2, property1, property2, cell_name = process(drugs, cell)
                result = generate(data1, data2, smiles1, smiles2, property1, property2, cell_name, model, tokenizer, device)
            except Exception as e:
                print(f"第 {idx} 行处理出错: {e}")
                continue

            # 每条数据单独写入一行 JSON
            json_line = json.dumps({
                "drug1": row["drug1"],
                "drug2": row["drug2"],
                "cell": cell,
                "result": result
            }, ensure_ascii=False)
            f.write(json_line + "\n")

    print(f"处理完成，结果已按行写入 {output_file}")


if __name__ == '__main__':
    main(get_args()) 