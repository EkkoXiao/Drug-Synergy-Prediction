import pandas as pd
import numpy as np

dict_path = "../dicts/"
data_path = "../data/ddi_data/"

all_drug = pd.read_csv(dict_path + "drugdict.csv")

dict_0_2 = {row[0]: row[2] for index, row in all_drug.iterrows()}
dict_1_3 = {row[1]: row[3] for index, row in all_drug.iterrows()}

all_drug = pd.read_excel(dict_path + "drugdict.xlsx", sheet_name='drugbank')

all_drug = np.array(all_drug)
drug_dict_new = {row[0]: row[4] for row in all_drug}
drug_dict = dict_0_2.copy()  
drug_dict.update(dict_1_3)  

property_df = pd.read_excel("../cancer/propertyCancer.xlsx")

for index, row in property_df.iterrows():
    # 检查 SMILES 列是否为 NaN 或 "Not Available"
    if pd.isna(row['SMILES']) or row['SMILES'] == "Not Available":
        drugbank_id = row['DrugBank ID']
        print(f"Find DrugBank ID: {drugbank_id}")
    
        if drugbank_id in drug_dict:
            # 如果找到了匹配的 DrugBank ID，替换 SMILES
            new_smiles = drug_dict[drugbank_id]
            property_df.at[index, 'SMILES'] = new_smiles
            print(f"Substitute new SMILES: {new_smiles}")
        else:
            print(f"DrugBank ID {drugbank_id} not found in drug_info.xlsx.")

# 保存处理后的数据到新的 Excel 文件
property_df.to_excel("propertyCancer.xlsx", index=False)

print("处理完成，结果已保存到 'propertyCancer.xlsx'.")