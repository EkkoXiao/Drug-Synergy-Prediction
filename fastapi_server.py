import argparse
import warnings
import os

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

from collections import defaultdict
from itertools import combinations
import json
import re
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import requests
import torch
from model.moltc import MolTC
import pandas as pd
import itertools
import statistics
import uvicorn
# import functions
app = FastAPI()

with open("backend/database/cancer_targets.json", "r", encoding="utf-8") as f:
    CANCER_TARGETS = json.load(f)

with open("backend/database/cancer_cell_line.json", "r", encoding="utf-8") as f:
    CANCER_CELL_LINES = json.load(f)

df = pd.read_csv("backend/database/tested_drugs.csv")
TESTED_DRUGS = {}

for _, row in df.iterrows():
    drug = row['drug']
    if pd.notna(row['drug_row_group_cat']):
        label = int(row['drug_row_group_cat'])
    elif pd.notna(row['drug_col_group_cat']):
        label = int(row['drug_col_group_cat'])
    else:
        label = 2  # 默认 Experimental / Investigational
    TESTED_DRUGS[drug] = label

# MODEL = "deepseek-r1:70b"
MODEL = "qwen2.5:0.5b"

def get_args():
    parser = argparse.ArgumentParser()
    parser = MolTC.add_model_specific_args(parser)
    parser.add_argument('--prompt', type=str, default='[START_SMILES]{}[END_SMILES]')
    parser.add_argument('--cancer', type=bool, default=True)
    parser.add_argument('--NAS', type=bool, default=False)
    parser.add_argument('--cell', type=bool, default=False)
    parser.add_argument('--string', type=bool, default=False)
    parser.add_argument('--category', type=int, default=203)
    parser.add_argument('--device', type=int, default=6)
    parser.add_argument('--llm_device', type=int, default=0)
    args = parser.parse_args()
    args.tune_gnn = True
    args.llm_tune = "lora"
    return args

args = get_args()
device = f"cuda:{args.device}"

# 暂时不启用 DDI 模型
# desc_model = MolTC(args)
# ckpt = torch.load("./all_checkpoints/ft_drugbankCancer_galactica_category_all/last.ckpt", map_location=device)
# desc_model.load_state_dict(ckpt['state_dict'], strict=False)

# args.category = -1

# severity_model = MolTC(args)
# ckpt = torch.load("./all_checkpoints/ft_drugbankCancer_galactica_severity_all/last.ckpt", map_location=device)
# severity_model.load_state_dict(ckpt['state_dict'], strict=False)

# device = f"cuda:{args.device}"

# desc_model.to(device)
# severity_model.to(device)

desc_model = None
severity_model = None

dict_path = "./cancer/"
category_df = pd.read_csv(dict_path + 'category_dict.csv')
category_dict = pd.Series(category_df['Item'].values, index=category_df['Category']).to_dict()

category_chinese_df = pd.read_csv(dict_path + 'category_chinese.csv')
category_chinese = pd.Series(category_chinese_df['Item'].values, index=category_chinese_df['Category']).to_dict()

cancer_drug = pd.read_csv(dict_path + "database.csv")
drug_id_dict = pd.Series(cancer_drug['DrugBank ID'].values, index=cancer_drug['Common name'].str.lower()).to_dict()
drug_dict = dict(zip(cancer_drug['DrugBank ID'], cancer_drug['SMILES']))
drug_dict_property = dict(zip(cancer_drug['DrugBank ID'], cancer_drug['description']))

target_df = pd.read_excel(dict_path + "targetDrugbank.xlsx")
target_dict = target_df.groupby(target_df.columns[1])[target_df.columns[2]].agg(list).to_dict()

drug_target_df = pd.read_csv("backend/database/drug_target_relations_with_atc.csv")

class Drug(BaseModel):
    name: str
    property: str
    target: str
    smiles: str

class TextRequest(BaseModel):
    messages: list

class DrugInfoRequest(BaseModel):
    drug: Drug

class DrugInteractRequest(BaseModel):
    drug1: Drug
    drug2: Drug

class Target(BaseModel):
    name: str
    uniprotId: str
    score: float

class ComboRequestDeprecated(BaseModel):
    targets: list[Target]
    count: int = 2

class ComboRequest(BaseModel):
    cell_line: str
    count: int = Field(default=2, ge=2, le=4)
    label: int = Field(default=1, ge=0, le=4) # label 为组合中药物是 Approved 的数量，不超过 count    
    topk: int = Field(default=10, ge=1, le=1000) # 返回 topk 个组合

@app.get("/info")
def get_drug_info(request: DrugInfoRequest):
    if not drug_id_dict.get(request.drug.name.lower()) and (not request.drug.property or not request.drug.target):
        raise HTTPException(status_code=404, detail="未查询到该药物，请确定输入为标准英文药物名称！")
    drug_name = request.drug.name
    drug_property = request.drug.property
    drug_target = request.drug.target
    drug_smiles = request.drug.smiles
    if request.drug.property == "":
        drug_property = "" if drug_dict_property.get(drug_id_dict.get(request.drug.name.lower()), "").strip().endswith("Overview") else drug_dict_property.get(drug_id_dict.get(request.drug.name.lower()), "")
    if request.drug.target == "":
        drug_target = ",".join(target_dict.get(drug_id_dict.get(request.drug.name.lower()), []))
    if request.drug.smiles == "":
        drug_smiles = drug_dict.get(drug_id_dict.get(request.drug.name.lower(),"NAN"), "Not Available")
    else:
        drug_smiles = request.drug.smiles
    if pd.isna(drug_smiles) or drug_smiles == "Not Available":
        drug_smiles = ""
    return {"name": drug_name, "property": drug_property, "target": drug_target, "smiles": drug_smiles}


@app.get("/interaction")
def drug_interaction(request: DrugInteractRequest):
    prompt = f"</s> #Drug1 is {request.drug1.name}."
    if request.drug1.property != "": 
        prompt += f"[START_PROPERTY]{request.drug1.property[:128]}[END_PROPERTY]"
    if request.drug1.target != "": 
        prompt += f"[START_TARGET]{request.drug1.target[:128]}[END_TARGET]"
    if request.drug1.smiles != "": 
        prompt += f"[START_SMILES]{request.drug1.smiles[:128]}[END_SMILES]"
    prompt += " </s>"
    prompt += f"</s> #Drug2 is {request.drug2.name}."
    if request.drug2.property != "": 
        prompt += f"[START_PROPERTY]{request.drug2.property[:128]}[END_PROPERTY]"
    if request.drug2.target != "": 
        prompt += f"[START_TARGET]{request.drug2.target[:128]}[END_TARGET]"
    if request.drug2.smiles != "": 
        prompt += f"[START_SMILES]{request.drug2.smiles[:128]}[END_SMILES]"
    prompt += " </s>"
    desc_prompt = prompt + "What are the side effects of these two drugs?"

    valid = [request.drug1.smiles != "", request.drug2.smiles != ""]

    output = desc_model.blip2opt.cancer_qa({'prompts': [desc_prompt]}, valid=valid, device=device, num_beams=1, output_scores=True)
    cat_token_dict = desc_model.blip2opt.cat_token_dict

    probabilities = torch.nn.functional.softmax(output["scores"][0][0], dim=-1)
    probabilities, indices = torch.topk(probabilities[50000:], 10)
    indices += 50000

    interaction_dict = defaultdict(lambda: [0. , []])

    for idx, prob in zip(indices, probabilities):
        category = category_chinese[cat_token_dict[idx.item()]].strip()
        interaction_dict[category][0] += prob.item()
        interaction_dict[category][1].append(cat_token_dict[idx.item()])

    interaction_dict = dict(interaction_dict)

    interaction_dict_drugname = {}
    for key, value in interaction_dict.items():
        new_key = key.replace("#Drug1", f"[{request.drug1.name}]").replace("#Drug2",f"[{request.drug2.name}]")
        interaction_dict_drugname[new_key] = value

    return {"interactions": interaction_dict_drugname}

@app.post("/generate")
def generate_text(request: TextRequest):
    try:
        response = requests.post(
            "http://localhost:11600/api/chat",
            json={
                "model": MODEL,
                "messages": request.messages,
                "stream": False,
            }
        )
        return {"generated_text": response.json()['message']['content']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/stream")
def generate_text_stream(request: TextRequest):
    try:
        response = requests.post(
            "http://localhost:11600/api/chat",
            json={
                "model": MODEL,
                "messages": request.messages,
                "stream": True,  # 启用流式响应
            },
            stream=True  # requests 也要开启 stream 模式
        )

        def event_stream():
            try:
                for chunk in response.iter_lines():
                    if chunk:
                        yield chunk.decode('utf-8') + "\n"
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        return StreamingResponse(event_stream(), media_type="text/plain")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/generate_qwen")
def generate_text_qwen(request: TextRequest):
    try:
        response = requests.post(
            "http://localhost:11500/api/chat",
            json={
                "model": MODEL,
                "messages": request.messages,
                "stream": False,
            }
        )
        return {"generated_text": response.json()['message']['content']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/stream_qwen")
def generate_text_stream_qwen(request: TextRequest):
    try:
        response = requests.post(
            "http://localhost:11500/api/chat",
            json={
                "model": MODEL,
                "messages": request.messages,
                "stream": True,  # 启用流式响应
            },
            stream=True  # requests 也要开启 stream 模式
        )

        def event_stream():
            try:
                for chunk in response.iter_lines():
                    if chunk:
                        yield chunk.decode('utf-8') + "\n"
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        return StreamingResponse(event_stream(), media_type="text/plain")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cancer_targets")
def get_cancer_targets(cancer_type: str):
    """返回指定癌症的靶点清单"""
    targets = CANCER_TARGETS.get(cancer_type)
    targets = targets[:50]
    if targets is None:
        raise HTTPException(status_code=404, detail="未知癌症类型")
    return {"targets": targets}

@app.get("/cancer_cell_line")
def get_cancer_cell_line(cancer_type: str):
    """返回指定癌症的细胞系清单"""
    cell_lines = CANCER_CELL_LINES.get(cancer_type)
    if cell_lines is None:
        raise HTTPException(status_code=404, detail="未知癌症类型")
    return {"cell_lines": cell_lines}

@app.post("/recommend_combo_deprecated")
def recommend_combo_deprecated(req: ComboRequestDeprecated):
    targets = req.targets
    k = max(2, min(req.count, 4))
    if len(targets) == 0:
        raise HTTPException(status_code=400, detail="至少需要一个靶点")

    # --- ① 获取候选药物 ---
    uniprots = [t.uniprotId for t in targets]
    df = drug_target_df[
        (drug_target_df["target_uniprot"].isin(uniprots))
        & (drug_target_df["atc_codes"].str.contains(r"^L01|^L02", na=False, regex=True))
    ]

    unique_drugs = df["drug_name"].drop_duplicates().tolist()
    if len(unique_drugs) < 2:
        return {"combos": []}

    # --- ② 计算单药得分 ---
    def _single_drug_score(name: str):
        rows = df[df["drug_name"] == name]
        covered = {r["target_uniprot"] for _, r in rows.iterrows()}
        score = sum(t.score for t in targets if t.uniprotId in covered)     # <── 用属性
        tgt_names = [t.name for t in targets if t.uniprotId in covered]     # <── 用属性
        return round(score, 3), tgt_names

    drug_meta = {
        d: {"score": _single_drug_score(d)[0], "targets": _single_drug_score(d)[1]}
        for d in unique_drugs
    }

    unique_drugs = unique_drugs[:10]

    # 预计算所有药物对的交互风险
    pair_risks = {}
    for d1, d2 in itertools.combinations(unique_drugs, 2):
        req_local = DrugInteractRequest(
            drug1=Drug(name=d1, property="", target="", smiles=""),
            drug2=Drug(name=d2, property="", target="", smiles=""),
        )
        interactions = drug_interaction(req_local)["interactions"]
        ddi_risk = sum(v[0] for v in interactions.values())
        pair_risks[(d1, d2)] = ddi_risk

    combo_candidates = []

    # 1. 枚举所有 k 个药物的组合并计算组合得分
    for combo_drugs in itertools.combinations(unique_drugs, k):
        # 计算该组合中所有药物对的平均分数
        pair_scores = []
        for d1, d2 in itertools.combinations(combo_drugs, 2):
            # 获取预计算的交互风险
            ddi_risk = pair_risks.get((d1, d2)) or pair_risks.get((d2, d1))
            pair_score = (drug_meta[d1]["score"] + drug_meta[d2]["score"]) * (1 - ddi_risk) / 2
            pair_scores.append(pair_score)
        
        # 组合分数为所有两两分数的平均值
        combo_score = round(sum(pair_scores) / len(pair_scores), 3)
        
        combo_candidates.append({
            "drugs": list(combo_drugs),
            "score": combo_score,
        })

    # 2. 仅保留分数最高的前 10 个组合
    top_combos = sorted(
        combo_candidates, key=lambda x: x["score"], reverse=True
    )[:10]

    # 3. 为前 10 个组合生成一句话解释
    for combo in top_combos:
        drugs = combo["drugs"]
        if len(drugs) == 2:
            d1, d2 = drugs[0], drugs[1]
            explanation = generate_text_qwen(
                TextRequest(
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                f"根据下面的信息，使用一句话解释药物组合{d1}和{d2}是如何抗癌的。"
                                f"药物1的名称是{d1}，作用的靶点是{drug_meta[d1]['targets']}，"
                                f"药物2的名称是{d2}，作用的靶点是{drug_meta[d2]['targets']}。"
                            ),
                        }
                    ]
                )
            )['generated_text']
        else:
            # 对于多个药物的组合，生成通用解释
            drug_names = "、".join(drugs)
            target_info = "，".join([f"{d}作用的靶点是{drug_meta[d]['targets']}" for d in drugs])
            explanation = generate_text_qwen(
                TextRequest(
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                f"根据下面的信息，使用一句话解释药物组合{drug_names}是如何协同抗癌的。"
                                f"{target_info}。"
                            ),
                        }
                    ]
                )
            )['generated_text']
        combo["explanation"] = explanation

    return {"combos": top_combos}


@app.post("/recommend_combo")
def recommend_combo(req: ComboRequest):
    # 根据 cell_line 构造文件路径
    csv_file = f"backend/database/result_{req.cell_line}.csv"
    if not os.path.exists(csv_file):
        raise HTTPException(status_code=404, detail=f"未找到细胞系 {req.cell_line} 相关数据")
    df = pd.read_csv(csv_file)  

    all_drugs = list(TESTED_DRUGS.keys())
    approved= [d for d in all_drugs if TESTED_DRUGS[d] == 1]
    investigational = [d for d in all_drugs if TESTED_DRUGS[d] != 1]

    combos = []

    # 遍历所有 approved 药物的组合
    for approved_subset in combinations(approved, req.label):
        remaining_needed = req.count - req.label
        # 遍历 investigational 药物组合
        for investigational_subset in combinations(investigational, remaining_needed):
            combo = list(approved_subset) + list(investigational_subset)
            combos.append(combo)

    score_lookup = {}
    for _, row in df.iterrows():
        pair1 = (row['drug_row'], row['drug_col'])
        pair2 = (row['drug_col'], row['drug_row'])
        score_lookup[pair1] = row['score']
        score_lookup[pair2] = row['score']

    combo_scores = []

    for combo in combos:
        pair_scores = []
        # 遍历所有两两组合
        for drug1, drug2 in combinations(combo, 2):
            score = score_lookup.get((drug1, drug2))
            if score is not None:
                pair_scores.append(score)
            else:
                # 如果没有找到对应分数，可以选择跳过或报错，这里跳过
                continue
        # 计算平均分
        avg_score = sum(pair_scores) / len(pair_scores) if pair_scores else None
        combo_scores.append({'combo': combo, 'score': avg_score})

    # 选择分数最高的前 N 个组合
    top_combos = sorted(combo_scores, key=lambda x: x['score'] if x['score'] is not None else float('-inf'), reverse=True)[:req.topk]

    return {"combos": top_combos}


if __name__ == "__main__":
    uvicorn.run("fastapi_server:app", host="0.0.0.0", port=8000, reload=True)

