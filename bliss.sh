export NCCL_P2P_LEVEL=NVL
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
python main.py \
    --root '/DATA/DATANAS1/xlx21/data/ScaffoldHsa/train/' \
    --valid_root '/DATA/DATANAS1/xlx21/data/ScaffoldHsa/valid/' \
    --test_root '/DATA/DATANAS1/xlx21/data/ScaffoldHsa/test/' \
    --devices '1' \
    --filename "ft_SynergyScaffoldHsa" \
    --opt_model '/DATA/DATANAS1/xlx21/modelscope/galactica-1.3b' \
    --mode ft \
    --tune_gnn \
    --prompt '[START_SMILES]{}[END_SMILES]. ' \
    --inference_batch_size 4 \
    --batch_size 4 \
    --max_len 50  \
    --cell True \
    --NAS True \
    --question 'Do the two drugs exhibit synergy effects? What is their hsa synergy score?' \
    --llm_tune lora \
    --max_epochs 50 \
    --caption_eval_epoch 10 \
    --save_every_n_epochs 10
    