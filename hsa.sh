export NCCL_P2P_LEVEL=NVL
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
python main.py \
    --root '/DATA/DATANAS1/xlx21/data/ScaffoldHsa/' \
    --train_root '/DATA/DATANAS1/xlx21/data/ScaffoldHsa/train/' \
    --valid_root '/DATA/DATANAS1/xlx21/data/ScaffoldHsa/valid/' \
    --test_root '/DATA/DATANAS1/xlx21/data/ScaffoldHsa/test/' \
    --devices '0,1,2,3' \
    --mode 'pretrain' \
    --filename "ft_SynergyScaffoldHsa" \
    --opt_model 'modelscope/galactica-1.3b' \
    --tune_gnn \
    --prompt '[START_SMILES]{}[END_SMILES]. ' \
    --inference_batch_size 4 \
    --batch_size 4 \
    --max_len 50  \
    --cell True \
    --NAS True \
    --question 'Do the two drugs exhibit synergy effects? What is their hsa synergy score?' \
    --llm_tune lora \
    --max_epochs 20 \
    --caption_eval_epoch 5 \
    --save_every_n_epochs 5
    