#!/bin/bash
#SBATCH --job-name=zs-text
#SBATCH --partition=killable.q
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# --- Environment ---
module purge
module load CUDA/12.1.1 cuDNN/8.9.2.26-CUDA-12.1.1
source ~/miniconda3/etc/profile.d/conda.sh  # adjust if you use anaconda3
conda activate llm-receipts

export HF_HOME=/homes/josh98/hf_cache
export TOKENIZERS_PARALLELISM=false

# --- Configurable via --export at submit time ---
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
OCR="${OCR:-paddle_ocr_text}"
TAG="${TAG:-llama31_8b_paddle}"
QUANT="${QUANT:-}"   # set to "--quantize" if VRAM is tight

cd /homes/josh98/Fine-Tuning-OCR-Extracted-Receipt
mkdir -p outputs logs

python -m zero_shot.run_text_only \
    --csv datasets/Receipt_Dataset.csv \
    --model "$MODEL" \
    --ocr-column "$OCR" \
    --out "outputs/zero_shot_${TAG}.jsonl" \
    $QUANT
