#!/bin/bash
# ===========================================================================
# setup_beocat.sh — One-shot setup for the receipt extraction project
# on Beocat (K-State HPC).
#
# Usage:
#   1. Copy this script to your Beocat home directory
#   2. chmod +x setup_beocat.sh
#   3. ./setup_beocat.sh
#
# What it does:
#   - Installs Miniconda (if not already installed)
#   - Creates the conda env 'llm-receipts' with all dependencies
#   - Logs into HuggingFace (interactive — you need your token ready)
#   - Downloads both models (Llama 3.1 8B Instruct, Gemma 2 9B IT)
#   - Clones the repo and checks out your branch
#
# BEFORE RUNNING:
#   1. Get a HuggingFace account at https://huggingface.co
#   2. Accept the license agreements for BOTH models:
#      - https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
#      - https://huggingface.co/google/gemma-2-9b-it
#   3. Create an access token at https://huggingface.co/settings/tokens
#      (read access is sufficient)
#   4. Have the GitHub repo URL ready. The default below points to the
#      team repo. Change GIT_REPO and GIT_BRANCH as needed.
#
# NOTES ON THE ENVIRONMENT:
#   - We install PyTorch from the cu124 wheel channel, NOT cu121.
#     Reason: cu121 wheels are stuck at torch 2.5.1, but the latest
#     transformers requires torch >= 2.6. cu124 wheels work fine on
#     Beocat's CUDA 12.1 driver (backward compatible within major
#     version 12.x).
#   - This env is ONLY for LLM inference. Do NOT install PaddleOCR or
#     other OCR dependencies into it — they have conflicting deps.
#     OCR was already run; results are in the CSV.
#
# ===========================================================================

set -euo pipefail

# ---- Configuration (edit these) -------------------------------------------

GIT_REPO="https://github.com/YOUR_ORG/Fine-Tuning-OCR-Extracted-Receipt.git"
GIT_BRANCH="your-branch-name"        # e.g., "MBranch" for Manny
HF_CACHE_DIR="$HOME/hf_cache"
REPO_DIR="$HOME/Fine-Tuning-OCR-Extracted-Receipt"
CONDA_DIR="$HOME/miniconda3"
ENV_NAME="llm-receipts"

# ---- Color helpers --------------------------------------------------------

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ---- Step 1: Miniconda ----------------------------------------------------

if [ -f "$CONDA_DIR/etc/profile.d/conda.sh" ]; then
    info "Miniconda already installed at $CONDA_DIR"
else
    info "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
    rm /tmp/miniconda.sh
    info "Miniconda installed."
fi

# Activate conda for this script
source "$CONDA_DIR/etc/profile.d/conda.sh"

# ---- Step 2: Conda environment --------------------------------------------

if conda env list | grep -q "^${ENV_NAME} "; then
    info "Conda env '$ENV_NAME' already exists. Skipping creation."
    info "  To rebuild from scratch: conda env remove -n $ENV_NAME"
else
    info "Creating conda env '$ENV_NAME' (Python 3.11)..."
    conda create -y -n "$ENV_NAME" python=3.11

    info "Installing packages into '$ENV_NAME'..."
    conda activate "$ENV_NAME"

    # PyTorch from cu124 channel (NOT cu121 — see note above)
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124

    # HuggingFace ecosystem
    pip install transformers accelerate bitsandbytes \
                sentencepiece protobuf huggingface_hub

    # Data / utility
    pip install pandas tqdm pillow ujson

    conda deactivate
    info "Conda env '$ENV_NAME' created and packages installed."
fi

# ---- Step 3: HuggingFace auth ---------------------------------------------

conda activate "$ENV_NAME"

info "Checking HuggingFace authentication..."
if hf auth whoami &>/dev/null; then
    HF_USER=$(hf auth whoami 2>/dev/null | head -1)
    info "Already logged in as: $HF_USER"
else
    warn "Not logged into HuggingFace. Starting interactive login..."
    warn "Paste your access token when prompted."
    warn "  (Get one at https://huggingface.co/settings/tokens)"
    echo ""
    hf auth login
    echo ""
    if hf auth whoami &>/dev/null; then
        info "Login successful."
    else
        error "HuggingFace login failed. Re-run this script after sorting it out."
    fi
fi

# ---- Step 4: Download models ----------------------------------------------

export HF_HOME="$HF_CACHE_DIR"
mkdir -p "$HF_CACHE_DIR"

info "Downloading models to $HF_CACHE_DIR ..."
info "  (This will take a while on first run — ~16GB per model)"
info "  (If you haven't accepted the license agreements, this will fail.)"

python -c "
from huggingface_hub import snapshot_download
import sys

models = [
    'meta-llama/Llama-3.1-8B-Instruct',
    'google/gemma-2-9b-it',
]

for m in models:
    print(f'Downloading {m}...', flush=True)
    try:
        snapshot_download(m)
        print(f'  Done: {m}', flush=True)
    except Exception as e:
        print(f'  FAILED: {m} — {e}', file=sys.stderr, flush=True)
        print(f'  Make sure you accepted the license at https://huggingface.co/{m}', file=sys.stderr, flush=True)
        sys.exit(1)
"

info "Models downloaded."

conda deactivate

# ---- Step 5: Clone repo ---------------------------------------------------

if [ -d "$REPO_DIR" ]; then
    info "Repo already exists at $REPO_DIR"
    info "  Fetching latest and checking out $GIT_BRANCH..."
    cd "$REPO_DIR"
    git fetch --all
    git checkout "$GIT_BRANCH" 2>/dev/null || git checkout -b "$GIT_BRANCH" "origin/$GIT_BRANCH"
    git pull origin "$GIT_BRANCH" || true
else
    info "Cloning repo..."
    git clone "$GIT_REPO" "$REPO_DIR"
    cd "$REPO_DIR"
    git checkout "$GIT_BRANCH" 2>/dev/null || git checkout -b "$GIT_BRANCH" "origin/$GIT_BRANCH"
fi

mkdir -p outputs logs

# ---- Step 6: Verify -------------------------------------------------------

info ""
info "=========================================="
info "  Setup complete!"
info "=========================================="
info ""
info "  Conda env:     $ENV_NAME"
info "  Repo:          $REPO_DIR"
info "  Branch:        $GIT_BRANCH"
info "  HF cache:      $HF_CACHE_DIR"
info "  Models:        Llama 3.1 8B Instruct, Gemma 2 9B IT"
info ""
info "  Quick test (run on login node, just checks imports):"
info "    conda activate $ENV_NAME"
info "    python -c \"import torch, transformers; print(torch.__version__, transformers.__version__)\""
info ""
info "  To submit a job:"
info "    cd $REPO_DIR"
info "    sbatch --export=ALL,MODEL=meta-llama/Llama-3.1-8B-Instruct,OCR=paddle_ocr_text,TAG=llama31_8b_paddle zero_shot/slurm/text_only.sbatch"
info ""
info "=========================================="
info ""
info "  IMPORTANT REMINDERS:"
info "  - The CSV is Windows-1252 encoded (cp1252), NOT utf-8."
info "    Always read with: pd.read_csv(path, sep=',', encoding='cp1252')"
info "  - Gemma 2 does NOT support system role in its chat template."
info "    System prompts must be merged into the user message."
info "  - Do NOT install OCR packages into this conda env."
info "  - Use killable.q partition with --requeue for GPU jobs."
info "  - The run script has resume support — if a job gets preempted,"
info "    just resubmit and it picks up where it left off."
info ""

# ===========================================================================
# WHERE TO LOOK FOR EXTENDING THE PIPELINE
# ===========================================================================
#
# The current pipeline does zero-shot prompting (no examples, no training).
# Here's where to hook in for other approaches:
#
# FEW-SHOT PROMPTING (Manny):
#   - Create a `few_shot/` directory mirroring `zero_shot/`.
#   - The key change is in `prompts.py`: instead of just a system prompt +
#     raw OCR, you include N labeled examples (OCR text → expected JSON)
#     before the actual query. The run script stays almost identical.
#   - Copy `zero_shot/run_text_only.py` → `few_shot/run_text_only.py`
#     and change the import to `from few_shot.prompts import ...`
#   - Copy `zero_shot/slurm/text_only.sbatch` → `few_shot/slurm/text_only.sbatch`
#     and change `python -m zero_shot.run_text_only` → `python -m few_shot.run_text_only`
#   - The evaluator (`zero_shot/evaluate.py`) works on any JSONL with the
#     same schema — just point it at your output file.
#   - Main design questions: how many examples (1? 3? 5?), which examples
#     (random? diverse?), does the context window fit (Llama 8K default)?
#
# FINE-TUNING (Sijan):
#   - Create a `fine_tuning/` directory.
#   - You need a train/test split of the dataset. Coordinate with the team
#     so zero-shot and few-shot are evaluated on the SAME test set.
#   - Look into LoRA or QLoRA via the `peft` library:
#       pip install peft trl datasets
#   - The training script will load the base model, apply a LoRA adapter,
#     and train on (OCR text → ground truth JSON) pairs.
#   - For inference, load base model + trained adapter, then run identically
#     to the zero-shot script.
#   - The same evaluator works on the output JSONL.
#   - Main design questions: which base model, LoRA rank, learning rate,
#     number of epochs, train/test split ratio.
#
# SHARED EVALUATOR (future):
#   - Once the team agrees on scoring policy, move the evaluator to
#     `shared/evaluate.py` and have all three approaches use it.
#   - The JSONL schema is the interface contract:
#       {"image_id": int, "ground_truth": str|dict, "prediction": dict|null,
#        "error": str|null, "raw_response": str|null}
#     As long as everyone outputs this format, one evaluator scores all.
#
# ===========================================================================
