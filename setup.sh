#!/usr/bin/env bash
# setup.sh — run once before starting the overnight loop
set -euo pipefail

echo "=== BERT Distillation Auto-Researcher Setup ==="

# Install dependencies
pip install torch transformers datasets accelerate --quiet

# Pre-download tokenizer and pretrained tiny BERT (optional init)
echo "Caching tokenizer and tiny BERT init weights..."
python3 - <<'EOF'
from transformers import AutoTokenizer, BertModel
AutoTokenizer.from_pretrained("bert-base-uncased")
BertModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
print("Tokenizer and pretrained init weights cached.")
EOF

# Pre-download datasets
echo "Caching ms_marco v1.1 and SQuAD..."
python3 - <<'EOF'
from datasets import load_dataset
load_dataset("ms_marco", "v1.1", split="train[:1%]")
load_dataset("ms_marco", "v1.1", split="validation[:1%]")
load_dataset("squad", split="train[:1%]")
print("Datasets cached.")
EOF

# Create working directories
mkdir -p logs checkpoints

# Initialize empty results and todo files
touch results.txt
touch todo.log

echo ""
echo "=== Setup complete ==="
echo "Start the overnight loop with:"
echo "  python3 scheduler.py 2>&1 | tee logs/scheduler_\$(date +%Y%m%d_%H%M%S).log"
echo ""
echo "KEY CONSTRAINT: Student must be <10MB fp16."
echo "Factorized embeddings (vocab→embed_dim→hidden_dim) are mandatory."
echo "Example: embed=32, hidden=128, 2 layers ≈ 2.6MB. embed=64, hidden=256, 4 layers ≈ 8.2MB."
