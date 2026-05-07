#!/usr/bin/env bash
# =============================================================
#  SET50 Portfolio Optimization — Setup & Run
#  Python >= 3.11 required
# =============================================================

set -e  # Exit immediately if a command exits with a non-zero status

# ── 1. Create Virtual Environment ────────────────────────────
python3 -m venv .venv
source .venv/bin/activate

# ── 2. Install Dependencies ──────────────────────────────────
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✅  Setup complete. Environment: $(which python)"
echo ""
echo "Usage Instructions:"
echo "  source .venv/bin/activate          # Run this every time you open a new terminal"
echo ""
echo "  # 1. Prepare Data (Initial setup only)"
echo "  python src/data_pipeline.py --config configs/config.yaml"
echo ""
echo "  # 2. Training (Specify a run name or leave blank for timestamp)"
echo "  python src/train.py --config configs/config.yaml --run-name <Your name>"
echo ""
echo "  # 3. Evaluation & Visualization (Point to a specific run directory)"
echo "  python src/evaluation.py --run-dir results/<Your name>"
echo "  python src/backtest.py   --run-dir results/<Your name>"
echo ""
echo "  # 4. Resume from Checkpoint"
echo "  python src/train.py --config configs/config.yaml --run-name <Your name> \\"
echo "                      --resume results/<Your name>/checkpoints/epoch_0240.pt"