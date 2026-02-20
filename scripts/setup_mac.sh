#!/bin/bash
# Setup script for Mac (MPS) - installs all dependencies for ao_interpret_demo notebook
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== JB_mech Mac/MPS Setup ==="
echo "Project root: $PROJECT_ROOT"
echo ""

# Check for venv
if [ ! -d "$PROJECT_ROOT/.venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$PROJECT_ROOT/.venv"
fi

# Activate venv
source "$PROJECT_ROOT/.venv/bin/activate"
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install Mac-compatible requirements
echo ""
echo "Installing Mac-compatible dependencies..."
pip install -r "$PROJECT_ROOT/requirements-mac.txt"

# Install the activation_oracles third-party package (editable mode)
# Note: pyproject.toml has been modified to skip CUDA-only deps (flash-attn, vllm, liger-kernel)
echo ""
echo "Installing activation_oracles (third-party)..."
cd "$PROJECT_ROOT/third_party/activation_oracles"
pip install -e . --no-build-isolation

# Install jb-mech package
echo ""
echo "Installing jb-mech package..."
cd "$PROJECT_ROOT"
pip install -e . --no-deps 2>/dev/null || pip install -e . 2>/dev/null || true

# Verify installation
echo ""
echo "=== Verification ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'CUDA available: {torch.cuda.is_available()}')
device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Default device: {device}')
"

echo ""
python -c "
print('Checking key imports...')
import transformers; print(f'  transformers: {transformers.__version__}')
import peft; print(f'  peft: {peft.__version__}')
import accelerate; print(f'  accelerate: {accelerate.__version__}')
import datasets; print(f'  datasets: {datasets.__version__}')
print('All key dependencies installed!')
"

echo ""
echo "=== Setup Complete ==="
echo "Activate with: source .venv/bin/activate"
echo "Run notebook: jupyter notebook notebooks/ao_interpret_demo.ipynb"
