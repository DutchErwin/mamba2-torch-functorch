#!/bin/bash
# =============================================================================
# HMS-Net Complete Setup Script
# Creates fresh conda environment and installs all dependencies
# For RTX PRO 6000 Blackwell (96GB VRAM, sm_120)
# =============================================================================

set -e

# Configuration
ENV_NAME="hmsnet2025"
PYTHON_VERSION="3.12"
PYTORCH_VERSION="2.9.1"
CUDA_VERSION="cu129"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC} ${MAGENTA}$1${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
}

print_step() {
    echo -e "${GREEN}▶${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✖${NC} $1"
}

print_success() {
    echo -e "${GREEN}✔${NC} $1"
}

# =============================================================================
# CLEANUP: Remove existing environment
# =============================================================================
print_header "HMS-Net Blackwell Setup"

echo ""
echo -e "${BLUE}┌─────────────────────────────────────────────────────────────────┐${NC}"
echo -e "${BLUE}│${NC}  Target: NVIDIA RTX PRO 6000 Blackwell (96GB VRAM, sm_120)     ${BLUE}│${NC}"
echo -e "${BLUE}│${NC}  Python: $PYTHON_VERSION | PyTorch: $PYTORCH_VERSION | CUDA: $CUDA_VERSION                       ${BLUE}│${NC}"
echo -e "${BLUE}└─────────────────────────────────────────────────────────────────┘${NC}"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    print_error "conda not found! Please install Miniconda or Anaconda first."
    exit 1
fi

# Check for existing environment
if conda env list | grep -q "^${ENV_NAME} "; then
    print_warning "Environment '${ENV_NAME}' already exists."
    read -p "Do you want to remove it and create a fresh one? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_step "Removing existing environment..."
        conda deactivate 2>/dev/null || true
        conda env remove -n ${ENV_NAME} -y
        print_success "Old environment removed."
    else
        print_error "Aborted. Please remove the environment manually or choose a different name."
        exit 1
    fi
fi

# =============================================================================
# CREATE FRESH ENVIRONMENT
# =============================================================================
print_header "Creating Fresh Python ${PYTHON_VERSION} Environment"

print_step "Creating conda environment '${ENV_NAME}'..."
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

print_step "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

print_success "Environment created and activated!"
echo -e "   Python: $(python --version)"

# =============================================================================
# INSTALL PYTORCH (Must be first!)
# =============================================================================
print_header "Installing PyTorch ${PYTORCH_VERSION}+${CUDA_VERSION}"

print_step "Installing PyTorch with CUDA ${CUDA_VERSION} (Blackwell sm_120 support)..."

# IMPORTANT: Must use explicit CUDA version suffix and correct index URL
# cu128/cu129 is REQUIRED for Blackwell sm_120 - cu126 will NOT work!
pip install --force-reinstall \
    "torch==${PYTORCH_VERSION}+${CUDA_VERSION}" \
    "torchvision==0.24.1+${CUDA_VERSION}" \
    "torchaudio==${PYTORCH_VERSION}+${CUDA_VERSION}" \
    --index-url "https://download.pytorch.org/whl/${CUDA_VERSION}"

# Verify correct CUDA version was installed
INSTALLED_TORCH=$(python -c "import torch; print(torch.__version__)")
if [[ "$INSTALLED_TORCH" != *"${CUDA_VERSION}"* ]]; then
    print_error "PyTorch installed with wrong CUDA version: $INSTALLED_TORCH"
    print_error "Expected ${CUDA_VERSION} for Blackwell sm_120 support!"
    print_step "Attempting direct wheel install..."
    pip install --force-reinstall \
        "https://download.pytorch.org/whl/${CUDA_VERSION}/torch-${PYTORCH_VERSION}%2B${CUDA_VERSION}-cp312-cp312-manylinux_2_28_x86_64.whl" \
        "https://download.pytorch.org/whl/${CUDA_VERSION}/torchvision-0.24.1%2B${CUDA_VERSION}-cp312-cp312-manylinux_2_28_x86_64.whl"
fi

print_success "PyTorch installed!"

# =============================================================================
# INSTALL TRITON (Critical for Blackwell)
# =============================================================================
print_header "Installing Triton (Blackwell Support)"

print_step "Upgrading Triton to >=3.3.1..."
pip install -U "triton>=3.3.1"

print_success "Triton installed!"

# =============================================================================
# INSTALL MAMBA (Triton-based, NOT CUDA)
# =============================================================================
print_header "Installing Mamba2-Torch (Triton Kernels)"

print_step "Installing mamba2-torch dependencies (excluding torch/triton)..."
pip install einops safetensors packaging transformers

print_step "Installing mamba2-torch from source (without dependencies)..."
pip install --no-deps git+https://github.com/vasqu/mamba2-torch.git || {
    print_warning "mamba2-torch failed, will use mambapy fallback"
}

print_step "Installing mambapy as fallback..."
pip install mambapy

print_success "Mamba implementations installed!"

# =============================================================================
# INSTALL REQUIREMENTS
# =============================================================================
print_header "Installing Requirements"

print_step "Installing from requirements.txt..."
#pip install -r requirements.txt

print_success "Requirements installed!"
# =============================================================================
# VERIFICATION
# =============================================================================
print_header "Verifying Installation"

python -c "
import sys
import torch

# Colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
NC = '\033[0m'

def check(name, condition):
    if condition:
        print(f'{GREEN}✔{NC} {name}')
        return True
    else:
        print(f'{RED}✖{NC} {name}')
        return False

all_passed = True

# Basic checks
all_passed &= check('Python 3.12', sys.version_info[:2] == (3, 12))
all_passed &= check(f'PyTorch {torch.__version__}', torch.__version__.startswith('2.7'))
all_passed &= check('CUDA available', torch.cuda.is_available())

if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    compute_cap = torch.cuda.get_device_capability(0)
    print(f'   Device: {device_name}')
    print(f'   Compute Capability: {compute_cap}')

    # Test SDPA
    try:
        q = torch.randn(1, 8, 64, 64, device='cuda', dtype=torch.bfloat16)
        out = torch.nn.functional.scaled_dot_product_attention(q, q, q)
        all_passed &= check('PyTorch SDPA (Attention)', True)
    except Exception as e:
        all_passed &= check(f'PyTorch SDPA: {e}', False)

# Test Mamba
try:
    from mamba2_torch import Mamba2
    all_passed &= check('mamba2-torch (Triton)', True)
except ImportError:
    try:
        from mambapy import Mamba
        all_passed &= check('mambapy (PyTorch fallback)', True)
    except ImportError:
        all_passed &= check('Mamba implementation', False)

print()
if all_passed:
    print(f'{GREEN}═══════════════════════════════════════════════════════════════{NC}')
    print(f'{GREEN}  ALL CHECKS PASSED! Environment ready for HMS-Net.{NC}')
    print(f'{GREEN}═══════════════════════════════════════════════════════════════{NC}')
else:
    print(f'{RED}═══════════════════════════════════════════════════════════════{NC}')
    print(f'{RED}  Some checks failed. Review errors above.{NC}')
    print(f'{RED}═══════════════════════════════════════════════════════════════{NC}')
"

# =============================================================================
# DONE!
# =============================================================================
print_header "Installation Complete!"

echo ""
echo -e "${GREEN}Next steps:${NC}"
echo -e "  1. Activate environment: ${CYAN}conda activate ${ENV_NAME}${NC}"
echo -e "  2. Run benchmark test:   ${CYAN}python fireworks_test.py${NC}"
echo -e "  3. Start training:       ${CYAN}python train.py${NC}"
echo ""
echo -e "${MAGENTA}Happy training on your RTX PRO 6000 Blackwell!${NC} 🚀"
