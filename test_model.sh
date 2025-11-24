#!/bin/bash

echo "=========================================="
echo "METHYLATION MODEL TEST"
echo "=========================================="
echo "Start time: $(date)"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo ""

# Activate your virtual environment
echo "Activating virtual environment..."
source ~/methylation/myenv/bin/activate

# Check Python and packages
echo ""
echo "Python version:"
python --version
echo ""
echo "PyTorch version:"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
fi
echo ""

# Navigate to project directory
cd ~/methylation

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the test
echo "=========================================="
echo "Running model tests..."
echo "=========================================="
python src/test_model.py 2>&1 | tee logs/test_$(date +%Y%m%d_%H%M%S).log

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ ALL TESTS PASSED!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "  1. Review the test output above"
    echo "  2. Run full training: bash train_model.sh"
else
    echo ""
    echo "=========================================="
    echo "✗ TESTS FAILED"
    echo "=========================================="
    echo "Check the error messages above"
    exit 1
fi

echo ""
echo "End time: $(date)"