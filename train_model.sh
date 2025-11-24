#!/bin/bash

echo "=========================================="
echo "METHYLATION MODEL TRAINING - CHR19"
echo "=========================================="
echo "Start time: $(date)"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source ~/methylation/myenv/bin/activate

# Check setup
echo ""
echo "Environment info:"
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
    python -c "import torch; print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
fi
echo ""

# Navigate to project
cd ~/methylation

# Create directories
mkdir -p checkpoints logs results

# Run training
echo "=========================================="
echo "Starting training on chr19..."
echo "  - Train: 60% (9,144 samples)"
echo "  - Val:   20% (3,048 samples)"
echo "  - Test:  20% (3,048 samples)"
echo "  - Epochs: 50 (with early stopping)"
echo "  - Batch size: 64"
echo "  - Learning rate: 1e-4"
echo "=========================================="
echo ""

python src/train.py 2>&1 | tee logs/train_$(date +%Y%m%d_%H%M%S).log

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ TRAINING COMPLETED SUCCESSFULLY!"
    echo "=========================================="
    echo ""
    echo "Results saved in:"
    ls -lh checkpoints/best_model.pth 2>/dev/null && echo "  ✓ checkpoints/best_model.pth"
    echo "  ✓ logs/train_*.log"
    echo ""
    echo "Next steps:"
    echo "  1. Check the training log for final metrics"
    echo "  2. Evaluate on test set: python src/evaluate.py"
else
    echo ""
    echo "=========================================="
    echo "✗ TRAINING FAILED"
    echo "=========================================="
    echo "Check logs/train_*.log for details"
    exit 1
fi

echo ""
echo "End time: $(date)"