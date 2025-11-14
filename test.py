import time
print("[0] Starting PyTorch import test...")

t0 = time.time()
try:
    import torch
except Exception as e:
    print("❌ PyTorch failed to import:")
    print(e)
    raise SystemExit(1)
print(f"✅ torch imported successfully in {time.time() - t0:.2f}s")
print(f"PyTorch version: {torch.__version__}")

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")
if cuda_available:
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU detected — will use CPU.")

# Simple tensor test
try:
    x = torch.rand((3, 3))
    y = torch.rand((3, 3))
    z = torch.matmul(x, y)
    print("✅ CPU tensor multiply passed.")
    if cuda_available:
        x_gpu = x.to("cuda")
        y_gpu = y.to("cuda")
        z_gpu = torch.matmul(x_gpu, y_gpu)
        print("✅ GPU tensor multiply passed.")
except Exception as e:
    print("❌ Tensor operation failed:")
    print(e)
    raise SystemExit(1)

print("\nAll Torch diagnostics passed successfully.")
