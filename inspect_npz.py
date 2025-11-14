import numpy as np
import argparse
from pprint import pprint

# -----------------------------
# Parse Arguments
# -----------------------------
parser = argparse.ArgumentParser(description="Inspect a methylation NPZ file")
parser.add_argument("--file", type=str, required=True, help="Path to .npz file")
args = parser.parse_args()

# -----------------------------
# Load file
# -----------------------------
print(f"\nLoading: {args.file}\n")
data = np.load(args.file, allow_pickle=True)

# -----------------------------
# List keys
# -----------------------------
print("Keys in file:", list(data.keys()), "\n")

# -----------------------------
# Show shapes and dtypes
# -----------------------------
for key in data.keys():
    arr = data[key]
    if isinstance(arr, np.ndarray):
        print(f"{key:12s}  shape={arr.shape}  dtype={arr.dtype}")
    else:
        print(f"{key:12s}  type={type(arr)}")

print("\n")

# -----------------------------
# Peek at data samples
# -----------------------------
try:
    print("First CpG DNA one-hot slice (trimmed):")
    print(data["dna"][0][:, :20])  # first 20 bp for readability

    print("\nFirst CpG histone signals:")
    print(data["histone"][0])

    print("\nFirst CpG methyl value:")
    print(data["methyl"][0])

    print("\nFirst CpG coordinates:")
    print(data["coords"][0])

    print("\nHistone marks order:")
    pprint(data["histone_names"])
except Exception as e:
    print("\n⚠️ Could not preview data elements — maybe file is empty?")
    print(e)
