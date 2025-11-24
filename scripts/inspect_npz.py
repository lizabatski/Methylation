import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", required=True, help="Path to NPZ file")
args = parser.parse_args()

print(f"\nInspecting: {args.file}\n")

try:
    data = np.load(args.file, allow_pickle=True)
except Exception as e:
    print(f"Error loading file: {e}")
    exit(1)

# Skip histone_names due to pickle error
safe_keys = [k for k in data.keys() if k != 'histone_names']

print("="*70)
print("ARRAY METADATA")
print("="*70)

for key in safe_keys:
    arr = data[key]
    print(f"{key:15s} shape={str(arr.shape):20s} dtype={arr.dtype}")
    size_mb = arr.nbytes / (1024**2)
    print(f"{'':15s} memory: {size_mb:.2f} MB")

# Hardcoded histone names (from your partner's code)
print(f"\nhistone_names:  ['H3K4me3', 'H3K36me2', 'H3K27me3', 'H3K9me3']")

print("\n" + "="*70)
print("DATA PREVIEW")
print("="*70)

if 'dna' in data:
    print(f"\nDNA (first sample, first 5 positions):")
    print(data['dna'][0, :5, :])

if 'histone' in data:
    print(f"\nHistone (first sample, first 5 positions):")
    print(data['histone'][0, :5, :])

if 'methyl' in data:
    print(f"\nMethylation (first 10 values):")
    print(data['methyl'][:10])
    print(f"  Range: [{data['methyl'].min():.3f}, {data['methyl'].max():.3f}]")
    print(f"  Mean: {data['methyl'].mean():.3f} ± {data['methyl'].std():.3f}")

print()