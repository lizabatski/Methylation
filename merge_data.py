import numpy as np
import glob
import os

in_dir = "./per_chrom_npz"
out_path = "./merged_dataset.npz"

files = sorted(glob.glob(os.path.join(in_dir, "*.npz")))
print(f"Found {len(files)} chromosome files")

dna_all, histone_all, methyl_all, coords_all = [], [], [], []

# merge
for f in files:
    print(f"Merging {os.path.basename(f)} ...")
    data = np.load(f, allow_pickle=True)
    dna_all.append(data["dna"])
    histone_all.append(data["histone"])
    methyl_all.append(data["methyl"])
    coords_all.append(data["coords"])
    histone_names = data["histone_names"]  


dna = np.concatenate(dna_all)
histone = np.concatenate(histone_all)
methyl = np.concatenate(methyl_all)
coords = np.concatenate(coords_all)


np.savez_compressed(
    out_path,
    dna=dna,
    histone=histone,
    methyl=methyl,
    coords=coords,
    histone_names=histone_names
)

print(f"\nMerged dataset saved to {out_path}")
print(f"Dimensions print: DNA: {dna.shape}, Histone: {histone.shape}, Methyl: {methyl.shape}")
