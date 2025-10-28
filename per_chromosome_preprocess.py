import pyBigWig, numpy as np, pandas as pd, os
from pyfaidx import Fasta
import argparse  

parser = argparse.ArgumentParser(description="Extract methylation and histone signal for CpGs")
parser.add_argument("--chrom", type=str, default=None, help="Chromosome to process (e.g. chr22)")
args = parser.parse_args()

# ---------- CONFIG ----------
bw_dir = "histone_bigwigs"
methyl_bw_path = "raw_data/methylation_signal_GRCh38.bigWig"
cpg_bed = "raw_data/cpg_sites.bed"   # chr, start, end
fasta_path = "raw_data/GRCh38.primary_assembly.genome.fa"
out_dir = "./per_chrom_npz"
os.makedirs(out_dir, exist_ok=True)

window, bin_size = 250, 100 # this is configurable
bins = np.arange(-window, window, bin_size)
n_bins, nucleotides = len(bins), ["A", "C", "G", "T"]

bw_files = sorted([os.path.join(bw_dir, f) for f in os.listdir(bw_dir) if f.endswith(".bigWig")])
bw_handles = [pyBigWig.open(f) for f in bw_files]
histone_names = [os.path.basename(f).replace(".bigWig", "") for f in bw_files]
ref, methyl_bw = Fasta(fasta_path), pyBigWig.open(methyl_bw_path)

def extract_histone_bins(bw, chrom, center):
    vals = []
    for b in bins:
        start, end = max(center + b, 0), max(center + b + bin_size, 0)
        try:
            v = bw.stats(chrom, int(start), int(end), type="mean")[0]
        except RuntimeError:
            v = None
        vals.append(v if v is not None else np.nan)
    return vals

def one_hot_encode(seq):
    seq = seq.upper()
    arr = np.zeros((4, len(seq)), dtype=np.float32)
    for i, nt in enumerate(seq):
        if nt in nucleotides: arr[nucleotides.index(nt), i] = 1
    return arr

def extract_dna(chrom, center):
    start, end = max(center - window, 0), center + window + 1
    seq = str(ref[chrom][start:end])
    return one_hot_encode(seq)

def extract_methylation(chrom, center):
    val = methyl_bw.stats(chrom, int(center), int(center)+1, type="mean")[0]
    return val if val is not None else np.nan

# ---------- MAIN LOOP ----------
cpgs = pd.read_csv(cpg_bed, sep="\t", header=None, names=["chr", "start", "end"])

if args.chrom:
    cpgs = cpgs[cpgs["chr"] == args.chrom]

for chrom, sub in cpgs.groupby("chr"):
    dna_data, histone_data, methyl_vals, coords = [], [], [], []
    print(f"Processing {chrom} with {len(sub)} CpGs")

    for _, row in sub.iterrows():
        center = (row["start"] + row["end"]) // 2
        dna = extract_dna(chrom, center)
        histone_matrix = np.array([extract_histone_bins(bw, chrom, center) for bw in bw_handles])
        histone_matrix = np.log2(histone_matrix + 1e-6)
        methyl = extract_methylation(chrom, center)

        if np.isnan(methyl):  # skip missing
            continue
        dna_data.append(dna)
        histone_data.append(histone_matrix)
        methyl_vals.append(methyl)
        coords.append((chrom, center - window, center + window))

    dna_data = np.stack(dna_data)
    histone_data = np.stack(histone_data)
    methyl_vals = np.array(methyl_vals, dtype=np.float32)
    coords = np.array(coords, dtype=object)

    np.savez_compressed(
        os.path.join(out_dir, f"{chrom}.npz"),
        dna=dna_data, histone=histone_data,
        methyl=methyl_vals, coords=coords,
        histone_names=histone_names
    )
    print(f"Saved {chrom}.npz → {dna_data.shape}")
