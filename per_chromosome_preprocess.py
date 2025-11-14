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
out_dir = "./per_chrom_npz_all"
os.makedirs(out_dir, exist_ok=True)

# Window and binning configuration
window, bin_size = 250, 100
# Source: You want 100bp resolution for histones
# This gives: 500bp total / 100bp per bin = 5 bins
bins = np.arange(-window, window, bin_size)
n_bins, nucleotides = len(bins), ["A", "C", "G", "T"]

print(f"Configuration:")
print(f"  Window: ±{window}bp")
print(f"  Bin size: {bin_size}bp")
print(f"  Number of bins: {len(bins)}")
print(f"  Bins: {bins}")

bw_files = sorted([os.path.join(bw_dir, f) for f in os.listdir(bw_dir) if f.endswith(".bigWig")])
bw_handles = [pyBigWig.open(f) for f in bw_files]
histone_names = [os.path.basename(f).replace(".bigWig", "") for f in bw_files]

print(f"\nHistone marks: {histone_names}")

ref, methyl_bw = Fasta(fasta_path), pyBigWig.open(methyl_bw_path)

def extract_histone_bins(bw, chrom, center):
    """
    Extract histone signal in bins around CpG center.
    
    Source: Bichrom approach - binned histone signals
    "raw coverage counts are binned into non-overlapping bins, 
    total tag normalized for each replicate"
    """
    vals = []
    for b in bins:
        start, end = max(center + b, 0), max(center + b + bin_size, 0)
        try:
            v = bw.stats(chrom, int(start), int(end), type="mean")[0]
        except RuntimeError:
            v = None
        # Use 0.0 instead of np.nan for missing values
        # Source: Common practice - allows for easier downstream processing
        vals.append(v if v is not None else 0.0)
    return vals

def one_hot_encode(seq):
    """
    One-hot encode DNA sequence.
    
    Source: Standard approach in DeepCpG and genomic deep learning
    Each nucleotide represented as vector: A=[1,0,0,0], C=[0,1,0,0], etc.
    """
    seq = seq.upper()
    arr = np.zeros((4, len(seq)), dtype=np.float32)
    for i, nt in enumerate(seq):
        if nt in nucleotides: 
            arr[nucleotides.index(nt), i] = 1
    return arr

def extract_dna(chrom, center):
    """
    Extract DNA sequence centered on CpG.
    
    Source: DeepCpG - "DNA module takes DNA sequences in windows 
    centred on target CpG sites as input"
    
    Returns: (4, 501) array - one-hot encoded sequence
    """
    start, end = max(center - window, 0), center + window + 1
    seq = str(ref[chrom][start:end])
    return one_hot_encode(seq)

def extract_methylation(chrom, center):
    """
    Extract and normalize methylation value to [0, 1].
    
    Source: "Epigenetic clocks use beta values ranging from 0 
    (unmethylated) to 1 (fully methylated)" - AltumAge and other 
    methylation age predictors
    """
    val = methyl_bw.stats(chrom, int(center), int(center)+1, type="mean")[0]
    
    if val is None:
        return np.nan
    
    # CRITICAL FIX: Normalize methylation values to [0, 1]
    # Check if values are on 0-100 scale (percentages)
    if val > 1.0:
        val = val / 100.0
    
    # Ensure valid range
    val = np.clip(val, 0.0, 1.0)
    
    return val

def process_histone_matrix(histone_matrix):
    """
    Process histone signals with proper handling of zeros.
    
    Source considerations:
    1. log2(x + 1e-6) creates extreme negative values (-19.93) for zeros
    2. log1p(x) = log(1+x) maps 0→0, avoiding extreme negatives
    3. Clipping prevents outliers from dominating gradients
    
    Final approach: Use log1p with clipping
    """
    # Convert to numpy array if not already
    histone_matrix = np.asarray(histone_matrix, dtype=np.float32)
    
    # Replace any remaining NaN with 0
    histone_matrix = np.nan_to_num(histone_matrix, nan=0.0)
    
    # Use log1p transformation: log(1 + x)
    # Source: Standard in RNA-seq and genomics for count data
    # Advantage: log1p(0) = 0, avoiding extreme negatives
    histone_matrix = np.log1p(histone_matrix)
    
    # Clip extreme values to prevent outliers
    # Source: Common practice in deep learning to prevent gradient issues
    histone_matrix = np.clip(histone_matrix, -5, 10)
    
    return histone_matrix

# ---------- MAIN LOOP ----------
cpgs = pd.read_csv(cpg_bed, sep="\t", header=None, names=["chr", "start", "end"])

if args.chrom:
    cpgs = cpgs[cpgs["chr"] == args.chrom]
    print(f"\nProcessing only chromosome: {args.chrom}")

print(f"\nTotal CpGs to process: {len(cpgs)}")

for chrom, sub in cpgs.groupby("chr"):
    dna_data, histone_data, methyl_vals, coords = [], [], [], []
    print(f"\nProcessing {chrom} with {len(sub)} CpGs...")
    
    skipped_count = 0
    
    for idx, row in sub.iterrows():
        center = (row["start"] + row["end"]) // 2
        
        # Extract features
        dna = extract_dna(chrom, center)
        histone_matrix = np.array([extract_histone_bins(bw, chrom, center) for bw in bw_handles])
        histone_matrix = process_histone_matrix(histone_matrix)
        methyl = extract_methylation(chrom, center)
        
        # Skip if methylation is missing
        if np.isnan(methyl):
            skipped_count += 1
            continue
        
        # Quality checks
        if dna.shape != (4, 501):
            print(f"  Warning: DNA shape mismatch at {chrom}:{center}: {dna.shape}")
            skipped_count += 1
            continue
            
        if histone_matrix.shape != (len(bw_handles), len(bins)):
            print(f"  Warning: Histone shape mismatch at {chrom}:{center}: {histone_matrix.shape}")
            skipped_count += 1
            continue
        
        if not (0 <= methyl <= 1):
            print(f"  Warning: Methylation out of range at {chrom}:{center}: {methyl}")
            skipped_count += 1
            continue
        
        dna_data.append(dna)
        histone_data.append(histone_matrix)
        methyl_vals.append(methyl)
        coords.append((chrom, center - window, center + window))
    
    if len(dna_data) == 0:
        print(f"  No valid CpGs found for {chrom}, skipping...")
        continue
    
    # Convert to arrays
    dna_data = np.stack(dna_data)
    histone_data = np.stack(histone_data)
    methyl_vals = np.array(methyl_vals, dtype=np.float32)
    coords = np.array(coords, dtype=object)
    
    # Print statistics
    print(f"  Valid CpGs: {len(dna_data)}")
    print(f"  Skipped CpGs: {skipped_count}")
    print(f"  DNA shape: {dna_data.shape}")
    print(f"  Histone shape: {histone_data.shape}")
    print(f"  Methylation range: [{methyl_vals.min():.3f}, {methyl_vals.max():.3f}]")
    print(f"  Methylation mean: {methyl_vals.mean():.3f} ± {methyl_vals.std():.3f}")
    print(f"  Histone range: [{histone_data.min():.3f}, {histone_data.max():.3f}]")
    
    # Save
    output_path = os.path.join(out_dir, f"{chrom}.npz")
    np.savez_compressed(
        output_path,
        dna=dna_data, 
        histone=histone_data,
        methyl=methyl_vals, 
        coords=coords,
        histone_names=histone_names
    )
    
    print(f"  Saved to: {output_path}")

print("\n" + "="*60)
print("Processing complete!")
print("="*60)