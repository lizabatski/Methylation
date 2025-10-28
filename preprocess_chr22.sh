#!/bin/bash
#SBATCH --job-name=cpg_chr22
#SBATCH --account=def-majewski
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/chr22_%A.out
#SBATCH --error=logs/chr22_%A.err

module load python/3.10
source /home/ekourb/scratch/methylation/myenv/bin/activate

mkdir -p logs per_chrom_npz

python per_chromosome_preprocess.py --chrom chr22
