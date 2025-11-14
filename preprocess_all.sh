#!/bin/bash
#SBATCH --job-name=cpg_preproc
#SBATCH --account=def-majewski
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=1-21            # ONLY chr1–chr21
#SBATCH --output=logs/preproc_%A_%a.out
#SBATCH --error=logs/preproc_%A_%a.err

module load python/3.10
source /home/ekourb/scratch/methylation/myenv/bin/activate

mkdir -p logs per_chrom_npz

# chromosome list without chr22
CHROMS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21)


CHR=${CHROMS[$SLURM_ARRAY_TASK_ID-1]}

echo "Processing chr${CHR}"

python per_chromosome_preprocess.py --chrom chr${CHR}
