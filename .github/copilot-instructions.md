## Project quick orientation

This repository trains and evaluates a small PyTorch model to predict CpG methylation from DNA sequence and histone signal vectors. Key scripts:

- `per_chromosome_preprocess.py` — produces per-chromosome `.npz` files (one CpG per example) into `per_chrom_npz/`.
- `train_chr22.py` — lightweight training script that loads `per_chrom_npz/chr22.npz` and saves `chr22_model.pth`.
- `eval_chr22.py` / `inspect_npz.py` — evaluation and lightweight inspection tools for `.npz` files and predictions.
- `merge_data.py` — simple merger for multiple per-chromosome `.npz` files into `merged_dataset.npz`.
- `model.py` / `dataset.py` — model and DataLoader wrappers used by training/eval.

## What the AI assistant should know (concise)

- Data shape expectations: `.npz` files contain keys `dna` (one-hot 4xL arrays per CpG), `histone` (4-length vectors per CpG), `methyl` (float label), and `coords` (coordinate tuples). Use `inspect_npz.py` to confirm shapes.
- Training split convention: `train_chr22.py` uses 60% train, 20% val, 20% test (computed with simple index slicing). Keep that convention when adding experiments.
- Environment: the repo assumes a Python venv at `myenv/` or `myenv/bin/activate` as used in SLURM scripts. `requirements.txt` lists Python deps — prefer editing that file when adding packages.
- Job runner: there are SLURM wrappers (`preprocess_chr22.sh`, `preprocess_all.sh`) that load `python/3.10` and activate a venv. Don't change SLURM conventions unless necessary.

## Coding patterns & conventions

- Minimal, explicit scripts: training and evaluation are designed as standalone scripts (top-level `if __name__ == "__main__":` blocks). For larger refactors, keep CLI entrypoints simple and add small helper functions.
- Numpy .npz usage: code uses `np.load(..., allow_pickle=True)`; treat `.npz` arrays as numpy first, convert to torch tensors in `dataset.py`.
- Device handling: scripts detect device with `device = 'cuda' if torch.cuda.is_available() else 'cpu'` and move model/data accordingly.
- Checkpointing: `train_chr22.py` saves a full `state_dict` to `chr22_model.pth`. When changing model architecture, update load/save accordingly.

## Helpful example edits you can make

- Add a new training script for merged dataset: use `merge_data.py` output (`merged_dataset.npz`) and follow the same DataLoader and split logic in `train_chr22.py`.
- When extending `model.py`, keep module names and exported class `BasicMethylationNet` so existing scripts continue to import it.
- Add a small CLI flag to `train_chr22.py` to choose batch size / epochs — follow existing verbose printing style.

## Useful commands (reproducible locally)

1. Inspect an NPZ (quick check):

   python inspect_npz.py --file per_chrom_npz/chr22.npz

2. Train on chr22 locally (no slurm):

   python train_chr22.py

3. Submit preprocessing jobs on cluster (uses SLURM; update venv path if needed):

   sbatch preprocess_chr22.sh
   sbatch preprocess_all.sh

4. Merge all per-chromosome npz files:

   python merge_data.py

## Where to look for behavior examples

- Data creation: `per_chromosome_preprocess.py` — examine how histone bigWigs (`histone_bigwigs/`) are read and how `histone_names` is stored in `.npz` files.
- Training flow & logging: `train_chr22.py` — contains simple progress prints and batch-level logging (every 100 batches).
- Evaluation: `eval_chr22.py` — shows how predictions are saved (`chr22_preds.npy`) and how coords are paired for downstream analysis.

## Edge cases to be mindful of

- `.npz` files may be large — prefer streaming or batching when adding tools that load multiple chromosomes.
- `per_chrom_npz_all/` exists as an alternate output location — double-check which folder a script writes to before modifying paths.
- SLURM scripts expect certain environment paths (example: venv at `/home/ekourb/scratch/methylation/myenv/`) — parameterize paths when adding new job scripts.

If anything here is unclear or you'd like more details about a specific file or workflow, tell me what to expand and I will iterate.
