#!/bin/sh
#SBATCH --cpus-per-task=8 # Number of cores
#SBATCH --mem=128gb  # Total RAM in GB
#SBATCH --time=16:00:00  # Time limit hrs:min:sec; for using days use --time= days-hrs:min:sec

#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1 # A100 GPU

#SBATCH --job-name=EXP1 # Job name
#SBATCH --mail-type=ALL # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=moajjem.chowdhury@louisville.edu # Where to send mail
#SBATCH --output=R_%x_%j.out # Standard output and error log

echo "Current working directory: $(pwd)"
echo "Starting on HOST: $(hostname)"
echo "Starting on DATE: $(date)"

# Load Conda first
echo "Loading Conda"
module load conda
echo "Activating Conda environment: labram"
conda activate labram
echo "Conda environment labram activated successfully"

# Python script
echo "Running python script"
python hparams_tune.py

echo "Ending on DATE: $(date)"