## Template : à faire tourner pour utiliser le gpu.

"""
Nos identifiants : 
  *  Identifiant : user-data_challenge-31
  * Mot de passe : enpc_2627
"""

#!/bin/sh
#SBATCH --job-name=test
#SBATCH --ntasks-per-node=3
#SBATCH --ntasks-per-core=1
#SBATCH --partition=ird_gpu 
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --mem=8G

echo "Running on: $SLURM_NODELIST"
python source/entrainement.py






