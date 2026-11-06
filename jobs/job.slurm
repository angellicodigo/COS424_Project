#!/bin/bash
#SBATCH --job-name=iclr       
#SBATCH --nodes=1                   # node count
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=4           # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=16G           # memory per cpu-core (4G is default)
#SBATCH --time=6:00:00              # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:1                # SBATCH --constraint=a100     
#SBATCH --mail-type=end,fail  
#SBATCH --mail-user=al5098@princeton.edu

module purge
module load anaconda3/2026.7
conda activate iclr

cd /home/al5098/GitHub/COS424_Project/scripts/
python iclr_analysis.py --output_path /scratch/network/al5098/COS424 --autotokenizer_path /scratch/network/al5098/COS424/autotokenizer