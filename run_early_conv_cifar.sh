#!/bin/bash
#SBATCH -n 1                # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -t 0-10:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p kempner_h100  # Partition to submit to
#SBATCH --account kempner_pehlevan_lab
#SBATCH --cpus-per-gpu=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=80G
#SBATCH -o log_files/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e log_files/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid

module load cuda/12.2.0-fasrc01
module load cudnn/8.9.2.26_cuda12-fasrc01
module load Mambaforge/22.11.1-fasrc01
#mamba activate scaling_laws_mamba

nvidia-smi

/n/home08/bbordelon/.conda/envs/flax/bin/python early_conv_vit_sweeps.py --width 4 --heads -1 --depth 2 --scale_exp 1.0 --steps 1000 --batch_size 128
