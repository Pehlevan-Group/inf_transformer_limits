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


## THIS SWEEP USED IN THE PAPER
#/n/home08/bbordelon/.conda/envs/flax/bin/python train_vit_cifar.py --save_model --gamma_zero 0.1 --width 16 --heads 16 --depth 32 --beta 4.0 --scale_exp 1.0 --steps 75000 --lr 0.2 --batch_size 64


/n/home08/bbordelon/.conda/envs/flax/bin/python train_vit_vary_depth_cifar.py --save_model --gamma_zero 0.1 --width 8 --heads 8 --depth 32 --depth_exp 1.0 --beta 6.0 --scale_exp 1.0 --steps 500000 --lr 0.2 --batch_size 64



#/n/home08/bbordelon/.conda/envs/flax/bin/python train_vit_vary_depth_cifar.py --save_model --gamma_zero 0.1 --width 16 --heads 16 --depth 64 --beta 4.0 --scale_exp 0.5 --depth_exp 0.5 --steps 75000 --lr 0.2 --batch_size 64

