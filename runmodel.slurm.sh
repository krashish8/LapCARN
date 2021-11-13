#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32  
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --job-name="l1"


cd /scratch/ssrivastava.cse18.iitbhu/CARN
module load python3.7/3.7
module load cuda/10.1
source venv/bin/activate
#python main.py --model ensr --save ENSR_lrmod_1 --scale 2 --reset --save_results --patch_size 448 --loss 1*MSE
#python3 newensr.py 2>&1 | tee -a output.txt
#python3 carn/train.py --patch_size 64 --batch_size 8 --max_steps 600000 --decay 400000 --model lapcarn --ckpt_name lapcarn --ckpt_dir checkpoint/lapcarn --scale 8 --num_gpu 1 --print_interval 1
python3 carn/train.py --patch_size 64 --batch_size 8 --max_steps 600000 --decay 400000 --model lapcarn --ckpt_name lapcarn --ckpt_dir checkpoint/lapcarn --scale 8 --num_gpu 2 --print_interval 100
