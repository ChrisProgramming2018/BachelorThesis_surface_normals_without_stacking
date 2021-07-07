#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080 # partition (queue)
#SBATCH --mem 64G # memory pool for all cores (4GB)
#SBATCH --time 24:00:00 # time (D-HH:MM)
#SBATCH -c 16 # number of cores
##SBATCH -a 1-4 # array size
#SBATCH --gres=gpu:1  # reserves one GPU
#SBATCH -D /work/dlclarge1/leiningc-chris_workspace/BachelorThesisSimToReal_parallel # Change working_dir
#SBATCH --mail-type=END,FAIL # (recive mails about end and timeouts/crashes of your job)
#SBATCH -o /work/dlclarge1/leiningc-chris_workspace/BachelorThesisSimToReal_parallel/%x.%N.%j.out # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value, whilst %a will be replaced by the SLURM_ARRAY_TASK_ID
#SBATCH -e /work/dlclarge1/leiningc-chris_workspace/BachelorThesisSimToReal_parallel/%x.%N.%j.err # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value, whilst %a will be replaced by the SLURM_ARRAY_TASK_ID
source ~/.bashrc
conda activate sn1
echo "start"
sleep 1
srun python /work/dlclarge1/leiningc-chris_workspace/BachelorThesisSimToReal_parallel/main.py
echo "end"
