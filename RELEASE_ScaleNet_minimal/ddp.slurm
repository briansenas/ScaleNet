#!/bin/bash
#SBATCH --job-name=ScaleNet      # create a short name for your job
#SBATCH --partition dios         # Cola para ejecutar
#SBATCH --nodelist atenea        # Nodo en concreto
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=2      # total number of tasks per node
#SBATCH --gres=gpu:2             # number of allocated gpus per node
#SBATCH --time=00:10:00          # total run time limit (HH:MM:SS)
#SBATCH --output=./slurmReports/output.%a.out # STDOUT

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_JOB_NUM_NODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE
export SLURM_GPUS_ON_NODE=2

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/bsena/ScaleNet/.venv
export TFHUB_CACHE_DIR=.

srun python scaleNet-DP.py
