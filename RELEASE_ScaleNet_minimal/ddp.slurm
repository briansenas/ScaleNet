#!/bin/bash
#SBATCH --job-name=ScaleNet      # create a short name for your job
#SBATCH --partition dios         # Cola para ejecutar
#SBATCH --nodelist atenea        # Nodo en concreto -- titan works.
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=3      # total number of tasks per node
#SBATCH --gres=gpu:3             # number of allocated gpus per node
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=./slurmReports/output.%a.%j.out # STDOUT

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_JOB_NUM_NODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE
export SLURM_GPUS_ON_NODE=3

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/bsena/ScaleNet/.venv
export TFHUB_CACHE_DIR=.

# Calibrate camera parameters first
srun python scaleNet-calib.py --task-name SUN360RCNN-NEW --save-every-iter 500 --summary-every-iter 500 --evaluate-every 1000 --iter 50000 --reset-iter --reset-scheduler --reset-lr --calib-file config/pano360_crops_dataset_cvpr_myDistWider_train.json
# Run the keypoint detection training phase
# srun python scaleNet-CocoScale.py --task-name KPSBBOX --est-bbox --est-kps --train-cameraCls --reset-iter --reset-lr --reset-scheduler --save-every-iter 500 --summary-every-iter 500 --iter 50000 --calib-file config/pano360_crops_dataset_cvpr_myDistWider_train.json --opts MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN images_per_gpu x 1000
# Run the height estimation height
# srun python scaleNet-CocoScale.py --task-name ROIH --resume KPSBBOX --est-bbox --est-kps --train-cameraCls --train-roi-h --pointnet-camH --pointnet-camH-refine --pointnet-personH-refine --accu-model --loss-person-all-layers --num-layers 3 --reset-iter --reset-lr --reset-scheduler --save-every-iter 500 --summary-every-iter 500 --iter 50000 --calib-file config/pano360_crops_dataset_cvpr_myDistWider_train.json --opts MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN images_per_gpu x 1000
