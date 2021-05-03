#!/bin/bash

#SBATCH --gres=gpu:2
#SBATCH --partition=n1s16-t4-2
#SBATCH --account=dl18
#SBATCH --time=20:00:00
#SBATCH --output=demo_%j.out
#SBATCH --error=demo_%j.err
#SBATCH --exclusive
#SBATCH --requeue

/share/apps/local/bin/p2pBandwidthLatencyTest > /dev/null 2>&1

set -x 

mkdir /tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER

cp -rp /scratch/DL21SP/student_dataset.sqsh /tmp
echo "Dataset is copied to /tmp"

cd $HOME/NYU_DL_comp/barlow/

singularity exec --nv \
--bind /scratch \
--overlay /scratch/DL21SP/conda.sqsh:ro \
--overlay /tmp/student_dataset.sqsh:ro \
/share/apps/images/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
/bin/bash -c "
source /ext3/env.sh
conda activate dev
CUDA_VISIBLE_DEVICES=0,1 python3  main.py --data /dataset --epochs 200 --batch-size 128 --learning-rate 0.7 --lambd 0.0051 --projector 8192-8192-8192 --scale-loss 0.024 --checkpoint-dir $SCRATCH/checkpoints/barlow "
