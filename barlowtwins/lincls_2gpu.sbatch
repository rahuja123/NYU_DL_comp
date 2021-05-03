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
CUDA_VISIBLE_DEVICES=0,1 python3 evaluate.py  /dataset $SCRATCH/checkpoints/barlow/checkpoint.pth --weights finetune  --epochs 120 --lr-backbone 0.002 --lr-classifier 0.5 --weight-decay 0 --checkpoint-dir $SCRATCH/checkpoints/barlow/finetune --batch-size 12"
