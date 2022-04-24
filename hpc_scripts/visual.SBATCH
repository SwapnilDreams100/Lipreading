#!/bin/bash

#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=18:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=Pre-process-Mouth-ROIs

module purge
module load openmpi/intel/4.1.1

singularity exec \
--overlay $SCRATCH/dl/overlay-25GB-500K.ext3 \
/scratch/work/public/singularity/ubuntu-21.04.sif \
/bin/bash -c "source /ext3/env.sh; conda activate lr; cd $SCRATCH/dl/MP2/Lipreading/preprocessing; python crop_mouth_from_video_red.py --video-direc $SCRATCH/dl/MP2/Lipreading/datasets/data_files --landmark-direc $SCRATCH/dl/MP2/Lipreading/datasets/landmarks --save-direc $SCRATCH/dl/MP2/Lipreading/datasets/visual_3_channel/"
