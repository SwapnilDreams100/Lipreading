#!/bin/bash

#SBATCH --nodes=8
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=Pre-process-audio-waveforms

module purge
module load openmpi/intel/4.1.1

singularity exec \
--overlay $SCRATCH/dl/overlay-25GB-500K.ext3 \
--overlay $SCRATCH/dl/MP2/Lipreading/lrw/lrw.sqf:ro \
/scratch/work/public/singularity/ubuntu-21.04.sif \
/bin/bash -c "source /ext3/env.sh; conda activate lr; cd $SCRATCH/dl/MP2/Lipreading/preprocessing; python extract_audio_from_video.py --video-direc /lrw --save-direc $SCRATCH/dl/MP2/Lipreading/datasets/audio_data/"
