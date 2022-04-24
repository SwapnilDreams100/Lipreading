#!/bin/bash

#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mem=96GB
#SBATCH --job-name=archive-data

module purge
module load openmpi/intel/4.1.1

cd $SCRATCH/dl/MP2/Lipreading/datasets/
tar -cvzf $ARCHIVE/visual_data.tar visual_data
cp -r visual_data /state/partition1/ps45/

cd /state/partition1/ps45
mksquashfs visual_data visual_data.sqf -keep-as-directory
