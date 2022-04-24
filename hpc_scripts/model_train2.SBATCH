#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=5:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:v100:1
#SBATCH --output=slurm2-model-train-2.out 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ps4534,djs9437,sp6646
#SBATCH --job-name=Visual-Short-Model-Train

module purge
module load openmpi/intel/4.1.1

singularity exec --nv \
--overlay /scratch/djs9437/env2/overlay-25GB-500K.ext3 \
--overlay $SCRATCH/lrw.sqf:ro \
/scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
/bin/bash -c "source /ext3/env.sh; conda activate env2; cd /scratch/ps4534/dl/MP2/Lipreading/; CUDA_VISIBLE_DEVICES=0 python main_devansh.py --config-path configs/lrw_resnet18_mstcn.json --annonation-direc datasets/data_files --data-dir datasets/visual_1_channel/ "
