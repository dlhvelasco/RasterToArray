#!/bin/bash 

#SBATCH --partition=batch
#SBATCH --job-name=GetFeatParallel
#SBATCH --output=GetFeatParallel.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --nodelist=tux-13
#SBATCH --requeue

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR

echo "working directory = "$SLURM_SUBMIT_DIR

# prepare working environment 
module load anaconda3

# activate your python environment
source activate thesis36
cd /home/dwight.velasco/scratch1/THESIS/RasterToArray/

# execute your application
srun python3 GetFeaturesParallel.py

source deactivate
