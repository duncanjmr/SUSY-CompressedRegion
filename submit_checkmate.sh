#!/bin/sh

#SBATCH --time=12:00:00
#SBATCH --job-name=checkmate
#SBATCH --partition=cpu_gce
#SBATCH --ntasks=145
#SBATCH --qos=regular
#SBATCH --error="slurm_cm.err"

#This script is adapted from the example from https://rcc.uchicago.edu/docs/tutorials/kicp-tutorials/running-jobs.html.
#module load parallel

# the --exclusive to srun makes srun use distinct CPUs for each job step
# -N1 -n1 allocates a single core to each task
srun="srun --exclusive -N1 -n1"
for i in {0..144}
do 
	$srun python run_cluster_checkmate.py $i &
done
wait
#$srun python run_local_checkmate.py
#$srun python run_cluster_checkmate.py $SLURM_ARRAY_TASK_ID
