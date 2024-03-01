#!/bin/bash
#SBATCH --time=120:00:00
#SBATCH --ntasks=513
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2GB
#SBATCH -p wmglab
#SBATCH --job-name=pso_D+
#SBATCH --output=logs/log_%x-%j.out
#SBATCH --error=logs/log_%x-%j.error
#SBATCH --mail-type=END,FAIL

WD='/hpc/home/gy42/tms-network-modeling'
cd $WD

module purge
module load OpenMPI

mpirun -n 513 python run_pso.py Parameters_iwaves_column_v2_2013_31_120RMT_iclamp_1HzAFF_tminima_long_tinterval800 0
