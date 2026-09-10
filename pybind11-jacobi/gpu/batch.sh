#!/bin/bash
#SBATCH --job-name=jacobi_gpu_stable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
#SBATCH --account=ICT25_MHPC_0
#SBATCH --partition=boost_usr_prod

module purge
module load profile/base
module load nvhpc/24.5 hpcx-mpi/2.19 cuda/12.3

source /leonardo/home/userexternal/psingh01/miniconda3/etc/profile.d/conda.sh
conda activate new_env

# Disable CUDA-aware MPI at the UCX level to stop libucs segmentation faults
export UCX_TLS=self,sm,tcp
export UCX_MEMTYPE_CACHE=n

srun python3 main_gpu.py