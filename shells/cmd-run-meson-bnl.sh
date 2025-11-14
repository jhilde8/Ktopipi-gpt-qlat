#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=test-%j.out
#SBATCH --error=test-%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=lqcd
#SBATCH --account=jhildebra
#SBATCH --mem=64G

module purge
module load python/3.12-anaconda-2024.10
module load gcc/15.2.0
module load openmpi/5.0.6

export OMP_NUM_THREADS=32
export q_verbose=2



# mpiexec -n 2 --oversubscribe --bind-to none python3 -m mpi4py /home/frank/Qlattice-cc/examples-py-gpt/gpt-qlat-data-gen-pipi-qed.py --mpi 1.1.1.2 --mpi 1.1.2 >log.full.txt 2>&1
# mpiexec -n 4 --oversubscribe --bind-to none python3 -m mpi4py /home/frank/Qlattice-cc/examples-py-gpt/gpt-qlat-data-gen-pipi-qed.py --mpi 1.1.2.2 --mpi 1.2.2 --job_tag_list "48I" --no-inversion >log.full.txt 2>&1

srun python3 -m mpi4py -n 1 /sdcc/u/jhildebra/Ktopipi-gpt-qlat/scripts/gpt-qlat-meson-corr.py --mpi 1.1.1.1 --mpi 1.1.1 --job_tag_list "48I" --no-inversion > log.full.txt 2>&1


