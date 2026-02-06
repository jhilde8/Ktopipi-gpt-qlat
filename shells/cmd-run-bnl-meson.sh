#!/bin/bash
#SBATCH --job-name=pipi_dc
#SBATCH --output=psrc_pipi_dc-%j.out
#SBATCH --error=psrc_pipi_dc-%j.err
#SBATCH --time=6:00:00
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=64
#SBATCH --partition=lqcd
#SBATCH --account=class-c-2pipsrc
#SBATCH --mem=0
#SBATCH --no-requeue

source /sdcc/u/jhildebra/Qlattice/build/setenv.sh

export OMP_NUM_THREADS=64
export q_verbose=2

rm -r /hpcgpfs01/work/lqcd/staging/RBC/qcddata/MDWF/2+1f/48nt96/IWASAKI/b2.13/ls24b+c2/M1.8/ms0.0362/mu0.00078/jhildebra/locks/*

# mpiexec -n 2 --oversubscribe --bind-to none python3 -m mpi4py /home/frank/Qlattice-cc/examples-py-gpt/gpt-qlat-data-gen-pipi-qed.py --mpi 1.1.1.2 --mpi 1.1.2 >log.full.txt 2>&1
# mpiexec -n 4 --oversubscribe --bind-to none python3 -m mpi4py /home/frank/Qlattice-cc/examples-py-gpt/gpt-qlat-data-gen-pipi-qed.py --mpi 1.1.2.2 --mpi 1.2.2 --job_tag_list "48I" --no-inversion >log.full.txt 2>&1

srun -n 4 python3 -m mpi4py /sdcc/u/jhildebra/Ktopipi-gpt-qlat/scripts/gpt-qlat-pipi-dc.py --mpi 1.1.2.2 --mpi 1.2.2 --job_tag_list "48I" --no-inversion > log.${SLURM_JOB_ID}.txt 2>&1


