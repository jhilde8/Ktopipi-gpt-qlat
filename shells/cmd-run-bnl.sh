#!/bin/bash
#SBATCH --job-name=psrc_meson
#SBATCH --output=psrc_meson-%j.out
#SBATCH --error=psrc_meson-%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=48
#SBATCH --partition=lqcd
#SBATCH --account=class-c-2pipsrc
#SBATCH --mem=0
#SBATCH --no-requeue

source /sdcc/u/jhildebra/Qlattice/build/setenv.sh

export OMP_NUM_THREADS=48
export q_verbose=2

rm -r /hpcgpfs01/work/lqcd/staging/RBC/qcddata/MDWF/2+1f/48nt96/IWASAKI/b2.13/ls24b+c2/M1.8/ms0.0362/mu0.00078/jhildebra/locks/*

# mpiexec -n 2 --oversubscribe --bind-to none python3 -m mpi4py /home/frank/Qlattice-cc/examples-py-gpt/gpt-qlat-data-gen-pipi-qed.py --mpi 1.1.1.2 --mpi 1.1.2 >log.full.txt 2>&1
# mpiexec -n 4 --oversubscribe --bind-to none python3 -m mpi4py /home/frank/Qlattice-cc/examples-py-gpt/gpt-qlat-data-gen-pipi-qed.py --mpi 1.1.2.2 --mpi 1.2.2 --job_tag_list "48I" --no-inversion >log.full.txt 2>&1

srun -n 8 python3 -m mpi4py /sdcc/u/jhildebra/Ktopipi-gpt-qlat/scripts/gpt-qlat-pipisc.py --mpi 1.1.2.4 --mpi 2.2.2 --job_tag_list "48I" --no-inversion > log.${SLURM_JOB_ID}.txt 2>&1


