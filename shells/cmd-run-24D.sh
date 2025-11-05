
export OMP_NUM_THREADS=32
export q_verbose=2

# mpiexec -n 2 --oversubscribe --bind-to none python3 -m mpi4py /home/frank/Qlattice-cc/examples-py-gpt/gpt-qlat-data-gen-pipi-qed.py --mpi 1.1.1.2 --mpi 1.1.2 >log.full.txt 2>&1
# mpiexec -n 4 --oversubscribe --bind-to none python3 -m mpi4py /home/frank/Qlattice-cc/examples-py-gpt/gpt-qlat-data-gen-pipi-qed.py --mpi 1.1.2.2 --mpi 1.2.2 --job_tag_list "48I" --no-inversion >log.full.txt 2>&1

time nix-shell ~/Qlattice/nixpkgs/shell.nix --run 'mpiexec -n 1 --oversubscribe --bind-to none python3 -m mpi4py /home/jhildebrand28/ktopipi/scripts/gpt-qlat-meson-wsrc-corr.py --mpi 1.1.1.1 --mpi 1.1.1 --job_tag_list "48I" --no-inversion' >log.full.txt 2>&1
