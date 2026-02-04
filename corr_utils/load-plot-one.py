import qlat as q
import matplotlib.pyplot as plt
import numpy as np
import sys

#load the single config, using the file path as a command line input
ac_label = sys.argv[1]
traj = sys.argv[2]
filename = sys.argv[3]
dst = f"/home/jhildebrand28/ktopipi/results/48I/auto-contract-{ac_label}/traj-{traj}/{filename}.lat"
print(dst)

data_arr =  q.load_lat_data(dst).to_numpy()

print(f"data shape: {data_arr.shape}")

t_ext = data_arr.shape[-1]

t = np.linspace(1,t_ext,t_ext)
fig,ax = plt.subplots()
ax.plot(t, data_arr[1,:],marker='.', ls='')
ax.set_xlabel('t/a')
ax.set_ylabel('C(t)')
ax.set_title(f'{filename}')
ax.set_yscale("log")
fig.savefig("/home/jhildebrand28/ktopipi/figures/single_conf_plot.pdf", bbox_inches='tight')

np.save(f"/home/jhildebrand28/ktopipi/data/{filename}/traj-{traj}.npy", data_arr)

