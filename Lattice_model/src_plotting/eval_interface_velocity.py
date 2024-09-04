import numpy as np
import matplotlib.pyplot as plt
import os
from create_anim import *
from Interface_dynamics import *

file_name = 'epsilon_-2p95_rho_v_0p3_dmu_0p0'
trans_time = 2000.0

path = f"/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Lattice_model/Data/{file_name}"

run_ids_str = next(os.walk(path))[-1]
run_ids_str.remove("lattice_params.txt")
run_ids = [int(run_id.split("No_")[-1]) for run_id in run_ids_str]
numb_of_simulations = len(run_ids)
size, epsilon, dmu, z_I, z_B, f_res, rho_v, t_max, save_interval = read_parameters(file_name)
L = size[1]
t = np.arange(0, t_max + save_interval, save_interval)
time_steps = len(t)

# Since it takes some time to compute nosie_amp_2 it will be saved in a txt file
if not os.path.isdir(f"{path}/theory_vals"):
    os.mkdir(f"{path}/theory_vals")
if os.path.isfile(f"{path}/theory_vals/noise_amp_2.txt"):
    file = open(f"{path}/theory_vals/noise_amp_2.txt", "r")
    noise_amp_2 = file.readlines()
    file.close()
    noise_amp_2 = float(noise_amp_2[0])
else:
    noise_amp_2 = noise_amp_squared(epsilon, rho_v, f_res, k_IB, dk_IB_du, dmu)
    with open(f"{path}/theory_vals/noise_amp_2.txt", "w") as file:
        file.write(str(noise_amp_2))

trans_time_steps = np.round(trans_time / save_interval).astype(int)
trans_time = trans_time_steps * save_interval


def read_interface_pos(run_id):
    data = []
    with h5py.File(f"{path}/simulation_No_{run_id}", "r") as f:
        data = np.array(f["Interface_position/positions"])
    return np.reshape(data[:,0], time_steps), np.reshape(data[:,1], time_steps)

interface_pos = np.zeros((2 * numb_of_simulations, time_steps))

for i, run_id in enumerate(run_ids):
    int_pos1, int_pos2 = read_interface_pos(run_id)
    interface_pos[2 * i,:] = int_pos1
    interface_pos[2 * i + 1,:] = int_pos2

interface_pos = (interface_pos[:,trans_time_steps:].transpose() - interface_pos[:,trans_time_steps]).transpose()
t = t[trans_time_steps:]

for i in range(2 * numb_of_simulations):
    plt.plot(t, interface_pos[i,:] , color='black', lw=0.5, alpha = 0.1)


plt.plot(t, np.mean(interface_pos, axis=0), color='blue', lw=1.5)
plt.plot(t, np.mean(interface_pos, axis=0) + np.std(interface_pos, axis=0), color='blue', lw=1., ls= '--')
plt.plot(t, np.mean(interface_pos, axis=0) - np.std(interface_pos, axis=0), color='blue', lw=1., ls= '--')
plt.fill_between(t, np.mean(interface_pos, axis=0) - np.std(interface_pos, axis=0), np.mean(interface_pos, axis=0) + np.std(interface_pos, axis=0), color='blue', alpha=0.2)

plt.plot(t, np.sqrt(noise_amp_2 * (t - trans_time) / L), color='red', lw=1., ls= '--')
plt.plot(t, -np.sqrt(noise_amp_2 * (t - trans_time) / L), color='red', lw=1., ls= '--')
plt.plot(t, np.zeros(len(t)), color='red', lw=1.)
plt.xlabel('t')
plt.ylabel('Interface position')


plt.grid()
plt.savefig("plot.pdf")