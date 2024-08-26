import numpy as np
import matplotlib.pyplot as plt
import os
from create_anim import *
from Interface_dynamics import *

file_name = 'eps_-2p95_f_res_2p85_rho_v_0p05_dmu_0p0_xsize_256_ysize_512'
trans_time = 2000.0

run_ids_str = next(os.walk(f"/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Lattice_model/Data/{file_name}"))[1]
run_ids = [int(run_id) for run_id in run_ids_str]
numb_of_simulations = len(run_ids)
run_id = run_ids[0]
size, length_scale, epsilon, dmu, z_I, z_B, f_res, rho_v, t_max, save_interval = read_parameters(file_name, run_id)
L = size[1]
t = np.arange(0, t_max + save_interval, save_interval)
time_steps = len(t)
noise_amp_2 = noise_amp_squared(epsilon, rho_v, f_res, k_IB_model1, dk_IB_du_model1, dmu)
trans_time_steps = np.round(trans_time / save_interval).astype(int)
trans_time = trans_time_steps * save_interval


def read_interface_pos(file_name, run_id):
    file = open(f"/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Lattice_model/Data/{file_name}/{run_id}/interface_pos_upper_boundary.txt", "r")
    data = file.readlines()
    file.close()
    data1 = [line.split("\n")[0].split("\t") for line in data]
    data1 = [[float(element) for element in line] for line in data1]

    file = open(f"/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Lattice_model/Data/{file_name}/{run_id}/interface_pos_lower_boundary.txt", "r")
    data = file.readlines()
    file.close()
    data2 = [line.split("\n")[0].split("\t") for line in data]
    data2 = [[float(element) for element in line] for line in data2]
    return np.reshape(np.array(data1), time_steps), np.reshape(np.array(data2), time_steps)

interface_pos = np.zeros((2 * numb_of_simulations, time_steps))

for i, run_id in enumerate(run_ids):
    int_pos1, int_pos2 = read_interface_pos(file_name, run_id)
    interface_pos[2 * i,:] = int_pos1
    interface_pos[2 * i + 1,:] = int_pos2

interface_pos = (interface_pos[:,trans_time_steps:].transpose() - interface_pos[:,trans_time_steps]).transpose()
t = t[trans_time_steps:]

for i in range(2 * numb_of_simulations):
    plt.plot(t, interface_pos[i,:] , color='black', lw=0.5)

plt.plot(t, np.mean(interface_pos, axis=0), color='blue', lw=1.5)
plt.plot(t, np.mean(interface_pos, axis=0) + np.std(interface_pos, axis=0), color='blue', lw=1., ls= '--')
plt.plot(t, np.mean(interface_pos, axis=0) - np.std(interface_pos, axis=0), color='blue', lw=1., ls= '--')
plt.fill_between(t, np.mean(interface_pos, axis=0) - np.std(interface_pos, axis=0), np.mean(interface_pos, axis=0) + np.std(interface_pos, axis=0), color='blue', alpha=0.2)

plt.plot(t, np.sqrt(noise_amp_2 * (t - trans_time) / L), color='red', lw=1.)
plt.plot(t, -np.sqrt(noise_amp_2 * (t - trans_time) / L), color='red', lw=1.)
plt.xlabel('t')
plt.ylabel('Interface position')


plt.grid()
plt.show()


