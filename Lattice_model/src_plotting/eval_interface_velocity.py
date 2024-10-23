import numpy as np
import matplotlib.pyplot as plt
import os
from create_anim import *
from Interface_dynamics import *

file_name = 'epsilon_-2p95_rho_v_0p05_dmu_1p0_long'
trans_time = 0.
order = 5
k_IB = lambda u, f_res, dmu: k_IB_model1(u, f_res, dmu)
f_res_init_guess = 4.

path = f"/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Lattice_model/Data/{file_name}"

run_ids_str = next(os.walk(path))[-1]
run_ids_str.remove("lattice_params.txt")
run_ids = [int(run_id.split("No_")[-1]) for run_id in run_ids_str]
numb_of_simulations = len(run_ids)
size, epsilon, dmu, z_I, z_B, f_res, rho_v, t_max, save_interval = read_parameters(file_name)
L = size[1]
t = np.arange(0, t_max + save_interval, save_interval)
time_steps = len(t)

eta, sigma, chi2, _ = get_eta_sigma_chi2(epsilon, rho_v, k_IB, dmu, f_res_init_guess, order)
eta = eta[-1]
sigma = sigma[-1]
chi2 = chi2[-1]

trans_time_steps = np.round(trans_time / save_interval).astype(int)
trans_time = trans_time_steps * save_interval


def read_interface_pos(run_id):
    data = []
    with h5py.File(f"{path}/simulation_No_{run_id}", "r") as f:
        data = np.array(f["Interface_position/positions"])
    return np.reshape(data[:,0], time_steps), -np.reshape(data[:,1], time_steps)

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

plt.plot(t, np.sqrt(chi2/(eta**2) * (t - trans_time) / L), color='red', lw=1., ls= '--')
plt.plot(t, -np.sqrt(chi2/(eta**2) * (t - trans_time) / L), color='red', lw=1., ls= '--')
plt.plot(t, np.zeros(len(t)), color='red', lw=1.)
plt.xlabel('t')
plt.ylabel('Interface position')
plt.title(f"Interface velocity, epsilon = {epsilon}, rho_v = {rho_v}, dmu = {dmu}")

plt.grid()

n_measured = (np.std(interface_pos, axis=0)[10:])**2 / (t[10:]- trans_time)
n_meas = np.mean(n_measured)
print(chi2/(eta**2))
print(n_meas * L)
print(n_meas * L / (chi2/(eta**2)))

plt.savefig("interface_velocity.pdf")

plt.figure()
plt.plot(t[10:]- trans_time, n_measured, color='black')
plt.savefig("noise_amp_squared_from_interface_velocity.pdf")
