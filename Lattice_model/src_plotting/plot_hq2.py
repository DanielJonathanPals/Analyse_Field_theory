import numpy as np
import matplotlib.pyplot as plt
from create_anim import *
from Interface_dynamics import *
import os

file_name = 'eps_-2p95_f_res_2p85_rho_v_0p05_dmu_0p0_xsize_256_ysize_512'
q = 40

run_ids_str = next(os.walk(f"/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Lattice_model/Data/{file_name}"))[1]
run_ids = [int(run_id) for run_id in run_ids_str]
size, length_scale, epsilon, dmu, z_I, z_B, f_res, rho_v, t_max, save_interval = read_parameters(file_name, run_ids[0])
L = size[1]
t = np.arange(0, t_max + save_interval, save_interval)
q_arr = np.arange(0, 2*np.pi, 2*np.pi / L)

n = noise_amp_squared(epsilon, rho_v, f_res, k_IB_model1, dk_IB_du_model1, dmu)
et = eta(epsilon, rho_v, f_res, k_IB_model1, dk_IB_du_model1, dmu)
sig = sigma(epsilon, rho_v, f_res, k_IB_model1, dmu)
I = I_active(epsilon, rho_v, f_res, k_IB_model1, dk_IB_du_model1, dmu)
theory = -et*n/(2*(epsilon*sig + (1 - np.exp(dmu)) * I))

def get_hq2_data(file_name):
    file = open(f"/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Lattice_model/Data/{file_name}/expectation_hq2.txt", "r")
    data = file.readlines()
    file.close()
    data = [line.split("\n")[0].split("\t") for line in data]
    data = [[float(element) for element in line] for line in data]
    return np.array(data)

hq2 = get_hq2_data(file_name)

fig = plt.figure()
plt.plot(q_arr[1:] , hq2[-1,1:], label='Simulation', color = "black")
plt.plot(q_arr[10:], theory/q_arr[10:]**2, color='red', label='Theory')
plt.legend()
plt.grid()
plt.xlabel(r'$q$')
plt.ylabel(r'$|h_q|^2$')

fig2 = plt.figure()
t5 = t_max // 5
times = np.arange(t5, t_max + t5, t5)
t5_steps = np.round(t5 / save_interval).astype(int)
for i, time in enumerate(times):
    plt.loglog(q_arr[1:] , hq2[t5_steps * (i + 1),1:], label=f'Simulation, t={time}', color = "black", alpha = 0.2 * (i + 1))
plt.loglog(q_arr[1:], theory/q_arr[1:]**2, color='red', label='Theory')
plt.legend()
plt.grid()
plt.xlabel(r'$q$')
plt.ylabel(r'$|h_q|^2$')

fig3 = plt.figure()
plt.plot(t, hq2[:,q], label='Simulation', color = "black")
plt.plot(t, theory/(2*np.pi / L *q)**2 * np.ones(len(t)), color='red', label='Theory (equilibrium value)')
plt.grid()
plt.xlabel('t')
plt.ylabel(r'$|h_q|^2$')
plt.legend()
plt.title(f'$q = {q}$')

plt.show()