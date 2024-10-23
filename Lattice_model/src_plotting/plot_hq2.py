import numpy as np
import matplotlib.pyplot as plt
from create_anim import *
from Interface_dynamics import *
import os

file_name = 'epsilon_-2p95_rho_v_0p05_dmu_1p0'
q = 10
order = 10
k_IB = lambda u, f_res, dmu: k_IB_model1(u, f_res, dmu)
f_res_init_guess = 3.

path = f"/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Lattice_model/Data/{file_name}"

files_in_path = next(os.walk(path))[-1]
files_in_path.remove("lattice_params.txt")
run_ids = [int(run_id.split("No_")[-1]) for run_id in files_in_path]
size, epsilon, dmu, z_I, z_B, f_res, rho_v, t_max, save_interval = read_parameters(file_name)
L = size[1]
t = np.arange(0, t_max + save_interval, save_interval)
q_arr = np.arange(0, 2*np.pi, 2*np.pi / L)

eta, sigma, chi2,_ = get_eta_sigma_chi2(epsilon, rho_v, k_IB, dmu, f_res_init_guess, order)
eta = eta[-1]
sigma = sigma[-1]
chi2 = chi2[-1]

theory = chi2/(2*sigma*eta) 


def get_hq2_average(fixed_q_data):
    l = len(fixed_q_data)
    data = fixed_q_data[l//2:]
    return np.mean(data), np.std(data)/np.sqrt(len(data))

with h5py.File(f"{path}/expectation_hq2/expectation_hq2", "r") as f:
    
    fig = plt.figure()
    plt.plot(q_arr[1:L//2] / np.pi  , np.array(f[f"Fixed_time/{len(t)}"])[1:L//2], label='Simulation', color = "black")
    plt.plot(q_arr[1:L//2] / np.pi , theory/q_arr[1:L//2]**2, color='red', label='Theory')
    plt.legend()
    plt.grid()
    plt.xlabel(r'$q / \pi$')
    plt.ylabel(r'$|h_q|^2$')
    plt.savefig(f"expectation_hq2.pdf")

    fig2 = plt.figure()
    t5 = t_max // 5
    times = np.arange(t5, t_max + t5, t5)
    t5_steps = np.round(t5 / save_interval).astype(int)
    for i, time in enumerate(times):
        plt.loglog(q_arr[1:L//2] / np.pi , np.array(f[f"Fixed_time/{t5_steps * (i + 1)}"])[1:L//2], label=f'Simulation, t={time}', color = "black", alpha = 0.2 * (i + 1))
    plt.loglog(q_arr[1:L//2] / np.pi , theory/q_arr[1:L//2]**2, color='red', label='Theory')
    plt.legend()
    plt.grid()
    plt.xlabel(r'$q / \pi$')
    plt.ylabel(r'$|h_q|^2$')
    plt.savefig(f"expectation_hq2_loglog.pdf")

    fig3 = plt.figure()
    plt.plot(t, np.array(f[f"Fixed_q/{q}"]), label='Simulation', color = "black")
    plt.plot(t, theory/(2*np.pi / L *(q))**2 * np.ones(len(t)), color='red', label='Theory (equilibrium value)')
    plt.grid()
    plt.xlabel('t')
    plt.ylabel(r'$|h_q|^2$')
    plt.legend()
    plt.title(r'$q = \frac{2 \pi}{L} \cdot$' + str(q))
    plt.savefig(f"expectation_hq2_time.pdf")

    x = np.arange(1, np.round(L // 2), 1)
    y = np.array([get_hq2_average(np.array(f[f"Fixed_q/{int(q)}"]))[0]*(2*np.pi / L *(q))**2 for q in x])
    z = np.array([get_hq2_average(np.array(f[f"Fixed_q/{int(q)}"]))[1]*(2*np.pi / L *(q))**2 for q in x])
    fig4 = plt.figure()
    ax = plt.axes()
    ax.scatter(x, y, color='black')
    ax.errorbar(x, y, z, fmt='o', color='black')

    val, cov = np.polyfit(x, y, 1, w = 1 / z, cov = True)
    ax.plot(x, val[0]*x + val[1], color='red')
    measured = val[1]
    error = np.sqrt(cov[1,1])
    print(f"The theoretical value is {theory}")
    print(f"The measured value is {measured} +- {error}")
    print(f"The resulting factor is {measured/theory} +- {error/theory}")
    print(f"noise_amp_2 = {chi2/(eta**2)}")

    plt.savefig(f"expectation_hq2_scaling.pdf")