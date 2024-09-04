import numpy as np
import matplotlib.pyplot as plt
from create_anim import *
from Interface_dynamics import *
import os

file_name = 'epsilon_-2p95_rho_v_0p05_dmu_0p0_old'
q = 30

path = f"/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Lattice_model/Data/{file_name}"

files_in_path = next(os.walk(path))[-1]
files_in_path.remove("lattice_params.txt")
run_ids = [int(run_id.split("No_")[-1]) for run_id in files_in_path]
size, epsilon, dmu, z_I, z_B, f_res, rho_v, t_max, save_interval = read_parameters(file_name)
L = size[1]
t = np.arange(0, t_max + save_interval, save_interval)
q_arr = np.arange(0, 2*np.pi, 2*np.pi / L)

# Since it takes some time to compute nosie_amp_2 it will be saved in a txt file
if not os.path.isdir(f"{path}/theory_vals"):
    os.mkdir(f"{path}/theory_vals")
if os.path.isfile(f"{path}/theory_vals/noise_amp_2.txt"):
    file = open(f"{path}/theory_vals/noise_amp_2.txt", "r")
    n = file.readlines()
    file.close()
    n = float(n[0])
else:
    n = noise_amp_squared(epsilon, rho_v, f_res, k_IB, dk_IB_du, dmu)
    with open(f"{path}/theory_vals/noise_amp_2.txt", "w") as file:
        file.write(str(n))
if os.path.isfile(f"{path}/theory_vals/eta.txt"):
    file = open(f"{path}/theory_vals/eta.txt", "r")
    et = file.readlines()
    file.close()
    et = float(et[0])
else:
    et = eta(epsilon, rho_v, f_res, k_IB, dk_IB_du, dmu)
    with open(f"{path}/theory_vals/eta.txt", "w") as file:
        file.write(str(et))
if os.path.isfile(f"{path}/theory_vals/sigma.txt"):
    file = open(f"{path}/theory_vals/sigma.txt", "r")
    sig = file.readlines()
    file.close()
    sig = float(sig[0])
else:
    sig = sigma(epsilon, rho_v, f_res, k_IB, dmu)
    with open(f"{path}/theory_vals/sigma.txt", "w") as file:
        file.write(str(sig))
if os.path.isfile(f"{path}/theory_vals/I_active.txt"):
    file = open(f"{path}/theory_vals/I_active.txt", "r")
    I = file.readlines()
    file.close()
    I = float(I[0])
else:
    I = I_active(epsilon, rho_v, f_res, k_IB, dk_IB_du, dmu)
    with open(f"{path}/theory_vals/I_active.txt", "w") as file:
        file.write(str(I))

theory = -et*n/(2*(epsilon*sig + (1 - np.exp(dmu)) * I)) * 1.3

with h5py.File(f"{path}/expectation_hq2/expectation_hq2", "r") as f:
    
    fig = plt.figure()
    plt.plot(q_arr[1:L//2] / np.pi  , np.array(f[f"Fixed_time/{len(t)}"])[1:L//2], label='Simulation', color = "black")
    plt.plot(q_arr[10:L//2] / np.pi , theory/q_arr[10:L//2]**2, color='red', label='Theory')
    plt.legend()
    plt.grid()
    plt.xlabel(r'$q / \pi$')
    plt.ylabel(r'$|h_q|^2$')
    plt.savefig("plot3.pdf")

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
    plt.savefig("plot2.pdf")

    fig3 = plt.figure()
    plt.plot(t, np.array(f[f"Fixed_q/{q}"]), label='Simulation', color = "black")
    plt.plot(t, theory/(2*np.pi / L *(q))**2 * np.ones(len(t)), color='red', label='Theory (equilibrium value)')
    plt.grid()
    plt.xlabel('t')
    plt.ylabel(r'$|h_q|^2$')
    plt.legend()
    plt.title(r'$q = \frac{2 \pi}{L} \cdot$' + str(q))

plt.savefig("plot1.pdf")