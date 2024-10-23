import numpy as np
import matplotlib.pyplot as plt
from create_anim import *
from Interface_dynamics import *
from Generate_theory_data import *
import matplotlib as mpl
import os

name = "epsilon_-2p5_rho_v_0p05_dmu_-1p0"



def Plot_fluctuation_autocorr(file_name):
    path = f"/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Lattice_model/Data/{file_name}"
    size, epsilon, dmu, z_I, z_B, f_res, rho_v, t_max, save_interval = read_parameters(file_name)
    L = size[1]
    times = np.arange(save_interval, t_max + save_interval, save_interval)

    # here we implicitly arrsume that evaluating the interface at times that differ by len(times) // 5, leads to uncorrelated fluctuations and that the fluctuations have equilibrated after time t_max // 2
    idx_arr = np.arange(0, len(times) )[len(times) // 2-1:]

    simulation_file_names = next(os.walk(path))[-1]
    simulation_file_names = [name for name in simulation_file_names if "simulation_No" in name]

    run_ids_str = next(os.walk(path))[-1]
    run_ids_str.remove("lattice_params.txt")
    run_ids = [int(run_id.split("No_")[-1]) for run_id in run_ids_str]

    all_fluctuations = np.zeros((len(run_ids), len(idx_arr)))
    
    
    for i, run_id in enumerate(run_ids):
        with h5py.File(f"{path}/simulation_No_{run_id}", "r") as f:
            for j, idx in enumerate(idx_arr):
                fluct_j = np.sqrt(np.sum((np.array(f[f"Fourier_Data/fourier_upper_real_{idx}"])[1:L//2]) ** 2 + (np.array(f[f"Fourier_Data/fourier_upper_imag_{idx}"])[1:L//2]) ** 2))
                all_fluctuations[i,j] = fluct_j
    
    all_fluctuations = all_fluctuations - np.mean(all_fluctuations, axis = 0)
    autocorr_arr = all_fluctuations * np.stack([all_fluctuations[:,0] for i in idx_arr], axis=1)

    plt.figure()
    plt.scatter(times[idx_arr] - times[idx_arr[0]], np.mean(autocorr_arr, axis=0))
    plt.errorbar(times[idx_arr] - times[idx_arr[0]], np.mean(autocorr_arr, axis=0), yerr = np.std(autocorr_arr, axis=0)/np.sqrt(len(run_ids)))
    plt.grid()
    plt.savefig("Fuctuation_autocorr.pdf")



Plot_fluctuation_autocorr(name)