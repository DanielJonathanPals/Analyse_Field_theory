import numpy as np
import matplotlib.pyplot as plt
from create_anim import *
from Interface_dynamics import *
from Generate_theory_data import *
import matplotlib as mpl
import os

file_names = ['epsilon_-2p95_rho_v_0p05_dmu_-1p0_long_field',
              'epsilon_-2p95_rho_v_0p05_dmu_-0p8_long_field',
              'epsilon_-2p95_rho_v_0p05_dmu_-0p6_long_field',
              'epsilon_-2p95_rho_v_0p05_dmu_-0p4_long_field',
              'epsilon_-2p95_rho_v_0p05_dmu_-0p2_long_field',
              'epsilon_-2p95_rho_v_0p05_dmu_0p0_long_field',
              'epsilon_-2p95_rho_v_0p05_dmu_0p2_long_field',
              'epsilon_-2p95_rho_v_0p05_dmu_0p4_long_field',
              'epsilon_-2p95_rho_v_0p05_dmu_0p6_long_field',
              'epsilon_-2p95_rho_v_0p05_dmu_0p8_long_field',
              'epsilon_-2p95_rho_v_0p05_dmu_1p0_long_field']

"""
# measured interface fluctuations
def get_hq2_average(fixed_q_data):
    l = len(fixed_q_data)
    data = fixed_q_data[l//2:]
    return np.mean(data), np.std(data)/np.sqrt(len(data))

def measure_interface_fluctuations(file_name):
    path = f"/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Lattice_model/Data/{file_name}"
    size, epsilon, dmu, z_I, z_B, f_res, rho_v, t_max, save_interval = read_parameters(file_name)
    L = size[1]

    t100 = t_max // 200
    times = np.arange(t100, t_max + t100, t100)
    t100_steps = np.round(t100 / save_interval).astype(int)

    measured_arr = np.zeros(len(times))
    
    with h5py.File(f"{path}/expectation_hq2/expectation_hq2", "r") as f:
        for i, time in enumerate(times):
            measured_arr[i] = np.sqrt(np.sum(np.array(f[f"Fixed_time/{t100_steps * (i + 1)}"])[1:L//2]))

    return np.mean(measured_arr), np.std(measured_arr) / np.sqrt(len(measured_arr))

"""

# evaluates the interface fluctuations of each simulation at the times corresponding to idx_arr and returns the mean and std of the fluctuations
def measure_interface_fluctuations(file_name):
    path = f"/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Lattice_model/Data/{file_name}"
    size, epsilon, dmu, z_I, z_B, f_res, rho_v, t_max, save_interval = read_parameters(file_name)
    L = size[1]
    times = np.arange(save_interval, t_max + save_interval, save_interval)

    # here we implicitly arrsume that evaluating the interface at times that differ by 50*save_interval, leads to uncorrelated fluctuations and that the fluctuations have equilibrated after time t_max // 2
    idx_arr = np.arange(1, len(times) + 1)[len(times) // 2-1::50]

    simulation_file_names = next(os.walk(path))[-1]
    simulation_file_names = [name for name in simulation_file_names if "simulation_No" in name]

    run_ids_str = next(os.walk(path))[-1]
    run_ids_str.remove("lattice_params.txt")
    run_ids = [int(run_id.split("No_")[-1]) for run_id in run_ids_str]

    delta_h_arr = np.zeros(len(run_ids) * 2 * len(idx_arr))

    for i, run_id in enumerate(run_ids):
        with h5py.File(f"{path}/simulation_No_{run_id}", "r") as f:
            for j, idx in enumerate(idx_arr):
                delta_h_arr[2*len(idx_arr)*i + j] = np.sqrt(np.sum((np.array(f[f"Fourier_Data/fourier_upper_real_{idx}"])[1:L]) ** 2 + (np.array(f[f"Fourier_Data/fourier_upper_imag_{idx}"])[1:L]) ** 2))
                delta_h_arr[2*len(idx_arr)*i + len(idx_arr) + j] = np.sqrt(np.sum((np.array(f[f"Fourier_Data/fourier_lower_real_{idx}"])[1:L]) ** 2 + (np.array(f[f"Fourier_Data/fourier_lower_imag_{idx}"])[1:L]) ** 2))
    
    return np.mean(delta_h_arr), np.std(delta_h_arr) / np.sqrt(len(delta_h_arr))




measured_arr = np.zeros(len(file_names))
measured_arr_err = np.zeros(len(file_names))

for i, file_name in enumerate(file_names):
    print(file_name)
    measured_arr[i], measured_arr_err[i] = measure_interface_fluctuations(file_name)

with open('/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Lattice_model/src_plotting/measured_interface_fluctuations.npy', 'wb') as f:
    np.save(f, measured_arr)

with open('/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Lattice_model/src_plotting/measured_interface_fluctuations_err.npy', 'wb') as f:
    np.save(f, measured_arr_err)


# Measure the noise amplitude squared and interface velocity
def measure_noise_amp_squared(file_name):
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
    trans_time_steps = np.round(trans_time / save_interval).astype(int)
    trans_time = trans_time_steps * save_interval

    interface_pos = np.zeros((2 * numb_of_simulations, time_steps))

    for i, run_id in enumerate(run_ids):
        data = []
        with h5py.File(f"{path}/simulation_No_{run_id}", "r") as f:
            data = np.array(f["Interface_position/positions"])
        int_pos1, int_pos2 = np.reshape(data[:,0], time_steps), -np.reshape(data[:,1], time_steps)
        interface_pos[2 * i,:] = int_pos1
        interface_pos[2 * i + 1,:] = int_pos2
    
    

    interface_pos = (interface_pos[:,trans_time_steps:].transpose() - interface_pos[:,trans_time_steps]).transpose()
    t = t[trans_time_steps:]
    n_measured = (np.std(interface_pos, axis=0)[10:])**2 / (t[10:]- trans_time)
    n_meas = np.mean(n_measured)

    
    velocity = np.mean(interface_pos[:,-1]) / (t_max - trans_time)
    velocity_std = np.std(interface_pos[:,-1]) / (t_max - trans_time) / np.sqrt(2 * numb_of_simulations)


    return n_meas * L, dmu, velocity, velocity_std

mu_arr = np.zeros(len(file_names))
noise_amp_squared_arr = np.zeros(len(file_names))
velocity_arr = np.zeros(len(file_names))
velocity_std_arr = np.zeros(len(file_names))

for i, file_name in enumerate(file_names):
    print(file_name)
    noise_amp_squared_arr[i], mu_arr[i], velocity_arr[i], velocity_std_arr[i] = measure_noise_amp_squared(file_name)

with open('/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Lattice_model/src_plotting/noise_amp_squared.npy', 'wb') as f:
    np.save(f, noise_amp_squared_arr)

with open('/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Lattice_model/src_plotting/dmu.npy', 'wb') as f:
    np.save(f, mu_arr)

with open('/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Lattice_model/src_plotting/velocity.npy', 'wb') as f:
    np.save(f, velocity_arr)

with open('/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Lattice_model/src_plotting/velocity_std.npy', 'wb') as f:
    np.save(f, velocity_std_arr)



