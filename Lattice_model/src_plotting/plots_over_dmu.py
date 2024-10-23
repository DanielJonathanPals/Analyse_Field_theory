import numpy as np
import matplotlib.pyplot as plt
from create_anim import *
from Interface_dynamics import *
from Generate_theory_data import *
import matplotlib as mpl
import os

file_names = ['epsilon_-2p95_rho_v_0p05_dmu_-1p0_FLEX',
              'epsilon_-2p95_rho_v_0p05_dmu_-0p8',
              'epsilon_-2p95_rho_v_0p05_dmu_-0p6',
              'epsilon_-2p95_rho_v_0p05_dmu_-0p4', 
              'epsilon_-2p95_rho_v_0p05_dmu_-0p2',
              'epsilon_-2p95_rho_v_0p05_dmu_0p0',
              'epsilon_-2p95_rho_v_0p05_dmu_0p2',
              'epsilon_-2p95_rho_v_0p05_dmu_0p4',
              'epsilon_-2p95_rho_v_0p05_dmu_0p6',
              'epsilon_-2p95_rho_v_0p05_dmu_0p8',
              'epsilon_-2p95_rho_v_0p05_dmu_1p0']

size, epsilon, dmu, z_I, z_B, f_res, rho_v, t_max, save_interval = read_parameters(file_names[0])
L = size[1]


with open('/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Lattice_model/src_plotting/chi_2.npy', 'rb') as f:
    chi2_arr = np.load(f)

with open('/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Lattice_model/src_plotting/sigma.npy', 'rb') as f:
    sigma_arr = np.load(f)

with open('/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Lattice_model/src_plotting/eta.npy', 'rb') as f:
    eta_arr = np.load(f)

with open('/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Lattice_model/src_plotting/noise_amp_squared.npy', 'rb') as f:
    noise_amp_squared_arr = np.load(f)

with open('/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Lattice_model/src_plotting/dmu.npy', 'rb') as f:
    mu_arr = np.load(f)

with open ('/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Lattice_model/src_plotting/f_res.npy', 'rb') as f:
    f_res_arr = np.load(f)

with open('/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Lattice_model/src_plotting/velocity.npy', 'rb') as f:
    velocity_arr = np.load(f)

with open('/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Lattice_model/src_plotting/velocity_std.npy', 'rb') as f:
    velocity_std_arr = np.load(f)

with open ('/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Lattice_model/src_plotting/measured_interface_fluctuations.npy', 'rb') as f:
    measured_interface_fluctuations = np.load(f)

with open ('/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Lattice_model/src_plotting/measured_interface_fluctuations_err.npy', 'rb') as f:
    measured_interface_fluctuations_err = np.load(f)


eq_idx = np.argmin(np.abs(dmu_arr))
sigma_eq = sigma_arr[:,eq_idx]
chi2_eq = chi2_arr[:,eq_idx]
eta_eq = eta_arr[:,eq_idx]

eq_idx_meas = np.argmin(np.abs(mu_arr))
measured_interface_fluctuations_eq = measured_interface_fluctuations[eq_idx_meas]
measured_interface_fluctuations_err_eq = measured_interface_fluctuations_err[eq_idx_meas]

def noise_amp_squared(eta, chi2):
    return chi2/(eta**2)

def Delta_h(eta, sigma, chi2, L):
    if chi2 * L /(24 * sigma * eta) > 100:
        return 10
    elif chi2 * L /(24 * sigma * eta) < 0.:
        return 0.
    else:
        return np.sqrt(chi2 /(sigma * eta))

Delta_h_eq = Delta_h(eta_eq[-1], sigma_eq[-1], chi2_eq[-1], L)



plt.figure(figsize=(6,3))
for alph, i in enumerate([1,4,7,8,9]):
    plt.plot(dmu_arr, np.array([noise_amp_squared(eta_arr[i,j], chi2_arr[i,j]) for j in range(len(dmu_arr))]), label=f'order {i}', color = 'black', alpha = (alph+1)/(order+1), lw = 1)

plt.plot(dmu_arr, np.array([noise_amp_squared(eta_arr[order,j], chi2_arr[order,j]) for j in range(len(dmu_arr))]), label=f'order {order}', color = 'black', lw = 3)

plt.scatter(mu_arr, noise_amp_squared_arr, color='red', label='Simulation')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlim(-1.05,1.05)
plt.ylim(0,0.02)
plt.ylabel(r"$\chi^2$")
plt.xlabel(r"$\beta d \mu$")
plt.grid()
plt.title(r"Theory and Measurements of $\chi^2$")
plt.tight_layout()

plt.savefig("noise_amp_squared.pdf")


plt.figure(figsize=(6,3))

# Plot FLEX predition
tilde_epsilon_arr_paper = np.array([tilde_epsilon(rho_v, f_res_flex(rho_v, k_IB, dmu, epsilon, f_res), k_IB, dmu, epsilon) for dmu in dmu_arr])
Delta_h_arr_flex_from_paper = (np.sinh(epsilon / 4) / np.sinh(tilde_epsilon_arr_paper / 4) - 1) * 100

tilde_epsilon_arr_non_coex = np.array([tilde_epsilon_non_coex(rho_v, f_res_arr[i], k_IB, dmu, epsilon) for i,dmu in enumerate(dmu_arr)])
Delta_h_arr_flex_non_coex = (np.sinh(epsilon / 4) / np.sinh(tilde_epsilon_arr_non_coex / 4) - 1) * 100
#plt.plot(dmu_arr, Delta_h_arr_flex_from_paper, label='FLEX', color='blue')
plt.plot(dmu_arr, Delta_h_arr_flex_non_coex, label='FLEX', color='blue', ls='--')

idx_exept_eq = np.array([np.arange(0,eq_idx_meas), np.arange(eq_idx_meas+1,len(mu_arr))]).flatten()
for alph, i in enumerate([1,4,7,8,9]):
    plt.plot(dmu_arr, np.array([(Delta_h(eta_arr[i,j], sigma_arr[i,j], chi2_arr[i,j], L) - Delta_h_eq) / Delta_h_eq * 100 for j in range(len(dmu_arr))]), label=f'order {i}', color = 'black', alpha = (alph+1)/(order+1), lw = 1)

plt.plot(dmu_arr, np.array([(Delta_h(eta_arr[order,j], sigma_arr[order,j], chi2_arr[order,j], L) - Delta_h_eq) / Delta_h_eq * 100 for j in range(len(dmu_arr))]), label=f'order {order}', color = 'black', lw = 3)

#plt.scatter(mu_arr[idx_exept_eq], ((measured_interface_fluctuations - measured_interface_fluctuations_eq) / measured_interface_fluctuations_eq * 100)[idx_exept_eq], color='red', label='Simulation', zorder = 10)
plt.errorbar(mu_arr[idx_exept_eq], ((measured_interface_fluctuations - measured_interface_fluctuations_eq) / measured_interface_fluctuations_eq * 100)[idx_exept_eq], yerr=(np.sqrt((measured_interface_fluctuations_err * measured_interface_fluctuations_eq)**2 + (measured_interface_fluctuations * measured_interface_fluctuations_err_eq)**2) / (measured_interface_fluctuations_eq**2) * 100)[idx_exept_eq], color='red', zorder = 10, marker='o', ms=3, label="simulation", ls='')
plt.scatter([0],[0], color='red', zorder = 10, s=3)

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlim(-1.05,1.05)
plt.ylim(-3,6)
plt.grid()
plt.ylabel(r'$(\Delta h - \Delta h_{{eq}}) / \Delta h_{{eq}}(\%)$')
plt.xlabel(r'$\beta \Delta \mu$')
plt.tight_layout()

plt.savefig("Delta_h.pdf")


plt.figure(figsize=(6,3))
#plt.scatter(mu_arr, velocity_arr, color='red', label='Simulation')
plt.errorbar(mu_arr, velocity_arr, yerr=velocity_std_arr, color='blue',marker='o', ls='', label="simulation FLEX", ms=3)
plt.plot([-1.05,1.05], np.zeros(2), color='black', label='theory')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlim(-1.05,1.05)
plt.ylim(-6e-5,1.5e-5)
plt.grid()
plt.xlabel(r'$\beta \Delta \mu$')
plt.ylabel(r'$\bar v$ (lattice sites/s)')
plt.title("Average interface velocity")
plt.tight_layout()

plt.savefig("velocity.pdf")


plt.figure(figsize=(6,3))

for alph, i in enumerate([1,4,7,8,9]):
    plt.plot(dmu_arr,sigma_arr[i,:], label=f'order {i}', color = 'black', alpha = (alph+1)/(order+1), lw = 1)

plt.plot(dmu_arr,sigma_arr[order,:], label=f'order {order}', color = 'black', lw = 3)
plt.xlim(-1.05,1.05)
plt.ylim(1,2.2)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid()
plt.ylabel(r"$\sigma$")
plt.xlabel(r'$\beta \Delta \mu$')
plt.title(r"Theoretical prediction of $\sigma$")
plt.tight_layout()

plt.savefig('sigma.pdf')

plt.figure(figsize=(6,3))

for alph, i in enumerate([1,4,7,8,9]):
    plt.plot(dmu_arr,eta_arr[i,:], label=f'order {i}', color = 'black', alpha = (alph+1)/(order+1), lw = 1)

plt.plot(dmu_arr,eta_arr[order,:], label=f'order {order}', color = 'black', lw = 3)
plt.xlim(-1.05,1.05)
plt.ylim(200,300)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid()
plt.ylabel(r"$\eta$")
plt.xlabel(r'$\beta \Delta \mu$')
plt.title(r"Theoretical prediction of $\eta$")
plt.tight_layout()


plt.savefig('eta.pdf')