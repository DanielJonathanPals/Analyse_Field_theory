import numpy as np
from create_anim import *
from Interface_dynamics import *


order = 10
k_IB = lambda u, f_res, dmu: k_IB_model1(u, f_res, dmu)
f_res_init_guess = 3.

epsilon = -2.95
rho_v = 0.05
dmu_arr = np.arange(-1.0, 1.05, 0.05)

eta_arr = np.zeros((order+1,len(dmu_arr)))    
sigma_arr = np.zeros((order+1,len(dmu_arr)))
chi2_arr = np.zeros((order+1,len(dmu_arr)))
f_res_arr = np.zeros(len(dmu_arr))


if __name__ == '__main__':  
    for i,dmu in enumerate(dmu_arr):
        print(dmu)
        eta, sigma, chi2, f_res = get_eta_sigma_chi2(epsilon, rho_v, k_IB, dmu, f_res_init_guess, order)
        eta_arr[:,i] = eta
        sigma_arr[:,i] = sigma
        chi2_arr[:,i] = chi2
        f_res_arr[i] = f_res
        

    with open('/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Lattice_model/src_plotting/chi_2.npy', 'wb') as f:
        np.save(f, chi2_arr)

    with open('/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Lattice_model/src_plotting/sigma.npy', 'wb') as f:
        np.save(f, sigma_arr)

    with open('/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Lattice_model/src_plotting/eta.npy', 'wb') as f:
        np.save(f, eta_arr)

    with open('/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Lattice_model/src_plotting/f_res.npy', 'wb') as f:
        np.save(f, f_res_arr)