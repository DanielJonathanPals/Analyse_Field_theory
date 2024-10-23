import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import scipy.integrate as integrate
import os


# Change k_IB_model such that it fits the model you want to investigate
def k_IB_model(u, f_res, dmu):
    return 0.1 * min(1, np.exp(- f_res - dmu - u))

def z_B(rho_v, f_res):
    return rho_v / ((1 - rho_v) * (1 + np.exp(f_res)))

def z_I(rho_v, f_res):
    return rho_v * np.exp(f_res) / ((1 - rho_v) * (1 + np.exp(f_res)))

def n_B_pm(epsilon, rho_v, f_res, k_IB, dmu):
    zB = z_B(rho_v, f_res)
    zI = z_I(rho_v, f_res)
    k = lambda n_B: k_IB(4 * epsilon * n_B, f_res, dmu)
    n_Bm = fsolve(lambda n_B: n_B*(1 + (zI + 1)/zB * np.exp(epsilon*4*n_B) + (np.exp(dmu) - 1) * np.exp(epsilon*4*n_B) * (zI * k(n_B)) / (zB + zI * k(n_B) + zB * k(n_B)) * (1 + zI + zB)/zB) - 1, 0)[0]
    n_Bp = fsolve(lambda n_B: n_B*(1 + (zI + 1)/zB * np.exp(epsilon*4*n_B) + (np.exp(dmu) - 1) * np.exp(epsilon*4*n_B) * (zI * k(n_B)) / (zB + zI * k(n_B) + zB * k(n_B)) * (1 + zI + zB)/zB) - 1, 1)[0]
    return n_Bm, n_Bp

def u_approx(order, n_B, rho_v, f_res, k_IB, dmu):
    zB = z_B(rho_v, f_res)
    zI = z_I(rho_v, f_res)
    if order == 0:
        return np.log((1-n_B)/n_B * zB / (1 + zI))
    else:
        k = lambda u: k_IB(u, f_res, dmu)
        u = u_approx(order - 1, n_B, rho_v, f_res, k_IB, dmu)
        return np.log((1-n_B)/n_B * zB / (1 + zI)) - np.log(1 + (np.exp(dmu) - 1) * (zI * k(u)) / (zB + zI * k(u) + zB * k(u)) * (1 + zI + zB)/(1 + zI)) 

def compute_numeric_f_res_coex(order, rho_v, k_IB, dmu, epsilon, f_res_init_guess):
    n_Bm = lambda f_res: n_B_pm(epsilon, rho_v, f_res, k_IB, dmu)[0]
    n_Bp = lambda f_res: n_B_pm(epsilon, rho_v, f_res, k_IB, dmu)[1]
    u = lambda nB, f_res: u_approx(order, nB, rho_v, f_res, k_IB, dmu)
    zI = lambda f_res: z_I(rho_v, f_res)
    zB = lambda f_res: z_B(rho_v, f_res)
    k = lambda nB, f_res: k_IB(u(nB, f_res), f_res, dmu)

    term1 = lambda nB, f_res: -2 * nB**2 - 1 / epsilon * (nB * np.log(nB) + (1 - nB) * np.log(1 - nB) + np.log((1 + z_I(rho_v, f_res)) / z_B(rho_v, f_res)) * nB)
    term2 = lambda f_res: integrate.quad(lambda nB: np.log(1 + ((np.exp(dmu) - 1) * zI(f_res) * k(nB, f_res)) / (zB(f_res) + zI(f_res) * k(nB, f_res) + zB(f_res) * k(nB, f_res)) * (1 + zI(f_res) + zB(f_res)) / (1 + zI(f_res))), n_Bm(f_res), n_Bp(f_res))[0]
    return fsolve(lambda f_res: term1(n_Bp(f_res), f_res) - term1(n_Bm(f_res), f_res) - 1/epsilon * term2(f_res), f_res_init_guess)[0]

# For FLEX
def p_tilde_B(n_B, rho_v, f_res, k_IB, dmu, epsilon):
    u = 4 * epsilon * n_B
    K_IB = k_IB(u, f_res, dmu)
    K_BI = K_IB * np.exp(f_res + dmu + u)
    zB = z_B(rho_v, f_res)
    zI = z_I(rho_v, f_res)
    return (zB + K_IB * (zI + zB)) / (np.exp(u) * (1 + K_IB + zI) + (K_BI + K_IB) * (zI + zB) + K_BI + zB)

def f_res_flex(rho_v, k_IB, dmu, epsilon, f_res_init_guess):
    return fsolve(lambda f_res: p_tilde_B(0.5, rho_v, f_res, k_IB, dmu, epsilon) - 0.5, f_res_init_guess)[0]
    


def create_name(epsilon, rho_v, dmu):
    return 'epsilon_' + str(np.round(epsilon, decimals=3)).replace(".","p") + '_rho_v_' + str(np.round(rho_v, decimals=3)).replace(".","p") + '_dmu_' + str(np.round(dmu, decimals=3)).replace(".","p") + '_long_field'

# Computes the f_res for the given parameters and sets up the folder structure for the simulations
def read_params():
    file = open("/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Lattice_model/src_simulation/simulation_parameters.txt", "r") 
    data = file.readlines()[2:]
    file.close()

    epsilon = np.zeros(len(data))
    rho_v = np.zeros(len(data))
    dmu = np.zeros(len(data))
    t_max = np.zeros(len(data))
    save_interval = np.zeros(len(data))
    numb_of_simulations = np.zeros(len(data))
    f_res_init_guess = np.zeros(len(data))
    f_res = np.zeros(len(data))
    names = np.zeros(len(data), dtype=object)

    warnings = []
    for i, line in enumerate(data):
        d = line.split('\n')[0].split(',')
        epsilon[i], rho_v[i], dmu[i], t_max[i], save_interval[i], numb_of_simulations[i], f_res_init_guess[i] = [float(e) for e in d]
        f_res[i] = compute_numeric_f_res_coex(10, rho_v[i], k_IB_model, dmu[i], epsilon[i], f_res_init_guess[i])
        #f_res[i] = f_res_flex(rho_v[i], k_IB_model, dmu[i], epsilon[i], f_res_init_guess[i])
        name = create_name(epsilon[i], rho_v[i], dmu[i])
        names[i] = name
        if os.path.isdir('/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Lattice_model/Data/' + name):
            os.rename('/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Lattice_model/Data/' + name, '/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Lattice_model/Data/' + name + '_old')
            warnings.append(name)
        os.mkdir('/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Lattice_model/Data/' + name)

    if len(warnings) > 0:
        np.savetxt('/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Lattice_model/Data/WARNINGS.txt', warnings, fmt='%s')
        
        
        
    return epsilon, rho_v, dmu, t_max, save_interval, numb_of_simulations, f_res, names

epsilon, rho_v, dmu, t_max, save_interval, numb_of_simulations, f_res, names = read_params()
np.savetxt('/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Lattice_model/src_parameter_processing/processed_parameters.txt', [epsilon, rho_v, dmu, t_max, save_interval, numb_of_simulations, f_res, names], fmt='%s')

