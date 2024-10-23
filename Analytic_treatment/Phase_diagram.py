import matplotlib.pyplot as plt
import numpy as np

Delta_mu = 1
epsilon = - 2.95
f_res = np.linspace(-20, 20, 100)
rho_v = np.linspace(0.01, 0.99, 100)


def k_IB(n_B, z_I, z_B, Delta_mu, epsilon, laplace_n_B):
    Delta_f = np.log(z_I/z_B)
    return 0.1 * min(1, np.exp(- Delta_f - Delta_mu - epsilon * (4 * n_B + laplace_n_B)))
    #return min(1, np.exp(- Delta_f - Delta_mu + epsilon * (4 * n_B + laplace_n_B)))

def z(f_res, rho_v):
    s = rho_v / (1 - rho_v)
    z_B = s/(1+np.exp(f_res))
    z_I = s-z_B
    return z_I, z_B

def RHS(n_B, z_I, z_B, Delta_mu, epsilon, laplace_n_B):
    u = epsilon * (4 * n_B + laplace_n_B)
    return n_B * (1 + 1/z_B * np.exp(u + ((np.exp(Delta_mu) - 1) * z_I * k_IB(n_B, z_I, z_B, Delta_mu, epsilon, laplace_n_B))/(z_B + z_I * k_IB(n_B, z_I, z_B, Delta_mu, epsilon, laplace_n_B) + z_B * k_IB(n_B, z_I, z_B, Delta_mu, epsilon, laplace_n_B))) + z_I/z_B * np.exp(u + ((np.exp(Delta_mu) - 1) * (z_I + z_B) * k_IB(n_B, z_I, z_B, Delta_mu, epsilon, laplace_n_B))/(z_B + z_I * k_IB(n_B, z_I, z_B, Delta_mu, epsilon, laplace_n_B) + z_B * k_IB(n_B, z_I, z_B, Delta_mu, epsilon, laplace_n_B)))) - 1

def numb_of_stable_sols(f_res, rho_v, Delta_mu, epsilon):
    z_I, z_B = z(f_res, rho_v)
    n_B_arr = np.linspace(0, 1, 100)
    aux_arr = [np.sign(RHS(n_B, z_I, z_B, Delta_mu, epsilon, 0)) for n_B in n_B_arr]
    numb_of_sols = int(np.sum(np.abs(np.diff(aux_arr)))/2)
    return numb_of_sols - numb_of_sols // 2

def get_phase_lines(Delta_mu):
    upper_line = np.zeros(np.shape(f_res))
    lower_line = np.zeros(np.shape(f_res))
    aux_arr = np.zeros(np.shape(rho_v))
    grid = np.zeros((np.shape(rho_v)[0], np.shape(f_res)[0]))
    for i in range(np.shape(f_res)[0]):
        for j in range(np.shape(rho_v)[0]):
            aux_arr[j] = numb_of_stable_sols(f_res[i], rho_v[j], Delta_mu, epsilon)
            grid[:,i] = aux_arr
        diff = np.diff(aux_arr)
        if np.any(diff == -1):
            upper_line[i] = rho_v[np.where(diff == -1)[0][0]+1]
        else:
            upper_line[i] = None
        if np.any(diff == 1):
            lower_line[i] = rho_v[np.where(diff == 1)[0][0]]
        else:   
            lower_line[i] = None
    return upper_line, lower_line, grid

upper_line, lower_line, grid = get_phase_lines(Delta_mu)
plt.plot(f_res, upper_line, color = 'black', lw = 8)    
plt.plot(f_res, lower_line, color = 'black', lw = 8)
plt.imshow(grid, extent = [np.min(f_res), np.max(f_res), np.min(rho_v), np.max(rho_v)], origin = 'lower', aspect = 'auto')
plt.show()


"""
f_res, rho_v = np.meshgrid(f_res, rho_v)
phase_arr = np.zeros(np.shape(f_res))
for i in range(np.shape(f_res)[0]):
    for j in range(np.shape(f_res)[1]):
        phase_arr[i,j] = numb_of_stable_sols(f_res[i,j], rho_v[i,j], Delta_mu, epsilon)

plt.imshow(phase_arr, extent = [np.min(f_res), np.max(f_res), np.min(rho_v), np.max(rho_v)], origin = 'lower', aspect = 'auto')
plt.show()
"""