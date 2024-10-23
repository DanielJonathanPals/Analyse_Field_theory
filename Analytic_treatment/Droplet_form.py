import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


f_res = 2.87
rho_v = 0.05
Delta_mu = 0.
epsilon = - 2.95
n_B_max = 0.027
dn_B_init = 0.002
laplace_n_B_arr = np.linspace(-2, 2, 500)


def z(f_res, rho_v):
    s = rho_v / (1 - rho_v)
    z_B = s/(1+np.exp(f_res))
    z_I = s-z_B
    return z_I, z_B

def k_IB(n_B, z_I, z_B, Delta_mu, epsilon, laplace_n_B):
    Delta_f = np.log(z_I/z_B)
    #return 0.1 * min(1, np.exp(- Delta_f - Delta_mu - epsilon * (4 * n_B + laplace_n_B)))
    return min(1, np.exp(- Delta_f - Delta_mu + epsilon * (4 * n_B + laplace_n_B)))

def RHS(n_B, z_I, z_B, Delta_mu, epsilon, laplace_n_B):
    u = epsilon * (4 * n_B + laplace_n_B)
    return n_B * (1 + 1/z_B * np.exp(u + ((np.exp(Delta_mu) - 1) * z_I * k_IB(n_B, z_I, z_B, Delta_mu, epsilon, laplace_n_B))/(z_B + z_I * k_IB(n_B, z_I, z_B, Delta_mu, epsilon, laplace_n_B) + z_B * k_IB(n_B, z_I, z_B, Delta_mu, epsilon, laplace_n_B))) + z_I/z_B * np.exp(u + ((np.exp(Delta_mu) - 1) * (z_I + z_B) * k_IB(n_B, z_I, z_B, Delta_mu, epsilon, laplace_n_B))/(z_B + z_I * k_IB(n_B, z_I, z_B, Delta_mu, epsilon, laplace_n_B) + z_B * k_IB(n_B, z_I, z_B, Delta_mu, epsilon, laplace_n_B)))) - 1

def solve_for_laplace_n_B(n_B, f_res, rho_v, Delta_mu, epsilon):
    z_I, z_B = z(f_res, rho_v)
    if RHS(n_B, z_I, z_B, Delta_mu, epsilon, laplace_n_B_arr[0]) <= 0:
        return None
    for laplace_n_B in laplace_n_B_arr:
        if RHS(n_B, z_I, z_B, Delta_mu, epsilon, laplace_n_B) <= 0:
            return laplace_n_B
    return None

def compute_curve(n_B_max, f_res, rho_v, Delta_mu):
    x_arr = np.linspace(0, 3, 500)
    profile_arr = np.zeros(len(x_arr))
    dx = x_arr[1] - x_arr[0]
    profile_arr[0] = n_B_max
    curve = solve_for_laplace_n_B(n_B_max, f_res, rho_v, Delta_mu, epsilon)
    profile_arr[1] = dn_B_init * dx + n_B_max
    for i in range(2, len(x_arr)):
        curve = solve_for_laplace_n_B(profile_arr[i-1], f_res, rho_v, Delta_mu, epsilon)
        profile_arr[i] = curve * dx**2 + 2 * profile_arr[i-1] - profile_arr[i-2]
    return x_arr, profile_arr
    
x,p = compute_curve(n_B_max, f_res, rho_v, Delta_mu)
fig, ax = plt.subplots(1, 1)
ax.plot(x,p)
plt.show()