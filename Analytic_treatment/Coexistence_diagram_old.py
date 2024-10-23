import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import scipy.integrate as integrate

# General setup and defining coeffs

epsilon = -2.95
rho_v = 0.05
df = np.linspace(1.5,4.5, 1000)

z_B = rho_v / ((1 - rho_v) * (1 + np.exp(df)))
z_I = rho_v * np.exp(df) / ((1 - rho_v) * (1 + np.exp(df)))

def u(n_B, epsilon, laplace_n_B):
    return epsilon * (4 * n_B + laplace_n_B)

def k_IB_model1(n_B, z_I, z_B, Delta_mu, epsilon, laplace_n_B):
    Delta_f = np.log(z_I/z_B)
    return 0.1 * min(1, np.exp(- Delta_f - Delta_mu - u(n_B, epsilon, laplace_n_B)))

def k_IB_model2(n_B, z_I, z_B, Delta_mu, epsilon, laplace_n_B):
    Delta_f = np.log(z_I/z_B)
    return min(1, np.exp(- Delta_f - Delta_mu + epsilon * (4 * n_B + laplace_n_B)))

def k_BI_model1(n_B, z_I, z_B, Delta_mu, epsilon, laplace_n_B):
    Delta_f = np.log(z_I/z_B)
    return np.exp(Delta_f + Delta_mu + u(n_B, epsilon, laplace_n_B)) * k_IB_model1(n_B, z_I, z_B, Delta_mu, epsilon, laplace_n_B)

def k_BI_model2(n_B, z_I, z_B, Delta_mu, epsilon, laplace_n_B):
    Delta_f = np.log(z_I/z_B)
    return np.exp(Delta_f + Delta_mu + epsilon * (4 * n_B + laplace_n_B)) * k_IB_model2(n_B, z_I, z_B, Delta_mu, epsilon, laplace_n_B)


# Compute FLEX solution

def p_tilde_B_model1(n_B, z_I, z_B, Delta_mu, epsilon, laplace_n_B):
    U = u(n_B, epsilon, laplace_n_B)
    K_IB = k_IB_model1(n_B, z_I, z_B, Delta_mu, epsilon, laplace_n_B)
    K_BI = k_BI_model1(n_B, z_I, z_B, Delta_mu, epsilon, laplace_n_B)
    return (z_B + K_IB * (z_I + z_B)) / (np.exp(U) * (1 + K_IB + z_I) + (K_BI + K_IB) * (z_I + z_B) + K_BI + z_B)

def p_tilde_B_model2(n_B, z_I, z_B, Delta_mu, epsilon, laplace_n_B):
    U = u(n_B, epsilon, laplace_n_B)
    K_IB = k_IB_model2(n_B, z_I, z_B, Delta_mu, epsilon, laplace_n_B)
    K_BI = k_BI_model2(n_B, z_I, z_B, Delta_mu, epsilon, laplace_n_B)
    return (z_B + K_IB * (z_I + z_B)) / (np.exp(U) * (1 + K_IB + z_I) + (K_BI + K_IB) * (z_I + z_B) + K_BI + z_B)

dmu_flex = np.zeros(len(df))

for i in range(len(df)):
    dmu_flex[i] = fsolve(lambda x: p_tilde_B_model1(0.5, z_I[i], z_B[i], x, epsilon, 0) - 0.5, 0)[0]

plt.plot(dmu_flex, df, label='FLEX model 1',color='red',ls='--')

for i in range(len(df)):
    dmu_flex[i] = fsolve(lambda x: p_tilde_B_model2(0.5, z_I[i], z_B[i], x, epsilon, 0) - 0.5, 0)[0]

plt.plot(dmu_flex, df, label='FLEX model 2',color='blue',ls='--')


# Analytic approximations

for i in range(len(df)):
    dmu_flex[i] = (-2*epsilon + np.log(rho_v) - df[i]) * (12 + rho_v * np.exp(-2*epsilon)) / (1 + rho_v * np.exp(-2*epsilon))

plt.plot(dmu_flex, df, label=r'Ana. approx. ($\rho_v \ll 1$) model 1',color='red') 


dmu_flex = np.linspace(-2, 2, 100)
df_aux = - (np.exp(dmu_flex) - 1) * np.exp(2 * epsilon) * (-1 -2 * epsilon + np.log(rho_v)) - 2 * epsilon + np.log(rho_v)

plt.plot(dmu_flex, df_aux, label=r'Ana. approx. ($\rho_v \ll 1$) model 2', color='blue')


# Analytic expression (compute dmu via dmu = ln(A/B = 1) as defined in the notes)

def n_I_equil_model1(n_B, z_I, z_B, Delta_mu, epsilon):
    K_IB = k_IB_model1(n_B, z_I, z_B, Delta_mu, epsilon, 0)
    return z_I/z_B * n_B * np.exp(epsilon * (4 * n_B) + (np.exp(Delta_mu) - 1) * (z_I + z_B) * K_IB / (z_B + z_I * K_IB + z_B * K_IB))

def n_I_equil_model2(n_B, z_I, z_B, Delta_mu, epsilon):
    K_IB = k_IB_model2(n_B, z_I, z_B, Delta_mu, epsilon, 0)
    return z_I/z_B * n_B * np.exp(epsilon * (4 * n_B) + (np.exp(Delta_mu) - 1) * (z_I + z_B) * K_IB / (z_B + z_I * K_IB + z_B * K_IB))

def n_E_equil_model1(n_B, z_I, z_B, Delta_mu, epsilon):
    K_IB = k_IB_model1(n_B, z_I, z_B, Delta_mu, epsilon, 0)
    return 1/z_B * n_B * np.exp(epsilon * (4 * n_B) + (np.exp(Delta_mu) - 1) * z_I * K_IB / (z_B + z_I * K_IB + z_B * K_IB))

def n_E_equil_model2(n_B, z_I, z_B, Delta_mu, epsilon):
    K_IB = k_IB_model2(n_B, z_I, z_B, Delta_mu, epsilon, 0)
    return 1/z_B * n_B * np.exp(epsilon * (4 * n_B) + (np.exp(Delta_mu) - 1) * z_I * K_IB / (z_B + z_I * K_IB + z_B * K_IB))

def u_tilde(n_B, z_I, z_B):
    return np.log((1 - n_B) / n_B) - np.log((1 + z_I) / z_B)  

def k_IB_u_tilde_model1(u_tild, z_I, z_B):
    Delta_f = np.log(z_I/z_B) 
    return 0.1 * min(1, np.exp(- Delta_f - u_tild))

def k_IB_u_tilde_model2(u_tild, z_I, z_B):
    Delta_f = np.log(z_I/z_B)
    return min(1, np.exp(- Delta_f + u_tild))

bar_n_B_plus = np.zeros(len(df))
bar_n_B_minus = np.zeros(len(df))

for i in range(len(df)):
    bar_n_B_plus[i] = fsolve(lambda x: x + n_I_equil_model1(x, z_I[i], z_B[i], 0, epsilon) + n_E_equil_model1(x, z_I[i], z_B[i], 0, epsilon) - 1, 1)[0]
    bar_n_B_minus[i] = fsolve(lambda x: x + n_I_equil_model1(x, z_I[i], z_B[i], 0, epsilon) + n_E_equil_model1(x, z_I[i], z_B[i], 0, epsilon) - 1, 0)[0]

def A_function(n_B, z_I, z_B, epsilon):
    return -2 * n_B**2 - 1 / epsilon * (n_B * np.log(n_B) + (1 - n_B) * np.log(1 - n_B) + np.log((1 + z_I) / z_B) * n_B)

A = np.zeros(len(df))
B_model1 = np.zeros(len(df))
B_model2 = np.zeros(len(df))

for i in range(len(df)):
    A[i] = A_function(bar_n_B_plus[i], z_I[i], z_B[i], epsilon) - A_function(bar_n_B_minus[i], z_I[i], z_B[i], epsilon)
    B_model1[i] = 1/epsilon * (1 + z_I[i] + z_B[i]) / (1 + z_I[i]) * integrate.quad(lambda x: z_I[i] * k_IB_u_tilde_model1(u_tilde(x, z_I[i], z_B[i]), z_I[i], z_B[i]) / (z_B[i] + z_I[i] * k_IB_u_tilde_model1(u_tilde(x, z_I[i], z_B[i]), z_I[i], z_B[i]) + z_B[i] * k_IB_u_tilde_model1(u_tilde(x, z_I[i], z_B[i]), z_I[i], z_B[i])), bar_n_B_minus[i], bar_n_B_plus[i])[0]
    B_model2[i] = 1/epsilon * (1 + z_I[i] + z_B[i]) / (1 + z_I[i]) * integrate.quad(lambda x: z_I[i] * k_IB_u_tilde_model2(u_tilde(x, z_I[i], z_B[i]), z_I[i], z_B[i]) / (z_B[i] + z_I[i] * k_IB_u_tilde_model2(u_tilde(x, z_I[i], z_B[i]), z_I[i], z_B[i]) + z_B[i] * k_IB_u_tilde_model2(u_tilde(x, z_I[i], z_B[i]), z_I[i], z_B[i])), bar_n_B_minus[i], bar_n_B_plus[i])[0]

dmu = np.log(A/B_model1 + 1)
plt.plot(dmu, df, label='Ana. approx. model 1',color='red', ls=':')

dmu = np.log(A/B_model2 + 1)
plt.plot(dmu, df, label='Ana. approx. model 2',color='blue', ls=':')


plt.legend()
plt.xlim(-2, 2)
plt.xlabel(r'$\beta \Delta \mu$')
plt.ylabel(r'$\beta \Delta f_{res}$')
plt.grid()
plt.savefig('df_against_dmu.pdf')
plt.show()