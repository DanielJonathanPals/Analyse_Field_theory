import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import scipy.integrate as integrate

if __name__ == '__main__':
    # General setup and defining coeffs

    epsilon = -2.95
    rho_v = 0.05
    dmu = np.linspace(-2,2,10)
    order = 9       # This neglects terms of order dmu^(order + 2)

    f_res_init_guess = 3.

    

def k_IB_model1(u, f_res, dmu):
    return 0.1 * min(1, np.exp(- f_res - dmu - u))

def k_IB_model2(u, f_res, dmu):
    return min(1, np.exp(- f_res - dmu + u))

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

def compute_numeric_f_res_coex(order, rho_v, k_IB, dmu, epsilon):
    n_Bm = lambda f_res: n_B_pm(epsilon, rho_v, f_res, k_IB, dmu)[0]
    n_Bp = lambda f_res: n_B_pm(epsilon, rho_v, f_res, k_IB, dmu)[1]
    u = lambda nB, f_res: u_approx(order, nB, rho_v, f_res, k_IB, dmu)
    zI = lambda f_res: z_I(rho_v, f_res)
    zB = lambda f_res: z_B(rho_v, f_res)
    k = lambda nB, f_res: k_IB(u(nB, f_res), f_res, dmu)

    term1 = lambda nB, f_res: -2 * nB**2 - 1 / epsilon * (nB * np.log(nB) + (1 - nB) * np.log(1 - nB) + np.log((1 + z_I(rho_v, f_res)) / z_B(rho_v, f_res)) * nB)
    term2 = lambda f_res: integrate.quad(lambda nB: np.log(1 + ((np.exp(dmu) - 1) * zI(f_res) * k(nB, f_res)) / (zB(f_res) + zI(f_res) * k(nB, f_res) + zB(f_res) * k(nB, f_res)) * (1 + zI(f_res) + zB(f_res)) / (1 + zI(f_res))), n_Bm(f_res), n_Bp(f_res))[0]
    return fsolve(lambda f_res: term1(n_Bp(f_res), f_res) - term1(n_Bm(f_res), f_res) - 1/epsilon * term2(f_res), f_res_init_guess)[0]


def compute_first_order_f_res_coex(rho_v, k_IB, dmu, epsilon):
    n_Bm = lambda f_res: n_B_pm(epsilon, rho_v, f_res, k_IB, 0)[0]
    n_Bp = lambda f_res: n_B_pm(epsilon, rho_v, f_res, k_IB, 0)[1]
    u = lambda nB, f_res: u_approx(0, nB, rho_v, f_res, k_IB, dmu)
    zI = lambda f_res: z_I(rho_v, f_res)
    zB = lambda f_res: z_B(rho_v, f_res)
    k = lambda nB, f_res: k_IB(u(nB, f_res), f_res, dmu)

    term1 = lambda nB, f_res: -2 * nB**2 - 1 / epsilon * (nB * np.log(nB) + (1 - nB) * np.log(1 - nB) + np.log((1 + z_I(rho_v, f_res)) / z_B(rho_v, f_res)) * nB)
    term2 = lambda f_res: integrate.quad(lambda nB: (dmu * zI(f_res) * k(nB, f_res)) / (zB(f_res) + zI(f_res) * k(nB, f_res) + zB(f_res) * k(nB, f_res)) * (1 + zI(f_res) + zB(f_res)) / (1 + zI(f_res)), n_Bm(f_res), n_Bp(f_res))[0]
    return fsolve(lambda f_res: term1(n_Bp(f_res), f_res) - term1(n_Bm(f_res), f_res) - 1/epsilon * term2(f_res), f_res_init_guess)[0]


# Compute FLEX solution

def p_tilde_B(n_B, rho_v, f_res, k_IB, dmu, epsilon):
    u = 4 * epsilon * n_B
    K_IB = k_IB(u, f_res, dmu)
    K_BI = K_IB * np.exp(f_res + dmu + u)
    zB = z_B(rho_v, f_res)
    zI = z_I(rho_v, f_res)
    return (zB + K_IB * (zI + zB)) / (np.exp(u) * (1 + K_IB + zI) + (K_BI + K_IB) * (zI + zB) + K_BI + zB)

def f_res_flex(rho_v, k_IB, dmu, epsilon):
    return fsolve(lambda f_res: p_tilde_B(0.5, rho_v, f_res, k_IB, dmu, epsilon) - 0.5, f_res_init_guess)[0]

if __name__ == '__main__':
    # Plot results
    
    plt.figure(figsize=(6,3))

    # Model 1

    # Numeric solution

    # first order
    df = np.zeros(len(dmu))
    for i in range(len(dmu)):
        df[i] = compute_first_order_f_res_coex(rho_v, k_IB_model1, dmu[i], epsilon)
    plt.plot(dmu, df, label=f'order {1}',color='black', lw = 1, ls='--')

    for alph, ord in enumerate([3,6,7,8]):
        df = np.zeros(len(dmu))
        for i in range(len(dmu)):
            df[i] = compute_numeric_f_res_coex(ord, rho_v, k_IB_model1, dmu[i], epsilon)

        plt.plot(dmu, df, label=f'order {ord + 1}',color='black', alpha = (alph+1)/(order + 1), lw = 1)
    
    df = np.zeros(len(dmu))
    for i in range(len(dmu)):
        df[i] = compute_numeric_f_res_coex(order, rho_v, k_IB_model1, dmu[i], epsilon)
    plt.plot(dmu, df, label=f'order {order + 1}',color='black', lw = 3)


    # FLEX solution
    df = np.zeros(len(dmu))
    for i in range(len(dmu)):
        df[i] = f_res_flex(rho_v, k_IB_model1, dmu[i], epsilon)

    plt.plot(dmu, df, label='FLEX',color='blue',ls='--')

    """
    # Model 2

    # Numeric solution
    df = np.zeros(len(dmu))
    for i in range(len(dmu)):
        df[i] = compute_numeric_f_res_coex(order, rho_v, k_IB_model2, dmu[i], epsilon)

    plt.plot(dmu, df, label='Numeric model 1 (5. order)',color='blue')

    # First order solution
    df = np.zeros(len(dmu))
    for i in range(len(dmu)):
        df[i] = compute_first_order_f_res_coex(rho_v, k_IB_model2, dmu[i], epsilon)

    plt.plot(dmu, df, label='First order model 1',color='blue',ls='--')

    # FLEX solution
    df = np.zeros(len(dmu))
    for i in range(len(dmu)):
        df[i] = f_res_flex(rho_v, k_IB_model2, dmu[i], epsilon)

    plt.plot(dmu, df, label='FLEX model 1',color='blue',ls=':')
    """

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlim(-2, 2)
    plt.xlabel(r'$\beta \Delta \mu$')
    plt.ylabel(r'$\beta \Delta f_{\mathrm{res.}}$')
    plt.grid()
    plt.title('Coexistence line')
    plt.tight_layout()
    plt.savefig(f'Coexistence_diagram_eps_{epsilon}_rho_v_{rho_v}.pdf')
    plt.show()



