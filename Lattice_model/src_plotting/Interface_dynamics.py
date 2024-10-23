import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import scipy.integrate as integrate


# n_B_partition determines into how many subintervalls the interval [n_B-, n_B+] is partitioned into
# in order to numerically integrate and differentiate functions
n_B_partition = 10000

order = 10      # Needed for ln function

# Define k_IB for model I
def k_IB_model1(u, f_res, dmu):
    return 0.1 * min(1, np.exp(- f_res - dmu - u))

# Define k_IB for model II
def k_IB_model2(u, f_res, dmu):
    return min(1, np.exp(- f_res - dmu + u))

# Compute z_B from rho_v and f_res
def z_B(rho_v, f_res):
    return rho_v / ((1 - rho_v) * (1 + np.exp(f_res)))

# Compute z_I from rho_v and f_res
def z_I(rho_v, f_res):
    return rho_v * np.exp(f_res) / ((1 - rho_v) * (1 + np.exp(f_res)))

# Compute n_B+ and n_B- with dmu dependence numerically
def n_B_pm(epsilon, rho_v, f_res, k_IB, dmu):
    zB = z_B(rho_v, f_res)
    zI = z_I(rho_v, f_res)
    k = lambda n_B: k_IB(4 * epsilon * n_B, f_res, dmu)
    n_Bm = fsolve(lambda n_B: n_B*(1 + (zI + 1)/zB * np.exp(epsilon*4*n_B) + (np.exp(dmu) - 1) * np.exp(epsilon*4*n_B) * (zI * k(n_B)) / (zB + zI * k(n_B) + zB * k(n_B)) * (1 + zI + zB)/zB) - 1, 0)[0]
    n_Bp = fsolve(lambda n_B: n_B*(1 + (zI + 1)/zB * np.exp(epsilon*4*n_B) + (np.exp(dmu) - 1) * np.exp(epsilon*4*n_B) * (zI * k(n_B)) / (zB + zI * k(n_B) + zB * k(n_B)) * (1 + zI + zB)/zB) - 1, 1)[0]
    return n_Bm, n_Bp

def w1(nI, nE, f_res, rho_v):
    zI = z_I(rho_v, f_res)
    return zI * (nI / zI + nE)

def w2(u, nB, nE, f_res, rho_v):
    zB = z_B(rho_v, f_res)
    return zB * (nB * np.exp(u) / zB + nE)

def w3(u, nB, nI, f_res, rho_v, dmu, kIB):
    zI = z_I(rho_v, f_res)
    zB = z_B(rho_v, f_res)
    k = kIB(u, f_res, dmu)
    return zI * k * (nB * np.exp(u) / zB + nI / zI)

def L_inv(u, nB, nI, nE, f_res, rho_v, dmu, kIB):
    W1 = w1(nI, nE, f_res, rho_v)
    W2 = w2(u, nB, nE, f_res, rho_v)
    W3 = w3(u, nB, nI, f_res, rho_v, dmu, kIB)
    return 2 / (9 * (W1 * W2 + W1 * W3 + W2 * W3)) * np.array([[4*W1 + W2 + W3, -2*W1 - 2*W2 + W3, -2*W1 + W2 - 2*W3], [-2*W1 - 2*W2 + W3, W1 + 4*W2 + W3, W1 - 2*W2 - 2*W3], [-2*W1 + W2 - 2*W3, W1 - 2*W2 - 2*W3, W1 + W2 + 4*W3]])

def L(u, nB, nI, nE, f_res, rho_v, dmu, kIB):
    W1 = w1(nI, nE, f_res, rho_v)
    W2 = w2(u, nB, nE, f_res, rho_v)
    W3 = w3(u, nB, nI, f_res, rho_v, dmu, kIB)
    return 1/2 * np.array([[W2 + W3, -W3, -W2],[-W3, W1 + W3, -W1],[-W2, -W1, W1 + W2]])

def mu(u, nB, nI, nE, f_res, rho_v):
    zB = z_B(rho_v, f_res)
    zI = z_I(rho_v, f_res)
    return np.array([ln(nB) + u - ln(zB), ln(nI) - ln(zI), ln(nE)])

def R(u, nB, nI, nE, f_res, rho_v, dmu, kIB):
    k_IB = lambda u: kIB(u, f_res, dmu)
    k_BI = lambda u: kIB(u, f_res, dmu) * np.exp(dmu + f_res + u)
    zB = z_B(rho_v, f_res)
    zI = z_I(rho_v, f_res)
    return np.array([nI * k_IB(u) + nE * zB - nB * (k_BI(u) + np.exp(u)), nB * k_BI(u) + nE * zI - nI * (k_IB(u) + 1), nB * np.exp(u) + nI - nE * (zB + zI)]) + np.dot(L(u, nB, nI, nE, f_res, rho_v, dmu, kIB), mu(u, nB, nI, nE, f_res, rho_v))


def u_approx(order, n_B, rho_v, f_res, k_IB, dmu):
    zB = z_B(rho_v, f_res)
    zI = z_I(rho_v, f_res)
    if order == 0:
        return ln((1-n_B)/n_B * zB / (1 + zI))
    else:
        k = lambda u: k_IB(u, f_res, dmu)
        u = u_approx(order - 1, n_B, rho_v, f_res, k_IB, dmu)
        return ln((1-n_B)/n_B * zB / (1 + zI)) - ln(1 + (np.exp(dmu) - 1) * (zI * k(u)) / (zB + zI * k(u) + zB * k(u)) * (1 + zI + zB)/(1 + zI)) 

def compute_numeric_f_res_coex(order, rho_v, k_IB, dmu, epsilon, f_res_init_guess):
    n_Bm = lambda f_res: n_B_pm(epsilon, rho_v, f_res, k_IB, dmu)[0]
    n_Bp = lambda f_res: n_B_pm(epsilon, rho_v, f_res, k_IB, dmu)[1]
    u = lambda nB, f_res: u_approx(order, nB, rho_v, f_res, k_IB, dmu)
    zI = lambda f_res: z_I(rho_v, f_res)
    zB = lambda f_res: z_B(rho_v, f_res)
    k = lambda nB, f_res: k_IB(u(nB, f_res), f_res, dmu)

    term1 = lambda nB, f_res: -2 * nB**2 - 1 / epsilon * (nB * ln(nB) + (1 - nB) * ln(1 - nB) + ln((1 + z_I(rho_v, f_res)) / z_B(rho_v, f_res)) * nB)
    term2 = lambda f_res: integrate.quad(lambda nB: ln(1 + ((np.exp(dmu) - 1) * zI(f_res) * k(nB, f_res)) / (zB(f_res) + zI(f_res) * k(nB, f_res) + zB(f_res) * k(nB, f_res)) * (1 + zI(f_res) + zB(f_res)) / (1 + zI(f_res))), n_Bm(f_res), n_Bp(f_res))[0]
    return fsolve(lambda f_res: term1(n_Bp(f_res), f_res) - term1(n_Bm(f_res), f_res) - 1/epsilon * term2(f_res), f_res_init_guess)[0]


def Integrate(array, dn_B):
    return np.sum(array) * dn_B

def differentiate(array, dn_B):
    return np.gradient(array, dn_B)

# This result should always be positive
def cumulative_Integrate(array, dn_B):
    cum =  np.cumsum(array[::-1])[::-1] * dn_B
    return np.where(cum>0, 0, cum)

def ln(x_arr):
    if np.all(x_arr > 0):
        return np.log(x_arr)
    else:
        result = 0
        for i in range(order):
            result += (-1)**i * (x_arr - 1)**(i+1) / (i+1)
        log_where_positive = np.log(np.where(x_arr > 0, x_arr, 1))
        return np.where(x_arr > 0, log_where_positive, result)
        


def eta(u, nB, nI, nE, partial_z_nB, f_res, rho_v, dmu, kIB):
    dnB = nB[1] - nB[0]
    dnI = differentiate(nI, dnB)
    dnE = differentiate(nE, dnB)
    return Integrate(partial_z_nB * np.array([np.dot(np.array([1,dnI[i],dnE[i]]), np.dot(L_inv(u[i], nB[i], nI[i], nE[i], f_res, rho_v, dmu, kIB), np.array([1,dnI[i],dnE[i]]))) for i in range(len(nB))]), dnB)

def sigma(u_i, nB, nI_i, nE_i, partial_z_nB_i_0, epsilon, f_res, rho_v, dmu, kIB, curv):
    dnB = nB[1] - nB[0]
    zI = z_I(rho_v, f_res)
    zB = z_B(rho_v, f_res)
    dnI = differentiate(nI_i, dnB)
    dnE = differentiate(nE_i, dnB)
    nBm = nB[0]
    nBp = nB[-1]
    k = lambda u: kIB(u, f_res, dmu)
    nIm = zI / zB * nBm * np.exp(4 * epsilon * nBm) + (np.exp(dmu) - 1) * (zI + zB) * (zI * nBm * np.exp(4 * epsilon * nBm) * k(4 * epsilon * nBm)) / (zB * (zB + k(4 * epsilon * nBm) * (zB + zI)))
    nIp = zI / zB * nBp * np.exp(4 * epsilon * nBp) + (np.exp(dmu) - 1) * (zI + zB) * (zI * nBp * np.exp(4 * epsilon * nBp) * k(4 * epsilon * nBp)) / (zB * (zB + k(4 * epsilon * nBp) * (zB + zI)))
    nEm = 1 / zB * nBm * np.exp(4 * epsilon * nBm) + (np.exp(dmu) - 1) * (zI * nBm * np.exp(4 * epsilon * nBm) * k(4 * epsilon * nBm)) / (zB * (zB + k(4 * epsilon * nBm) * (zB + zI)))
    nEp = 1 / zB * nBp * np.exp(4 * epsilon * nBp) + (np.exp(dmu) - 1) * (zI * nBp * np.exp(4 * epsilon * nBp) * k(4 * epsilon * nBp)) / (zB * (zB + k(4 * epsilon * nBp) * (zB + zI)))
    fm = nBm * ln(nBm) + nIm * ln(nIm) + nEm * ln(nEm) - nBm * ln(zB) - nIm * ln(zI) + 2 * epsilon * nBm**2
    fp = nBp * ln(nBp) + nIp * ln(nIp) + nEp * ln(nEp) - nBp * ln(zB) - nIp * ln(zI) + 2 * epsilon * nBp**2
    return (-fp + fm - epsilon * curv * Integrate(partial_z_nB_i_0, dnB) + Integrate(np.array([np.dot(np.array([1,dnI[i],dnE[i]]), np.dot(L_inv(u_i[i], nB[i], nI_i[i], nE_i[i], f_res, rho_v, dmu, kIB), R(u_i[i], nB[i], nI_i[i], nE_i[i], f_res, rho_v, dmu, kIB))) for i in range(len(nB))]), dnB)) / curv


def CC_T(u, nB, nI, nE, f_res, rho_v, dmu, kIB):
    zI = z_I(rho_v, f_res)
    zB = z_B(rho_v, f_res)
    k_IB = lambda u: kIB(u, f_res, dmu)
    k_BI = lambda u: kIB(u, f_res, dmu) * np.exp(dmu + f_res + u)

    a_11 = nI * k_IB(u) + nE * zB + nB * (k_BI(u) + np.exp(u))
    a_22 = nB * k_BI(u) + nE * zI + nI * (k_IB(u) + 1)
    a_33 = nB * np.exp(u) + nI + nE * (zB + zI)
    a_12 = nI * k_IB(u)  + nB * k_BI(u)
    a_13 = nE * zB + nB * np.exp(u)
    a_23 = nE * zI + nI
    return np.array([[a_11, -a_12, -a_13], [-a_12, a_22, -a_23], [-a_13, -a_23, a_33]])


def chi2(u, nB, nI, nE, partial_z_nB, f_res, rho_v, dmu, kIB):
    dnB = nB[1] - nB[0]
    dnI = differentiate(nI, dnB)
    dnE = differentiate(nE, dnB)
    zI = z_I(rho_v, f_res)
    zB = z_B(rho_v, f_res)
    return Integrate(np.array([partial_z_nB[i] * np.dot(np.array([1,dnI[i],dnE[i]]), np.dot(L_inv(u[i], nB[i], nI[i], nE[i], f_res, rho_v, dmu, kIB), np.dot(CC_T(u[i], nB[i], nI[i], nE[i], f_res, rho_v, dmu, kIB), np.dot(L_inv(u[i], nB[i], nI[i], nE[i], f_res, rho_v, dmu, kIB), np.array([1,dnI[i],dnE[i]]))))) for i in range(len(nB))]), dnB)



    

# Returns an array of n_B values from n_B- to n_B+ with n_B_partition which can be used to integrate and
# differentiate functions numerically
def create_n_B_array(epsilon, rho_v, f_res, k_IB, dmu, n_B_part):
    n_Bm, n_Bp = n_B_pm(epsilon, rho_v, f_res, k_IB, dmu)
    return np.linspace(n_Bm, n_Bp, n_B_part)




def initialization(epsilon, rho_v, f_res, k_IB, dmu, curv):
    nB = create_n_B_array(epsilon, rho_v, f_res, k_IB, dmu, n_B_partition)
    dnB = nB[1] - nB[0]
    zB = z_B(rho_v, f_res) 
    zI = z_I(rho_v, f_res)
    k = lambda u_arr: np.array([k_IB(u, f_res, dmu) for u in u_arr])

    u_0_0 = ln((1 - nB) / nB) - ln((1 + zI) / zB)
    nI_0_0 = zI / (1 + zI) * (1 - nB)
    nE_0_0 = 1 / (1 + zI) * (1 - nB)
    partial_z_nB_0_0 = np.sqrt(- 2 * cumulative_Integrate(u_0_0/epsilon - 4 * nB, dnB))
    eta_0_0 = eta(u_0_0, nB, nI_0_0, nE_0_0, partial_z_nB_0_0, f_res, rho_v, dmu, k_IB)
    partial_t_zeta_0 = epsilon * curv * Integrate(partial_z_nB_0_0, dnB)/eta_0_0
    u_0_1 = ln((1 - nB) / nB) - ln((1 + zI) / zB) + ln(1 + partial_t_zeta_0 * (partial_z_nB_0_0 * (1 + k(u_0_0) + zI + differentiate(nI_0_0, dnB) * (k(u_0_0) - zB))) / ((zB + k(u_0_0) * (zI + zB)) * (1 - nB)))
    u_0 = ln((1 - nB) / nB) - ln((1 + zI) / zB) + ln(1 - (np.exp(dmu) - 1) * ((np.exp(u_0_1) * (zI + zI**2 + zI * zB) * nB * k(u_0_1)) / (zB * (1 - nB) * (zB + k(u_0_1) * (zB + zI)))) + partial_t_zeta_0 * (partial_z_nB_0_0 * (1 + k(u_0_1) + zI + differentiate(nI_0_0, dnB) * (k(u_0_1) - zB))) / ((zB + k(u_0_1) * (zI + zB)) * (1 - nB)))
    nI_0_1 = zI / zB * nB * np.exp(u_0_1) + (np.exp(dmu) - 1) * (zI + zB) * (zI * nB * np.exp(u_0_1) * k(u_0_1)) / (zB * (zB + k(u_0_1) * (zB + zI))) + partial_t_zeta_0 * (partial_z_nB_0_0 * (zB * differentiate(nI_0_0, dnB) - zI)) / (zB + k(u_0_1) * (zI + zB))
    nE_0_1 = 1 / zB * nB * np.exp(u_0_1) + (np.exp(dmu) - 1) * (zI * nB * np.exp(u_0_1) * k(u_0_1)) / (zB * (zB + k(u_0_1) * (zB + zI))) - partial_t_zeta_0 * (partial_z_nB_0_0 * (1 + (1 + differentiate(nI_0_0, dnB)) * k(u_0_1))) / (zB + k(u_0_1) * (zI + zB))
    nI_0 = zI / zB * nB * np.exp(u_0) + (np.exp(dmu) - 1) * (zI + zB) * (zI * nB * np.exp(u_0) * k(u_0)) / (zB * (zB + k(u_0) * (zB + zI))) + partial_t_zeta_0 * (partial_z_nB_0_0 * (zB * differentiate(nI_0_0, dnB) - zI)) / (zB + k(u_0) * (zI + zB))
    nE_0 = 1 / zB * nB * np.exp(u_0) + (np.exp(dmu) - 1) * (zI * nB * np.exp(u_0) * k(u_0)) / (zB * (zB + k(u_0) * (zB + zI))) - partial_t_zeta_0 * (partial_z_nB_0_0 * (1 + (1 + differentiate(nI_0_0, dnB)) * k(u_0))) / (zB + k(u_0) * (zI + zB))
    partial_z_nB_0_1 = np.sqrt(-2 * cumulative_Integrate(u_0_1/epsilon - 4 * nB - curv * partial_z_nB_0_0, dnB))

    return np.stack((nB, u_0_0, nI_0_0, nE_0_0, partial_z_nB_0_0, u_0_1, u_0, nI_0_1, nE_0_1, nI_0, nE_0, partial_z_nB_0_1)), partial_t_zeta_0


def iteratoin_step(epsilon, rho_v, f_res, k_IB, dmu, curv, prev_results, prev_partial_t_zeta):
    nB = prev_results[0,:]
    dnB = nB[1] - nB[0]
    zB = z_B(rho_v, f_res) 
    zI = z_I(rho_v, f_res)
    k = lambda u_arr: np.array([k_IB(u, f_res, dmu) for u in u_arr])

    u_i_0_prev = prev_results[1,:]
    nI_i_0_prev = prev_results[2,:]
    nE_i_0_prev = prev_results[3,:]
    partial_z_nB_i_0_prev = prev_results[4,:]
    partial_t_zeta_i_prev = prev_partial_t_zeta
    u_i_1_prev = prev_results[5,:]
    u_i_prev = prev_results[6,:]
    nI_i_1_prev = prev_results[7,:]
    nE_i_1_prev = prev_results[8,:]
    nI_i_prev = prev_results[9,:]
    nE_i_prev = prev_results[10,:]
    partial_z_nB_i_1_prev = prev_results[11,:]

    u_i_0 = ln((1 - nB) / nB) - ln((1 + zI) / zB) + ln(1 - (np.exp(dmu) - 1) * ((np.exp(u_i_0_prev) * (zI + zI**2 + zI * zB) * nB * k(u_i_0_prev)) / (zB * (1 - nB) * (zB + k(u_i_0_prev) * (zB + zI)))))
    nI_i_0 = zI / zB * nB * np.exp(u_i_0) + (np.exp(dmu) - 1) * (zI + zB) * (zI * nB * np.exp(u_i_0) * k(u_i_0)) / (zB * (zB + k(u_i_0) * (zB + zI)))
    nE_i_0 = 1 / zB * nB * np.exp(u_i_0) + (np.exp(dmu) - 1) * (zI * nB * np.exp(u_i_0) * k(u_i_0)) / (zB * (zB + k(u_i_0) * (zB + zI)))
    partial_z_nB_i_0 = np.sqrt(- 2 * cumulative_Integrate(u_i_0/epsilon - 4 * nB, dnB))
    eta_i_0 = eta(u_i_0, nB, nI_i_0, nE_i_0, partial_z_nB_i_0, f_res, rho_v, dmu, k_IB)
    sigma_i = sigma(u_i_prev, nB, nI_i_prev, nE_i_prev, partial_z_nB_i_0, epsilon, f_res, rho_v, dmu, k_IB, curv)
    partial_t_zeta_i = - curv * sigma_i / eta_i_0
    u_i_1 = ln((1 - nB) / nB) - ln((1 + zI) / zB) + ln(1 - (np.exp(dmu) - 1) * ((np.exp(u_i_1_prev) * (zI + zI**2 + zI * zB) * nB * k(u_i_1_prev)) / (zB * (1 - nB) * (zB + k(u_i_1_prev) * (zB + zI)))) + partial_t_zeta_i * (partial_z_nB_i_0 * (1 + k(u_i_0) + zI + differentiate(nI_i_0, dnB) * (k(u_i_0) - zB))) / ((zB + k(u_i_0) * (zI + zB)) * (1 - nB)))
    u_i = ln((1 - nB) / nB) - ln((1 + zI) / zB) + ln(1 - (np.exp(dmu) - 1) * ((np.exp(u_i_1) * (zI + zI**2 + zI * zB) * nB * k(u_i_1)) / (zB * (1 - nB) * (zB + k(u_i_1) * (zB + zI)))) + partial_t_zeta_i * (partial_z_nB_i_0 * (1 + k(u_i_1) + zI + differentiate(nI_i_0, dnB) * (k(u_i_1) - zB))) / ((zB + k(u_i_1) * (zI + zB)) * (1 - nB)))
    nI_i_1 = zI / zB * nB * np.exp(u_i_1) + (np.exp(dmu) - 1) * (zI + zB) * (zI * nB * np.exp(u_i_1) * k(u_i_1)) / (zB * (zB + k(u_i_1) * (zB + zI))) + partial_t_zeta_i * (partial_z_nB_i_0 * (zB * differentiate(nI_i_0, dnB) - zI)) / (zB + k(u_i_1) * (zI + zB))
    nE_i_1 = 1 / zB * nB * np.exp(u_i_1) + (np.exp(dmu) - 1) * (zI * nB * np.exp(u_i_1) * k(u_i_1)) / (zB * (zB + k(u_i_1) * (zB + zI))) - partial_t_zeta_i * (partial_z_nB_i_0 * (1 + (1 + differentiate(nI_i_0, dnB)) * k(u_i_1))) / (zB + k(u_i_1) * (zI + zB))
    nI_i = zI / zB * nB * np.exp(u_i) + (np.exp(dmu) - 1) * (zI + zB) * (zI * nB * np.exp(u_i) * k(u_i)) / (zB * (zB + k(u_i) * (zB + zI))) + partial_t_zeta_i * (partial_z_nB_i_0 * (zB * differentiate(nI_i_0, dnB) - zI)) / (zB + k(u_i) * (zI + zB))
    nE_i = 1 / zB * nB * np.exp(u_i) + (np.exp(dmu) - 1) * (zI * nB * np.exp(u_i) * k(u_i)) / (zB * (zB + k(u_i) * (zB + zI))) - partial_t_zeta_i * (partial_z_nB_i_0 * (1 + (1 + differentiate(nI_i_0, dnB)) * k(u_i))) / (zB + k(u_i) * (zI + zB))
    partial_z_nB_i_1 = np.sqrt(-2 * cumulative_Integrate(u_i_1/epsilon - 4 * nB - curv * partial_z_nB_i_0, dnB))

    return np.stack((nB, u_i_0, nI_i_0, nE_i_0, partial_z_nB_i_0, u_i_1, u_i, nI_i_1, nE_i_1, nI_i, nE_i, partial_z_nB_i_1)), partial_t_zeta_i


def get_eta_sigma_chi2(epsilon, rho_v, k_IB, dmu, f_res_init_guess, order, curv = 0.001):
    f_res = compute_numeric_f_res_coex(order, rho_v, k_IB, dmu, epsilon, f_res_init_guess)

    eta_arr = np.zeros(order + 1)
    sigma_arr = np.zeros(order + 1)
    chi2_arr = np.zeros(order + 1)

    print("     Computing initialization")
    s_0, partial_t_zeta_0 = initialization(epsilon, rho_v, f_res, k_IB, dmu, curv)
    eta_arr[0] = eta(s_0[1,:], s_0[0,:], s_0[2,:], s_0[3,:], s_0[4,:], f_res, rho_v, dmu, k_IB)
    sigma_arr[0] = sigma(s_0[6,:], s_0[0,:], s_0[9,:], s_0[10,:], s_0[11,:], epsilon, f_res, rho_v, dmu, k_IB, curv)
    chi2_arr[0] = chi2(s_0[1,:], s_0[0,:], s_0[2,:], s_0[3,:], s_0[4,:], f_res, rho_v, dmu, k_IB)

    print("     Computing iteration steps")
    s_prev = s_0
    partial_t_zeta_prev = partial_t_zeta_0
    for i in range(1, order + 1):
        print("         Iteration step ", i)
        s_i, partial_t_zeta_i = iteratoin_step(epsilon, rho_v, f_res, k_IB, dmu, curv, s_prev, partial_t_zeta_prev)
        eta_arr[i] = eta(s_i[6,:], s_i[0,:], s_i[9,:], s_i[10,:], s_i[11,:], f_res, rho_v, dmu, k_IB)
        sigma_arr[i] = sigma(s_i[6,:], s_i[0,:], s_i[9,:], s_i[10,:], s_i[11,:], epsilon, f_res, rho_v, dmu, k_IB, curv)
        chi2_arr[i] = chi2(s_i[6,:], s_i[0,:], s_i[9,:], s_i[10,:], s_i[11,:], f_res, rho_v, dmu, k_IB)

        s_prev = s_i
        partial_t_zeta_prev = partial_t_zeta_i
    
    return eta_arr, sigma_arr, chi2_arr, f_res

# FLEX stuff
def p_tilde_B(n_B, rho_v, f_res, k_IB, dmu, epsilon):
    u = 4 * epsilon * n_B
    K_IB = k_IB(u, f_res, dmu)
    K_BI = K_IB * np.exp(f_res + dmu + u)
    zB = z_B(rho_v, f_res)
    zI = z_I(rho_v, f_res)
    return (zB + K_IB * (zI + zB)) / (np.exp(u) * (1 + K_IB + zI) + (K_BI + K_IB) * (zI + zB) + K_BI + zB)

def tilde_epsilon(rho_v, f_res, k_IB, dmu, epsilon):
    rhoB = p_tilde_B(0.25, rho_v, f_res, k_IB, dmu, epsilon)
    return np.log(rhoB / (1 - rhoB))

def tilde_epsilon_non_coex(rho_v, f_res, k_IB, dmu, epsilon):
    rhoB1 = p_tilde_B(0.25, rho_v, f_res, k_IB, dmu, epsilon)
    rhoB2 = p_tilde_B(0.5, rho_v, f_res, k_IB, dmu, epsilon)
    return np.log(rhoB1 * (1 - rhoB2) / (rhoB2 * (1 - rhoB1)))

def f_res_flex(rho_v, k_IB, dmu, epsilon, f_res_init_guess):
    return fsolve(lambda f_res: p_tilde_B(0.5, rho_v, f_res, k_IB, dmu, epsilon) - 0.5, f_res_init_guess)[0] 

