from Coexistance_diagram import *


def k_IB(u, f_res, dmu):
    return 0.1 * min(1, np.exp(- f_res - dmu - u))

# derivative of k_IB with respect to u
def dk_IB_du(u, f_res, dmu):
    if np.exp(- f_res - dmu - u) < 1:
        return -0.1 * np.exp(- f_res - dmu - u)
    else:
        return 0
    
def du_approx_dnB(order, n_B, rho_v, f_res, kIB, kIB_du, dmu):
    if order == 0:
        return - 1/(n_B * (1 - n_B))
    else:
        zB = z_B(rho_v, f_res)
        zI = z_I(rho_v, f_res)
        u = u_approx(order - 1, n_B, rho_v, f_res, kIB, dmu)
        k = kIB(u, f_res, dmu)
        dk = kIB_du(u, f_res, dmu)
        denom = zB + k * (zB + zI)
        du = du_approx_dnB(order - 1, n_B, rho_v, f_res, kIB, kIB_du, dmu)
        return - 1/(n_B * (1 - n_B)) - (((np.exp(dmu) - 1) * zI * zB * dk) / (denom**2) * (1 + zI + zB) / (1 + zI)) / (1 + ((np.exp(dmu) - 1) * zI * k) / (denom) * (1 + zI + zB) / (1 + zI)) * du

def bar_n_E(n_B, f_res, rho_v, dmu, kIB):
    zB = z_B(rho_v, f_res)
    zI = z_I(rho_v, f_res)
    u = u_approx(1, n_B, rho_v, f_res, kIB, dmu)
    k = kIB(u, f_res, dmu)
    return n_B / zB * np.exp(u) * (1 + ((np.exp(dmu) - 1) * zI * k) / (zB + zI * k + zB * k))

def bar_n_I(n_B, f_res, rho_v, dmu, kIB):
    zB = z_B(rho_v, f_res)
    zI = z_I(rho_v, f_res)
    u = u_approx(1, n_B, rho_v, f_res, kIB, dmu)
    k = kIB(u, f_res, dmu)
    return zI * n_B / zB * np.exp(u) * (1 + ((np.exp(dmu) - 1) * (zI + zB) * k) / (zB + zI * k + zB * k))

def R(n_B, f_res, rho_v, dmu, kIB, kIB_du):
    zB = z_B(rho_v, f_res)
    zI = z_I(rho_v, f_res)
    u = u_approx(0, n_B, rho_v, f_res, kIB, dmu)
    k = kIB(u, f_res, dmu)
    dk = kIB_du(u, f_res, dmu)

    denom = zB + zI * k + zB * k
    return zI * zB / (1 + zI)**2 * (k / denom + zB * dk / (denom**2 * n_B))


def A(n_B, epsilon, rho_v, f_res, kIB, dmu):
    n_Bm = n_B_pm(epsilon, rho_v, f_res, kIB, dmu)[0]
    zB = z_B(rho_v, f_res)
    zI = z_I(rho_v, f_res)
    f = lambda nB: -2 * nB**2 - 1 / epsilon * (nB * np.log(nB) + (1 - nB) * np.log(1 - nB) + np.log((1 + zI) / zB) * nB)
    return f(n_B) - f(n_Bm)     #This should be positive but due to numerical errors it can be slightly negative

def B(n_B, epsilon, rho_v, f_res, kIB, dmu):
    n_Bm = n_B_pm(epsilon, rho_v, f_res, kIB, dmu)[0]
    zB = z_B(rho_v, f_res)
    zI = z_I(rho_v, f_res)
    u = lambda nB: u_approx(0, nB, rho_v, f_res, kIB, dmu)
    k = lambda nB: kIB(u(nB), f_res, dmu)
    return 1 / epsilon * integrate.quad(lambda nB: np.log(1 + zI * k(nB) / (zB + zI * k(nB) + zB * k(n_B)) * (1 + zB + zI) / (1 + zI)), n_Bm, n_B)[0]

def dnB_dz(n_B, epsilon, rho_v, f_res, kIB, dmu):
    return np.sqrt(np.max([2 * (A(n_B, epsilon, rho_v, f_res, kIB, dmu) - (np.exp(dmu) - 1) * B(n_B, epsilon, rho_v, f_res, kIB, dmu)),0]))

"""
def dnE_dz(n_B, rho_v, f_res, kIB, kIB_du, dmu):
    zB = z_B(rho_v, f_res)
    zI = z_I(rho_v, f_res)
    return (- 1 / (1 + zI) + (np.exp(dmu) - 1) * R(n_B, f_res, rho_v, dmu, kIB, kIB_du)) * dnB_dz(n_B, epsilon, rho_v, f_res, kIB, dmu)
"""

def dnE_dnB(order, n_B, rho_v, f_res, kIB, kIB_du, dmu):
    zB = z_B(rho_v, f_res)
    zI = z_I(rho_v, f_res)
    u = u_approx(order, n_B, rho_v, f_res, kIB, dmu)
    k = kIB(u, f_res, dmu)
    dk = kIB_du(u, f_res, dmu)
    denom = zB + k * (zB + zI)
    nE = bar_n_E(n_B, f_res, rho_v, dmu, kIB)
    return (nE + n_B / zB * np.exp(u) * (np.exp(dmu) - 1) * zI * zB * dk / (denom**2)) * du_approx_dnB(order, n_B, rho_v, f_res, kIB, kIB_du, dmu) + nE / n_B

"""  
def dnI_dz(n_B, epsilon, rho_v, f_res, kIB, kIB_du, dmu):
    zB = z_B(rho_v, f_res)
    zI = z_I(rho_v, f_res)
    return (- zI / (1 + zI) - (np.exp(dmu) - 1) * R(n_B, f_res, rho_v, dmu, kIB, kIB_du)) * dnB_dz(n_B, epsilon, rho_v, f_res, kIB, dmu)
"""

def dnI_dnB(order, n_B, rho_v, f_res, kIB, kIB_du, dmu):
    zB = z_B(rho_v, f_res)
    zI = z_I(rho_v, f_res)
    u = u_approx(order, n_B, rho_v, f_res, kIB, dmu)
    k = kIB(u, f_res, dmu)
    dk = kIB_du(u, f_res, dmu)
    denom = zB + k * (zB + zI)
    nI = bar_n_I(n_B, f_res, rho_v, dmu, kIB)
    return (nI + zI * n_B / zB * np.exp(u) * (np.exp(dmu) - 1) * (zI + zB) * zB * dk / (denom**2)) * du_approx_dnB(order, n_B, rho_v, f_res, kIB, kIB_du, dmu) + nI / n_B
   
def derivative_vector_without_dnB(n_B, rho_v, f_res, kIB, kIB_du, dmu):
    return np.array([1, dnI_dnB(1, n_B, rho_v, f_res, kIB, kIB_du, dmu), dnE_dnB(1, n_B, rho_v, f_res, kIB, kIB_du, dmu)])

def w1(n_B, f_res, rho_v, dmu, kIB):
    zI = z_I(rho_v, f_res)
    return zI * (bar_n_I(n_B, f_res, rho_v, dmu, kIB) / zI + bar_n_E(n_B, f_res, rho_v, dmu, kIB))

def w2(n_B, f_res, rho_v, dmu, kIB):
    zB = z_B(rho_v, f_res)
    u = u_approx(1, n_B, rho_v, f_res, kIB, dmu)
    return zB * (n_B * np.exp(u) / zB + bar_n_E(n_B, f_res, rho_v, dmu, kIB))

def w3(n_B, f_res, rho_v, dmu, kIB):
    zI = z_I(rho_v, f_res)
    zB = z_B(rho_v, f_res)
    u = u_approx(1, n_B, rho_v, f_res, kIB, dmu)
    k = kIB(u, f_res, dmu)
    return zI * k * (n_B * np.exp(u) / zB + bar_n_I(n_B, f_res, rho_v, dmu, kIB) / zI)

def L_inv(n_B, f_res, rho_v, dmu, kIB):
    W1 = w1(n_B, f_res, rho_v, dmu, kIB)
    W2 = w2(n_B, f_res, rho_v, dmu, kIB)
    W3 = w3(n_B, f_res, rho_v, dmu, kIB)
    return 2 / (9 * (W1 * W2 + W1 * W3 + W2 * W3)) * np.array([[4*W1 + W2 + W3, -2*W1 - 2*W2 + W3, -2*W1 + W2 - 2*W3], [-2*W1 - 2*W2 + W3, W1 + 4*W2 + W3, W1 - 2*W2 - 2*W3], [-2*W1 + W2 - 2*W3, W1 - 2*W2 - 2*W3, W1 + W2 + 4*W3]])
    
def sigma(epsilon, rho_v, f_res, kIB, dmu):
    n_Bm, n_Bp = n_B_pm(epsilon, rho_v, f_res, kIB, dmu)
    return integrate.quad(lambda nB: dnB_dz(nB, epsilon, rho_v, f_res, kIB, dmu), n_Bm, n_Bp)[0]

def I_active(epsilon, rho_v, f_res, kIB, kIB_du, dmu):
    n_Bm, n_Bp = n_B_pm(epsilon, rho_v, f_res, kIB, dmu)
    zB = z_B(rho_v, f_res)
    zI = z_I(rho_v, f_res)
    u = lambda n_B: u_approx(0, n_B, rho_v, f_res, kIB, dmu)
    k = lambda n_B: kIB(u(n_B), f_res, dmu)
    dk = lambda n_B: kIB_du(u(n_B), f_res, dmu)
    dnB = lambda n_B: dnB_dz(n_B, epsilon, rho_v, f_res, kIB, dmu)
    return - epsilon * (zI + zI * zB + zI**2) / (1 + zI) * integrate.quad(lambda nB: dnB(nB) * zB * dk(nB) / (zB + k(nB) * (zB + zI))**2, n_Bm, n_Bp)[0]

def eta(epsilon, rho_v, f_res, kIB, kIB_du, dmu):
    n_Bm, n_Bp = n_B_pm(epsilon, rho_v, f_res, kIB, dmu)
    Linv = lambda n_B: L_inv(n_B, f_res, rho_v, dmu, kIB)
    vec = lambda n_B: derivative_vector_without_dnB(n_B, rho_v, f_res, kIB, kIB_du, dmu)
    dnB = lambda n_B: dnB_dz(n_B, epsilon, rho_v, f_res, kIB, dmu)
    return integrate.quad(lambda nB: dnB(nB) * np.dot(vec(nB), np.dot(Linv(nB), vec(nB))), n_Bm, n_Bp)[0]

"""
def noise_amp_squared(epsilon, rho_v, f_res, kIB, kIB_du, dmu):
    zI = z_I(rho_v, f_res)
    n_Bm, n_Bp = n_B_pm(epsilon, rho_v, f_res, kIB, dmu)
    Linv = lambda n_B: L_inv(n_B, f_res, rho_v, dmu, kIB)
    vec1 = lambda n_B: derivative_vector_without_dnB(n_B, rho_v, f_res, kIB, kIB_du, dmu)
    vec2 = lambda n_B: np.array([dnB_dz(n_B, epsilon, rho_v, f_res, kIB, dmu), dnI_dz(n_B, epsilon, rho_v, f_res, kIB, kIB_du, dmu), dnE_dz(n_B, epsilon, rho_v, f_res, kIB, kIB_du, dmu)])
    M = np.array([[1, -1, 0], [-1, 1, 0], [0, 0, 0]])
    u = lambda n_B: u_approx(0, n_B, rho_v, f_res, kIB, dmu)
    k = lambda n_B: kIB(u(n_B), f_res, dmu)
    et = eta(epsilon, rho_v, f_res, kIB, kIB_du, dmu)
    I = zI / (1 + zI) * integrate.quad(lambda nB: (1 - nB) * k(nB) * np.dot(vec1(nB), np.dot(Linv(nB), np.dot(M, np.dot(Linv(nB), vec2(nB))))), n_Bm, n_Bp)[0]
    return 2 / et + (np.exp(dmu) - 1) * I / et**2
"""

def noise_amp_squared(epsilon, rho_v, f_res, kIB, kIB_du, dmu):
    zI = z_I(rho_v, f_res)
    zB = z_B(rho_v, f_res)
    n_Bm, n_Bp = n_B_pm(epsilon, rho_v, f_res, kIB, dmu)
    Linv = lambda n_B: L_inv(n_B, f_res, rho_v, dmu, kIB)
    vec = lambda n_B: derivative_vector_without_dnB(n_B, rho_v, f_res, kIB, kIB_du, dmu)
    dnB = lambda n_B: dnB_dz(n_B, epsilon, rho_v, f_res, kIB, dmu)
    M = np.array([[1, -1, 0], [-1, 1, 0], [0, 0, 0]])
    u = lambda n_B: u_approx(0, n_B, rho_v, f_res, kIB, dmu)
    k = lambda n_B: kIB(u(n_B), f_res, dmu)
    et = eta(epsilon, rho_v, f_res, kIB, kIB_du, dmu)
    I = zI / zB * integrate.quad(lambda nB: dnB(nB) * nB * k(nB) * np.exp(u(nB)) * np.dot(vec(nB), np.dot(Linv(nB), np.dot(M, np.dot(Linv(nB), vec(nB))))), n_Bm, n_Bp)[0]
    return 2 / et + (np.exp(dmu) - 1) * I / et**2




def new_dnB_dz(n_B, epsilon, rho_v, f_res, kIB):
    n_Bm = n_B_pm(epsilon, rho_v, f_res, kIB, 0)[0]
    zB = z_B(rho_v, f_res)
    zI = z_I(rho_v, f_res)
    f = lambda x: -2 * x**2 - 1/epsilon * (x * np.log(x) + (1-x) * np.log(1-x) + np.log((zI + 1)/zB) * x)
    if f(n_B) - f(n_Bm) >= 0:
        return np.sqrt(2 * (f(n_B) - f(n_Bm)))
    else:
        return 0
    
def new_sigma(epsilon, rho_v, f_res, kIB):
    n_Bm, n_Bp = n_B_pm(epsilon, rho_v, f_res, kIB, 0)
    return integrate.quad(lambda n_B: new_dnB_dz(n_B, epsilon, rho_v, f_res, kIB), n_Bm, n_Bp)[0]


def new_L_inv(n_B, rho_v, f_res, kIB):
    zI = z_I(rho_v, f_res)
    zB = z_B(rho_v, f_res)
    u = np.log((1-n_B)/n_B) - np.log((zI+1)/zB)
    w1 = 2 * zI * (1-n_B) / (1+zI)
    w2 = 2 * zB * (1-n_B) / (1+zI)
    w3 = 2 * zI * kIB(u, f_res, 0) * (1-n_B) / (1+zI)
    prefac = 2 / (9*(w1 * w2 + w1 * w3 + w2 * w3))
    return prefac * np.array([[4*w1 + w2 + w3, -2*w1 -2*w2 + w3, -2*w1 + w2 -2*w3],
                             [-2*w1 -2*w2 + w3, w1 + 4*w2 +w3, w1 -2*w2 -2*w3],
                             [-2*w1 + w2 -2*w3, w1 -2*w2 -2*w3, w1 + w2 + 4*w3]])

def new_eta(epsilon, rho_v, f_res, kIB):
    n_Bm, n_Bp = n_B_pm(epsilon, rho_v, f_res, kIB, 0)
    zI = z_I(rho_v, f_res)
    zB = z_B(rho_v, f_res)
    vec = np.array([1, - zI / (1 + zI), -1 / (1+zI)])
    return integrate.quad(lambda nB: np.dot(vec, np.dot(new_L_inv(nB, rho_v, f_res, kIB), vec)) * new_dnB_dz(nB, epsilon, rho_v, f_res, kIB), n_Bm, n_Bp)[0]


