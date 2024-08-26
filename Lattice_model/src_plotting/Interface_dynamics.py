from Coexistance_diagram import *



# derivative of k_IB with respect to u
def dk_IB_du_model1(u, f_res, dmu):
    if np.exp(- f_res - dmu - u) < 1:
        return -0.1 * np.exp(- f_res - dmu - u)
    else:
        return 0
    
def dk_IB_du_model2(u, f_res, dmu):
    if np.exp(- f_res - dmu + u) < 1:
        return np.exp(- f_res - dmu + u)
    else:
        return 0
    

def bar_n_E(n_B, f_res, rho_v, dmu, k_IB):
    zB = z_B(rho_v, f_res)
    zI = z_I(rho_v, f_res)
    u = u_approx(1, n_B, rho_v, f_res, k_IB, dmu)
    k = k_IB(u, f_res, dmu)
    return n_B / zB * np.exp(u) * (1 + ((np.exp(dmu) - 1) * zI * k) / (zB + zI * k + zB * k))

def bar_n_I(n_B, f_res, rho_v, dmu, k_IB):
    zB = z_B(rho_v, f_res)
    zI = z_I(rho_v, f_res)
    u = u_approx(1, n_B, rho_v, f_res, k_IB, dmu)
    k = k_IB(u, f_res, dmu)
    return zI * n_B / zB * np.exp(u) * (1 + ((np.exp(dmu) - 1) * (zI + zB) * k) / (zB + zI * k + zB * k))

def R(n_B, f_res, rho_v, dmu, k_IB, k_IB_du):
    zB = z_B(rho_v, f_res)
    zI = z_I(rho_v, f_res)
    u = u_approx(0, n_B, rho_v, f_res, k_IB, dmu)
    k = k_IB(u, f_res, dmu)
    dk = k_IB_du(u, f_res, dmu)

    denom = zB + zI * k + zB * k
    return zI * zB / (1 + zI)**2 * (k / denom + zB * dk / (denom**2 * n_B))

def A(n_B, epsilon, rho_v, f_res, k_IB):
    n_Bm = n_B_pm(epsilon, rho_v, f_res, k_IB, 0)[0]
    zB = z_B(rho_v, f_res)
    zI = z_I(rho_v, f_res)
    f = lambda nB: -2 * nB**2 - 1 / epsilon * (nB * np.log(nB) + (1 - nB) * np.log(1 - nB) + np.log((1 + zI) / zB) * nB)
    return np.max([f(n_B) - f(n_Bm),0])       #This should be positive but due to numerical errors it can be slightly negative

def B(n_B, epsilon, rho_v, f_res, k_IB, dmu):
    n_Bm = n_B_pm(epsilon, rho_v, f_res, k_IB, dmu)[0]
    zB = z_B(rho_v, f_res)
    zI = z_I(rho_v, f_res)
    u = lambda nB: u_approx(0, nB, rho_v, f_res, k_IB, dmu)
    k = lambda nB: k_IB(u(nB), f_res, dmu)
    return (1 + zB + zI) / (1 + zI) * 1 / epsilon * integrate.quad(lambda nB: zI * k(nB) / (zB + zI * k(nB) + zB * k(n_B)), n_Bm, n_B)[0]

def dnB_dz(n_B, epsilon, rho_v, f_res, k_IB, dmu):
    return np.sqrt(2 * A(n_B, epsilon, rho_v, f_res, k_IB)) * (1 - (np.exp(dmu) - 1) * B(n_B, epsilon, rho_v, f_res, k_IB, dmu) / 2 * A(n_B, epsilon, rho_v, f_res, k_IB))

def dnE_dz(n_B, epsilon, rho_v, f_res, k_IB, k_IB_du, dmu):
    zB = z_B(rho_v, f_res)
    zI = z_I(rho_v, f_res)
    return (- 1 / (1 + zI) + (np.exp(dmu) - 1) * R(n_B, f_res, rho_v, dmu, k_IB, k_IB_du)) * dnB_dz(n_B, epsilon, rho_v, f_res, k_IB, dmu)

def dnI_dz(n_B, epsilon, rho_v, f_res, k_IB, k_IB_du, dmu):
    zB = z_B(rho_v, f_res)
    zI = z_I(rho_v, f_res)
    return (- zI / (1 + zI) - (np.exp(dmu) - 1) * R(n_B, f_res, rho_v, dmu, k_IB, k_IB_du)) * dnB_dz(n_B, epsilon, rho_v, f_res, k_IB, dmu)

def derivative_vector_without_dnB(n_B, rho_v, f_res, k_IB, k_IB_du, dmu):
    zB = z_B(rho_v, f_res)
    zI = z_I(rho_v, f_res)
    return np.array([1, - zI / (1 + zI) - (np.exp(dmu) - 1) * R(n_B, f_res, rho_v, dmu, k_IB, k_IB_du), - 1 / (1 + zI) + (np.exp(dmu) - 1) * R(n_B, f_res, rho_v, dmu, k_IB, k_IB_du)])

def w1(n_B, f_res, rho_v, dmu, k_IB):
    zI = z_I(rho_v, f_res)
    return zI * (bar_n_I(n_B, f_res, rho_v, dmu, k_IB) / zI + bar_n_E(n_B, f_res, rho_v, dmu, k_IB))

    zB = z_B(rho_v, f_res)
    zI = z_I(rho_v, f_res)
def w2(n_B, f_res, rho_v, dmu, k_IB):
    zB = z_B(rho_v, f_res)
    u = u_approx(1, n_B, rho_v, f_res, k_IB, dmu)
    return zB * (n_B * np.exp(u) / zB + bar_n_E(n_B, f_res, rho_v, dmu, k_IB))

def w3(n_B, f_res, rho_v, dmu, k_IB):
    zI = z_I(rho_v, f_res)
    zB = z_B(rho_v, f_res)
    u = u_approx(1, n_B, rho_v, f_res, k_IB, dmu)
    k = k_IB(u, f_res, dmu)
    return zI * k * (n_B * np.exp(u) / zB + bar_n_I(n_B, f_res, rho_v, dmu, k_IB) / zI)

def L_inv(n_B, f_res, rho_v, dmu, k_IB):
    W1 = w1(n_B, f_res, rho_v, dmu, k_IB)
    W2 = w2(n_B, f_res, rho_v, dmu, k_IB)
    W3 = w3(n_B, f_res, rho_v, dmu, k_IB)
    return 2 / (9 * (W1 * W2 + W1 * W3 + W2 * W3)) * np.array([[4*W1 + W2 + W3, -2*W1 - 2*W2 + W3, -2*W1 + W2 - 2*W3], [-2*W1 - 2*W2 + W3, W1 + 4*W2 + W3, W1 - 2*W2 - 2*W3], [-2*W1 + W2 - 2*W3, W1 - 2*W2 - 2*W3, W1 + W2 + 4*W3]])
    
def sigma(epsilon, rho_v, f_res, k_IB, dmu):
    n_Bm, n_Bp = n_B_pm(epsilon, rho_v, f_res, k_IB, dmu)
    return integrate.quad(lambda nB: dnB_dz(nB, epsilon, rho_v, f_res, k_IB, dmu), n_Bm, n_Bp)[0]

def I_active(epsilon, rho_v, f_res, k_IB, k_IB_du, dmu):
    n_Bm, n_Bp = n_B_pm(epsilon, rho_v, f_res, k_IB, dmu)
    zB = z_B(rho_v, f_res)
    zI = z_I(rho_v, f_res)
    u = lambda n_B: u_approx(0, n_B, rho_v, f_res, k_IB, dmu)
    k = lambda n_B: k_IB(u(n_B), f_res, dmu)
    dk = lambda n_B: k_IB_du(u(n_B), f_res, dmu)
    dnB = lambda n_B: dnB_dz(n_B, epsilon, rho_v, f_res, k_IB, dmu)
    return - epsilon * (zI + zI * zB + zI**2) / (1 + zI) * integrate.quad(lambda nB: dnB(nB) * zB * dk(nB) / (zB + k(nB) * (zB + zI))**2, n_Bm, n_Bp)[0]

def eta(epsilon, rho_v, f_res, k_IB, k_IB_du, dmu):
    n_Bm, n_Bp = n_B_pm(epsilon, rho_v, f_res, k_IB, dmu)
    Linv = lambda n_B: L_inv(n_B, f_res, rho_v, dmu, k_IB)
    vec1 = lambda n_B: derivative_vector_without_dnB(n_B, rho_v, f_res, k_IB, k_IB_du, dmu)
    vec2 = lambda n_B: np.array([dnB_dz(n_B, epsilon, rho_v, f_res, k_IB, dmu), dnI_dz(n_B, epsilon, rho_v, f_res, k_IB, k_IB_du, dmu), dnE_dz(n_B, epsilon, rho_v, f_res, k_IB, k_IB_du, dmu)])
    return integrate.quad(lambda nB: np.dot(vec1(nB), np.dot(Linv(nB), vec2(nB))), n_Bm, n_Bp)[0]

def noise_amp_squared(epsilon, rho_v, f_res, k_IB, k_IB_du, dmu):
    zI = z_I(rho_v, f_res)
    n_Bm, n_Bp = n_B_pm(epsilon, rho_v, f_res, k_IB, dmu)
    Linv = lambda n_B: L_inv(n_B, f_res, rho_v, dmu, k_IB)
    vec1 = lambda n_B: derivative_vector_without_dnB(n_B, rho_v, f_res, k_IB, k_IB_du, dmu)
    vec2 = lambda n_B: np.array([dnB_dz(n_B, epsilon, rho_v, f_res, k_IB, dmu), dnI_dz(n_B, epsilon, rho_v, f_res, k_IB, k_IB_du, dmu), dnE_dz(n_B, epsilon, rho_v, f_res, k_IB, k_IB_du, dmu)])
    M = np.array([[1, -1, 0], [-1, 1, 0], [0, 0, 0]])
    u = lambda n_B: u_approx(0, n_B, rho_v, f_res, k_IB, dmu)
    k = lambda n_B: k_IB(u(n_B), f_res, dmu)
    et = eta(epsilon, rho_v, f_res, k_IB, k_IB_du, dmu)
    I = zI / (1 + zI) * integrate.quad(lambda nB: (1 - nB) * k(nB) * np.dot(vec1(nB), np.dot(Linv(nB), np.dot(M, np.dot(Linv(nB), vec2(nB))))), n_Bm, n_Bp)[0]
    return 2 / et + (np.exp(dmu) - 1) * I / et**2

