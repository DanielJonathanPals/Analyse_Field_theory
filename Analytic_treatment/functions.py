import numpy as np

def z(f_res, rho_v):
    s = rho_v / (1 - rho_v)
    z_B = s/(1+np.exp(f_res))
    z_I = s-z_B
    return z_I, z_B

# x and y parameterize the trangle given by the intersection of the space
# n_B >= 0, n_I >= 0, n_E >= 0 and n_B + n_I + n_E = 1. Here x ranges from 0 to 1
# and decribes the position along the base of the triangle. y ranges from 0 to
# sqrt(3)/2*x and describes hight of the triangle. The function coord_trafo
# transforms these coordinates to the coordinates of the space.
def coord_trafo(x, y):
    return np.array([1,0,0]) + x*np.array([-1,1,0]) + y/np.sqrt(3)*np.array([-1,-1,2])

def inverse_coord_trafo(n_B, n_I, n_E):
    return np.array([n_I + 0.5 * n_E, np.sqrt(3)/2 * n_E])

# Projects a vector into the tangent plane defined by x + y + z = 0 and expresses
# it in the x,y basis.
def project_vector(vec):
    x_comp = np.dot(vec, np.array([-1,1,0]))/np.sqrt(2)
    y_comp = np.dot(vec, np.array([-1,-1,2])/np.sqrt(6))
    return np.array([x_comp, y_comp])

def k_IB(n_B, z_I, z_B, Delta_mu, epsilon, laplace_n_B):
    Delta_f = np.log(z_I/z_B)
    return 0.1 * min(1, np.exp(- Delta_f - Delta_mu - epsilon * (4 * n_B + laplace_n_B)))
    #return min(1, np.exp(- Delta_f - Delta_mu + epsilon * (4 * n_B + laplace_n_B)))

def vector_field_model_A(n_B, n_I, n_E, z_I, z_B, Delta_mu, epsilon, laplace_n_B):
    mu = np.array([np.log(n_B+0.00001)+epsilon*(4*n_B + laplace_n_B) -np.log(z_B),
                   np.log(n_I+0.00001)-np.log(z_I),
                   np.log(n_E+0.00001)])
    
    sigma_1 = [np.array([[0,1,0]]),np.array([[0,0,1]])]
    sigma_2 = [np.array([[1,0,0]]),np.array([[0,0,1]])]
    sigma_3 = [np.array([[1,0,0]]),np.array([[0,1,0]])]
    sigma = [sigma_1, sigma_2, sigma_3]

    k_1 = z_I
    k_2 = z_B
    k_3 = z_I * k_IB(n_B, z_I, z_B, Delta_mu, epsilon, laplace_n_B)
    k = [k_1, k_2, k_3]

    L = np.zeros((3,3))
    for i in range(3):
        L += 0.5 * k[i] * np.matmul(np.transpose(sigma[i][1] - sigma[i][0]), sigma[i][1] - sigma[i][0]) * (np.exp(np.matmul(sigma[i][0], mu))[0] + np.exp(np.matmul(sigma[i][1], mu))[0])
        
    v = (sigma_3[1] - sigma_3[0]) * k_3 * np.exp(mu[0])

    return - np.matmul(L, mu) + (np.exp(Delta_mu) - 1) * v

def vector_field_original(n_B, n_I, n_E, z_I, z_B, Delta_mu, epsilon, laplace_n_B):
    Delta_f = np.log(z_I/z_B)
    u = epsilon * (4 * n_B + laplace_n_B)
    k_I_B = k_IB(n_B, z_I, z_B, Delta_mu, epsilon, laplace_n_B)
    k_BI = k_I_B * np.exp(Delta_f + Delta_mu + u)
    return np.array([n_I * k_I_B + n_E * z_B - n_B * (k_BI + np.exp(u)),
                     n_B * k_BI + n_E * z_I - n_I * (k_I_B + 1),
                     n_B * np.exp(u) + n_I * 1 - n_E * (z_B + z_I)])

def drive(n_B, z_I, z_B, Delta_mu, epsilon, laplace_n_B):
    k_1 = z_I
    k_2 = z_B
    k_3 = z_I * k_IB(n_B, z_I, z_B, Delta_mu, epsilon, laplace_n_B)

    prefactor = k_3 / (3 * (k_1*k_2 + k_1*k_3 + k_2*k_3))
    return prefactor * np.array([-2*k_1 - k_2, k_1 + 2*k_2, k_1 - k_2])

def n_E_of_n_B(n_B, z_I, z_B, Delta_mu, epsilon, laplace_n_B):
    K_IB = k_IB(n_B, z_I, z_B, Delta_mu, epsilon, laplace_n_B)
    return n_B/z_B * np.exp(epsilon * (4 * n_B + laplace_n_B)) * (1 + (np.exp(Delta_mu) - 1) * z_I * K_IB / (z_B + z_I * K_IB + z_B * K_IB))

def n_I_of_n_B(n_B, z_I, z_B, Delta_mu, epsilon, laplace_n_B):
    K_IB = k_IB(n_B, z_I, z_B, Delta_mu, epsilon, laplace_n_B)
    return z_I * n_B/z_B * np.exp(epsilon * (4 * n_B + laplace_n_B)) * (1 + (np.exp(Delta_mu) - 1) * (z_I + z_B) * K_IB / (z_B + z_I * K_IB + z_B * K_IB))

def free_energy(n_B, n_I, n_E, z_I, z_B, epsilon):
    return n_B * np.log(n_B) + n_I * np.log(n_I) + n_E * np.log(n_E) + epsilon * (2 * n_B**2) - np.log(z_B) * n_B - np.log(z_I) * n_I

def triangular_meshgrid(n):
    numb = int(n*(n+1)/2)
    x = np.zeros(numb)
    y = np.zeros(numb)
    for i in range(n):
        for j in range(i+1):
            x[i*(i+1)//2+j] = 1/2 - i/(2*(n-1)) + j/(n-1)
            y[i*(i+1)//2+j] = np.sqrt(3)/2 * (1 - i/(n-1))
    return x, y