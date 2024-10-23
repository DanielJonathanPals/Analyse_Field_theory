import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider


f_res = 0
rho_v = 0.05
Delta_mu = 1
epsilon = - 2.95
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

n_B_arr = np.linspace(0, 1, 100)
def compute_laplace_arr(f_res, rho_v, Delta_mu):
    aux_laplace_arr = [solve_for_laplace_n_B(n_B, f_res, rho_v, Delta_mu, epsilon) for n_B in n_B_arr]
    aux_laplace_arr[0] = laplace_n_B_arr[0]
    aux_laplace_arr[-1] = laplace_n_B_arr[-1]
    return aux_laplace_arr

fig, ax = plt.subplots(1, 1)
ax.set_ylim(-0.1, 1.1)
ax.set_xlim(laplace_n_B_arr[0]-0.1, laplace_n_B_arr[-1]+0.1)
ax.set_xlabel(r'$\Delta n_B$')
ax.set_ylabel(r'$n_B$')
ax.grid()

line, = ax.plot(compute_laplace_arr(f_res, rho_v, Delta_mu), n_B_arr)

# adjust the main plot to make room for the sliders
fig.subplots_adjust(bottom=0.3)

ax_mu = fig.add_axes([0.25, 0.15, 0.65, 0.03])
mu_slider = Slider(
    ax=ax_mu,
    label=r'$\Delta \mu$',
    valmin=-1,
    valmax=1,
    valinit=Delta_mu,
)

ax_f_res = fig.add_axes([0.25, 0.1, 0.65, 0.03])
f_res_slider = Slider(
    ax=ax_f_res,
    label=r'$\beta \Delta f_{res}$',
    valmin=-10,
    valmax=10,
    valinit=f_res,
)

ax_rho_v = fig.add_axes([0.25, 0.05, 0.65, 0.03])
rho_v_slider = Slider(
    ax=ax_rho_v,
    label=r'$\rho_v$',
    valmin=0,
    valmax=0.99999,
    valinit=rho_v,
)

def update(val):
    Delta_mu = mu_slider.val
    f_res = f_res_slider.val
    rho_v = rho_v_slider.val
    line.set_xdata(compute_laplace_arr(f_res, rho_v, Delta_mu))
    fig.canvas.draw_idle()

mu_slider.on_changed(update)
f_res_slider.on_changed(update)
rho_v_slider.on_changed(update)

plt.show()