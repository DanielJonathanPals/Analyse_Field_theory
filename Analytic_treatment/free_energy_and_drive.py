import matplotlib.pyplot as plt
import numpy as np
from functions import *
from matplotlib.widgets import Slider

f_res = 0
rho_v = 0.05
z_I, z_B = z(f_res, rho_v)
#z_I = 1
#z_B = 0.1
Delta_mu = 1
epsilon = - 2.95
laplace_n_B = 0

vec_scale = 2


def compute_lines(n_B, z_I, z_B, Delta_mu, epsilon, laplace_n_B):
    n_EB = [n_E_of_n_B(nB, z_I, z_B, Delta_mu, epsilon, laplace_n_B) for nB in n_B]
    n_IB = [n_I_of_n_B(nB, z_I, z_B, Delta_mu, epsilon, laplace_n_B) for nB in n_B]
    EB_coords = np.array([inverse_coord_trafo(n_B[i], 1 - n_B[i] - n_EB[i], n_EB[i]) for i in range(len(n_B))])
    IB_coords = np.array([inverse_coord_trafo(n_B[i], n_IB[i], 1 - n_B[i] - n_IB[i]) for i in range(len(n_B))])
    return EB_coords, IB_coords

def is_in_triangle(x, y):
    if x <= 0 or x >= 1 or y <= 0 or y >= min(np.sqrt(3)*x, np.sqrt(3)*(1-x)):
        return False
    else:
        return True


def compute_drive(x, y, z_I, z_B, Delta_mu, epsilon, laplace_n_B):
    u = np.zeros(len(x))
    v = np.zeros(len(y))

    for i in range(len(x)):
        n_B = coord_trafo(x[i], y[i])[0]
        vec = drive(n_B, z_I, z_B, Delta_mu, epsilon, laplace_n_B)
        u[i], v[i] = -project_vector(vec) * (1 - np.exp(Delta_mu))
    return u, v

def compute_free_energy(x, y, z_I, z_B, epsilon):
    f = np.zeros(np.shape(x))

    for i in range(np.shape(x)[0]):
        for j in range(np.shape(x)[1]):
            if not is_in_triangle(x[i,j], y[i,j]):
                f[i,j] = np.nan
                continue
            f[i,j] = free_energy(*coord_trafo(x[i,j], y[i,j]), z_I, z_B, epsilon)
    return f

"""
for i in range(len(x)):
    vec = vector_field_model_A(*coord_trafo(x[i], y[i]), z_I, z_B, Delta_mu, epsilon)
    u[i], v[i] = project_vector(vec)


plt.figure()
plt.quiver(x, y, u, v, scale = 10, scale_units = 'xy')
plt.plot(EB_coords[:,0], EB_coords[:,1], label = 'n_E(n_B)', color =  'black')
plt.plot(IB_coords[:,0], IB_coords[:,1], label = 'n_E(n_I)', color =  'black')
plt.xlim(-0.1,1.1)
plt.ylim(-0.1, np.sqrt(3)/2+0.1)
"""


x, y = triangular_meshgrid(50)
n_B = np.linspace(0, 1, 100)
n_I = np.linspace(0, 1, 100)
xv, yv = np.meshgrid(n_B, n_I)

fig, ax = plt.subplots(1, 2)

ax[0].set_xlim(-0.1, 1.1)
ax[0].set_ylim(-0.1, np.sqrt(3)/2+0.1)
ax[0].set_title(r'$z_I = $' + str(np.round(z_I,decimals = 4)) + r', $z_B = $' + str(np.round(z_B,decimals = 4)))

ax[1].set_xlim(-0.1, 1.1)
ax[1].set_ylim(-0.1, np.sqrt(3)/2+0.1)

for axs in ax:
    axs.plot([0, 1], [0, 0], color = 'red', lw = 5)
    axs.plot([0, 0.5], [0, np.sqrt(3)/2], color = 'red', lw = 5)
    axs.plot([1, 0.5], [0, np.sqrt(3)/2], color = 'red', lw = 5)

u, v = compute_drive(x, y, z_I, z_B, Delta_mu, epsilon, laplace_n_B)
EB_coords, IB_coords = compute_lines(n_B, z_I, z_B, Delta_mu, epsilon, laplace_n_B)
f = compute_free_energy(xv, yv, z_I, z_B, epsilon)

heatmap = ax[0].imshow(f, aspect = 'auto', extent = [0, 1, 0, 1], cmap='viridis', origin = 'lower')
line1, = ax[0].plot(EB_coords[:,0], EB_coords[:,1], label = 'n_E(n_B)', color =  'black')
line2, = ax[0].plot(IB_coords[:,0], IB_coords[:,1], label = 'n_E(n_I)', color =  'black')

vector_field = ax[1].quiver(x, y, u, v, scale = vec_scale, scale_units = 'xy')
line3, = ax[1].plot(EB_coords[:,0], EB_coords[:,1], label = 'n_E(n_B)', color =  'black')
line4, = ax[1].plot(IB_coords[:,0], IB_coords[:,1], label = 'n_E(n_I)', color =  'black')

#fig.colorbar(heatmap, ax=axs, orientation='vertical', fraction=.1)

# adjust the main plot to make room for the sliders
fig.subplots_adjust(bottom=0.25)

ax_mu = fig.add_axes([0.25, 0.2, 0.65, 0.03])
mu_slider = Slider(
    ax=ax_mu,
    label=r'$\Delta \mu$',
    valmin=-1,
    valmax=1,
    valinit=Delta_mu,
)

ax_f_res = fig.add_axes([0.25, 0.15, 0.65, 0.03])
f_res_slider = Slider(
    ax=ax_f_res,
    label=r'$\beta \Delta f_{res}$',
    valmin=-10,
    valmax=10,
    valinit=f_res,
)

ax_rho_v = fig.add_axes([0.25, 0.1, 0.65, 0.03])
rho_v_slider = Slider(
    ax=ax_rho_v,
    label=r'$\rho_v$',
    valmin=0,
    valmax=0.99999,
    valinit=rho_v,
)

ax_laplace = fig.add_axes([0.25, 0.05, 0.65, 0.03])
laplace_slider = Slider(
    ax=ax_laplace,
    label=r'$\Delta n_B$',
    valmin=-2,
    valmax=2,
    valinit=laplace_n_B,
)

def update(val):
    Delta_mu = mu_slider.val
    f_res = f_res_slider.val
    rho_v = rho_v_slider.val
    laplace_n_B = laplace_slider.val
    z_I, z_B = z(f_res, rho_v)
    u, v = compute_drive(x, y, z_I, z_B, Delta_mu, epsilon, laplace_n_B)
    f = compute_free_energy(xv, yv, z_I, z_B, epsilon)
    vector_field.set_UVC(u, v)
    EB_coords, IB_coords = compute_lines(n_B, z_I, z_B, Delta_mu, epsilon, laplace_n_B)
    line1.set_xdata(EB_coords[:,0])
    line1.set_ydata(EB_coords[:,1])
    line2.set_xdata(IB_coords[:,0])
    line2.set_ydata(IB_coords[:,1])
    line3.set_xdata(EB_coords[:,0])
    line3.set_ydata(EB_coords[:,1])
    line4.set_xdata(IB_coords[:,0])
    line4.set_ydata(IB_coords[:,1])
    heatmap.set_data(f)
    ax[0].set_title(r'$z_I = $' + str(np.round(z_I,decimals = 4)) + r', $z_B = $' + str(np.round(z_B,decimals = 4)))
    fig.canvas.draw_idle()

mu_slider.on_changed(update)
f_res_slider.on_changed(update)
rho_v_slider.on_changed(update)
laplace_slider.on_changed(update)

plt.show()





"""
u = np.zeros(len(x))
v = np.zeros(len(y))
for i in range(len(x)):
    if np.sqrt(3)*x[i] - y[i]< 0.3:
        vec = vector_field_original(*coord_trafo(x[i], y[i]), z_I, z_B, Delta_mu, epsilon)
    else:
        vec = np.zeros(3)
    u[i], v[i] = project_vector(vec)



for i in range(len(x)):
    vec = vector_field_original(*coord_trafo(x[i], y[i]), z_I, z_B, Delta_mu, epsilon)
    u[i], v[i] = project_vector(vec)

plt.figure()
plt.quiver(x, y, u, v, scale = 10, scale_units = 'xy')
plt.show()
"""
