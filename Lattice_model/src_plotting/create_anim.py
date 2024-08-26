import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import convolve2d

file_name = 'eps_-2p95_f_res_2p24_rho_v_0p05_dmu_-1p0_xsize_256_ysize_512'
run_id = 1

def unpack_parameter(line_idx, data):
    line = data[line_idx].split("\n")[0]
    data = line.split("=")[-1]
    if line_idx == 1:
        delimiters = ["(",")",","]
        for delimiter in delimiters:
            data = data.replace(delimiter, " ")
        data = data.split()
        return (int(data[0]), int(data[1]))
    else:
        return float(data)
    

def read_parameters(file_name, run_id):
    file = open(f"/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Lattice_model/Data/{file_name}/{run_id}/lattice_params.txt", "r")
    data = file.readlines()
    file.close()
    size = unpack_parameter(1, data)
    epsilon = unpack_parameter(8, data)
    dmu = unpack_parameter(9, data)
    z_I = unpack_parameter(10, data)
    z_B = unpack_parameter(11, data)
    f_res = unpack_parameter(12, data)
    rho_v = unpack_parameter(13, data)
    t_max = unpack_parameter(17, data)
    save_interval = unpack_parameter(19, data)
    return size, epsilon, dmu, z_I, z_B, f_res, rho_v, t_max, save_interval


def read_data(file_name, run_id):
    file = open(f"/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Lattice_model/Data/{file_name}/{run_id}/states.txt", "r")
    data = file.readlines()
    file.close()
    data = [line.split("\n")[0].split("\t") for line in data]
    data = [[float(element) for element in line] for line in data]
    return np.array(data)


if __name__ == '__main__':
    size, epsilon, dmu, z_I, z_B, f_res, rho_v, t_max, save_interval = read_parameters(file_name, run_id)
    data = read_data(file_name, run_id)
    B_data = np.where(data == 1, 1, 0)
    I_data = np.where(data == 2, 1, 0)

    times = np.arange(0, t_max, save_interval)


    fig, ax = plt.subplots()

    def update(i):
        ax.clear()
        alpha_B = B_data[i*int(size[0]):(i+1)*int(size[0]),:].astype(np.float64)
        alpha_I = I_data[i*int(size[0]):(i+1)*int(size[0]),:].astype(np.float64)
        
        ax.imshow(np.ones((int(size[0]), int(size[1]))), 
                    cmap='Reds', 
                    vmin=0,
                    vmax=1.2,
                    alpha = alpha_B)
        
        ax.imshow(np.ones((int(size[0]), int(size[1]))), 
                    cmap='Blues', 
                    vmin=0,
                    vmax=1.2,
                    alpha = alpha_I)
        
        ax.set_axis_off()
        ax.set_title(f"time = {np.round(times[i], decimals = 2)}s")


    ani = animation.FuncAnimation(fig, update, len(times))
    #ani.save(f"/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Lattice_model/Animations/imagemagick.mp4", writer='imagemagick', fps=20)
    FFwriter = animation.FFMpegWriter(fps=20)
    ani.save(f"/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Lattice_model/Animations/{file_name}.mp4", writer=FFwriter)
