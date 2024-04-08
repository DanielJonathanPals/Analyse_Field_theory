import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import convolve2d

file_name = 'original_model1_zI_0117_zB_0409_xsize_128_ysize_192'
length_scale = 16

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
    

def read_parameters(file_name):
    file = open(f"/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Lattice_model/Data/{file_name}/lattice_params.txt", "r")
    data = file.readlines()
    file.close()
    size = unpack_parameter(1, data)
    dmu = unpack_parameter(9, data)
    z_I = unpack_parameter(10, data)
    z_B = unpack_parameter(11, data)
    t_max = unpack_parameter(15, data)
    save_interval = unpack_parameter(17, data)
    return size, dmu, z_I, z_B, t_max, save_interval


def read_data(file_name):
    file = open(f"/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Lattice_model/Data/{file_name}/states.txt", "r")
    data = file.readlines()
    data = [line.split("\n")[0].split("\t") for line in data]
    data = [[float(element) for element in line] for line in data]
    return np.array(data)

def average(data, length_scale, file_name):
    size, _, _, _, _, _ = read_parameters(file_name)
    if size[0] % length_scale != 0 or size[1] % length_scale != 0:
        raise ValueError("length_scale must be a divisor of the lattice size")
    kernel = np.ones((length_scale, length_scale)) / (length_scale ** 2)
    data = convolve2d(data, kernel, mode='valid')
    return data[::length_scale,::length_scale]


size, dmu, z_I, z_B, t_max, save_interval = read_parameters(file_name)
data = read_data(file_name)
B_data = np.where(data == 1, 1, 0)
I_data = np.where(data == 2, 1, 0)
if length_scale != 1:
    B_data = average(B_data, length_scale, file_name)
    I_data = average(I_data, length_scale, file_name)

times = np.arange(0, t_max, save_interval)


fig, ax = plt.subplots()

def update(i):
    ax.clear()
    alpha_B = B_data[i*int(size[0] / length_scale):(i+1)*int(size[0] / length_scale),:].astype(np.float64)
    alpha_I = I_data[i*int(size[0] / length_scale):(i+1)*int(size[0] / length_scale),:].astype(np.float64)
    
    ax.imshow(np.ones((int(size[0] / length_scale), int(size[1] / length_scale))), 
                cmap='Reds', 
                vmin=0,
                vmax=1.2,
                alpha = alpha_B)
    
    ax.imshow(np.ones((int(size[0] / length_scale), int(size[1] / length_scale))), 
                cmap='Blues', 
                vmin=0,
                vmax=1.2,
                alpha = alpha_I)
    
    ax.set_axis_off()
    ax.set_title(f"time = {np.round(times[i], decimals = 2)}")


ani = animation.FuncAnimation(fig, update, len(times))
ani.save(f"/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Lattice_model/Animations/{file_name}.mp4", writer='imagemagick', fps=20)
