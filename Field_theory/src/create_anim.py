import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import convolve2d

file_name = 'field_theory_model1_zI_001_zB_0516_xsize_128_ysize_192_len_scale_3_modified'

def unpack_parameter(line_idx, data):
    line = data[line_idx].split("\n")[0]
    data = line.split("=")[-1]
    if line_idx == 1:
        delimiters = ["(",")",","]
        for delimiter in delimiters:
            data = data.replace(delimiter, " ")
        data = data.split()
        return (int(data[0]), int(data[1]))
    elif line_idx == 14:
        if data == " false":
            return False
        else :
            return True
    else:
        return float(data)
    

def read_parameters(file_name):
    file = open(f"/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Field_theory/Data/{file_name}/lattice_params.txt", "r")
    data = file.readlines()
    file.close()
    size = unpack_parameter(1, data)
    dmu = unpack_parameter(9, data)
    z_I = unpack_parameter(10, data)
    z_B = unpack_parameter(11, data)
    len_scale = unpack_parameter(13, data)
    modified = unpack_parameter(14, data)
    t_max = unpack_parameter(17, data)
    save_interval = unpack_parameter(19, data)
    return size, dmu, z_I, z_B, len_scale, modified, t_max, save_interval


def read_data(file_name, specification):
    file = open(f"/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Field_theory/Data/{file_name}/{specification}.txt", "r")
    data = file.readlines()
    file.close()
    data = [line.split("\n")[0].split("\t") for line in data]
    data = [[float(element) for element in line] for line in data]
    return np.array(data)


size, dmu, z_I, z_B, len_scale, modified, t_max, save_interval = read_parameters(file_name)
B_states = read_data(file_name, "B_states")
I_states = read_data(file_name, "I_states")
B_ratios = read_data(file_name, "B_ratios")
I_ratios = read_data(file_name, "I_ratios")

times = np.arange(0, t_max, save_interval)


fig, ax = plt.subplots()

def update(i):
    ax.clear()
    alpha_B = B_states[i*size[0]:(i+1)*size[0],:].astype(np.float64)
    alpha_I = I_states[i*size[0]:(i+1)*size[0],:].astype(np.float64)

    ax.imshow(np.ones((size[0], size[1])), 
                cmap='Reds', 
                vmin=0,
                vmax=1.2,
                alpha = alpha_B)
    
    ax.imshow(np.ones((size[0], size[1])), 
                cmap='Blues', 
                vmin=0,
                vmax=1.2,
                alpha = alpha_I)
    
    ax.set_axis_off()
    ax.set_title(f"time = {np.round(times[i], decimals = 2)}s")


ani = animation.FuncAnimation(fig, update, len(times))
ani.save(f"/scratch/d/Daniel.Pals/Masterthesis/Coding/Analyse_Field_theory/Field_theory/Animations/{file_name}.mp4", writer='imagemagick', fps=20)
