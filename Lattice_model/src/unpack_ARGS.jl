x_size = 128 * 1
y_size = 192 * 1
length_of_array = 10
t_max = 100
max_transitions = 1e10
save_interval = 0.1


function create_name(z_I, z_B, x_size, y_size)
    z_I = split(string(round(z_I, digits = 4)),".")[2]
    z_B = split(string(round(z_B, digits = 4)),".")[2]
    return "original_model1_zI_" * z_I * "_zB_" * z_B * "_xsize_" * string(x_size) * "_ysize_" * string(y_size)
end

function unpack_ARGS(array_ID)
    size = (x_size, y_size)
    l_init = zeros(Int8, size)
    l_init[:, 1: Int64(size[2] / 2)] .= 1
    lattice_params = Dict("lattice_size" => size,
                            "lattice_init" => l_init,
                            "upper_bound"  => "periodic",
                            "lower_bound"  => "periodic",
                            "left_bound"   => "bonding",
                            "right_bound"  => "empty")

    k_IB(df, dμ, u) = 0.1 * min(1,exp(-df-dμ-u))

    z_array = collect(LinRange(0.0001, 0.0525, length_of_array))
    model_params = Dict("epsilon" => -2.95, 
                        "k_IB" => k_IB, 
                        "dμ" => 1, 
                        "z_I" => z_array[array_ID], 
                        "z_B" => 0.0526 - z_array[array_ID], 
                        "D" => 1)

    simulation_params = Dict("t_max" => t_max, 
                            "max_transitions" => max_transitions,
                            "save_interval" => save_interval)


    return lattice_params, model_params, simulation_params
end
