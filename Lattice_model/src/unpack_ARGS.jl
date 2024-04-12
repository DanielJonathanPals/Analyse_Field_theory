len_scale = [3]
t_max = [1]
max_transitions = [1e10]
save_interval = [0.1]
z_array = [0.01]
mode = ["bulk_empty"]


function create_name(z_I, z_B, x_size, y_size)
    z_I = split(string(round(z_I, digits = 4)),".")[2]
    z_B = split(string(round(z_B, digits = 4)),".")[2]
    return "original_model1_zI_" * z_I * "_zB_" * z_B * "_xsize_" * string(x_size) * "_ysize_" * string(y_size)
end

function unpack_ARGS(array_ID)
    if mode[array_ID] == "surface"
        x_size = 32
        y_size = 64
        s = (x_size, y_size) .* len_scale[array_ID]
        l_init = zeros(Int8, s)
        l_init[Int64(s[1] / 2) + 1: end, :] .= 1
        lattice_params = Dict("lattice_size" => s,
                                "lattice_init" => l_init,
                                "upper_bound"  => "empty",
                                "lower_bound"  => "bonding",
                                "left_bound"   => "periodic",
                                "right_bound"  => "periodic",
                                "len_scale" => len_scale[array_ID])
    
    else 
        x_size = 64
        y_size = 64
        s = (x_size, y_size) .* len_scale[array_ID]
        if mode[array_ID] == "bulk_bonding"
            l_init = ones(Int8, s)
        else
            l_init = zeros(Int8, s)
        end
        lattice_params = Dict("lattice_size" => s,
                                "lattice_init" => l_init,
                                "upper_bound"  => "periodic",
                                "lower_bound"  => "periodic",
                                "left_bound"   => "periodic",
                                "right_bound"  => "periodic",
                                "len_scale" => len_scale[array_ID])
    end

    k_IB(df, dμ, u) = 0.1 * min(1,exp(-df-dμ-u))

    model_params = Dict("epsilon" => -2.95, 
                        "k_IB" => k_IB, 
                        "dμ" => 1, 
                        "z_I" => z_array[array_ID], 
                        "z_B" => 0.0526 - z_array[array_ID], 
                        "D" => 1)

    simulation_params = Dict("t_max" => t_max[array_ID], 
                            "max_transitions" => max_transitions[array_ID],
                            "save_interval" => save_interval[array_ID])


    return lattice_params, model_params, simulation_params
end
