function unpack_ARGS(array_ID::Int64)
    # Define parameters here
    x_size = 128 
    y_size = 192 
    t_max = [0.06]
    save_interval = [0.02]
    len_scale = [1]
    modify_laplace = [true]
    z_array = [0.001]
    dt = 0.02 ./ len_scale


    size = (x_size, y_size)
    l_init = zeros(Float64, (size..., 2))
    l_init[:, 1: Int64(size[2] / 2), 1] .= 1.
    lattice_params = Dict("lattice_size" => size,
                            "lattice_init" => l_init,
                            "upper_bound"  => "periodic",
                            "lower_bound"  => "periodic",
                            "left_bound"   => "bonding",
                            "right_bound"  => "empty")

    k_IB(df, dμ, u) = 0.1 * min(1,exp(-df-dμ-u))

    model_params = Dict("epsilon" => -2.95, 
                        "k_IB" => k_IB, 
                        "dμ" => 1, 
                        "z_I" => z_array[array_ID], 
                        "z_B" => 0.0526 - z_array[array_ID], 
                        "D" => 1,
                        "len_scale" => len_scale[array_ID],
                        "modify_laplace" => modify_laplace[array_ID])

    simulation_params = Dict("t_max" => t_max[array_ID], 
                            "dt" => dt[array_ID],
                            "save_interval" => save_interval[array_ID])


    return lattice_params, model_params, simulation_params
end



function create_name(z_I, z_B, x_size, y_size, len_scale, modify_laplace)
    z_I = split(string(round(z_I, digits = 4)),".")[2]
    z_B = split(string(round(z_B, digits = 4)),".")[2]
    modify_laplace ? mod = "modified" : mod = "unmodified"
    return "field_theory_model1_zI_" * z_I * "_zB_" * z_B * "_xsize_" * string(x_size) * "_ysize_" * string(y_size) * "_len_scale_" * string(len_scale) * "_" * mod
end