len_scale = [1]
t_max = [1000]
max_transitions = [1e15]
save_interval = [50.]
f_res = [2.9]
rho_v = [0.05]
epsilon = [-2.95]
dμ = [0.]
mode = ["slap_geometry"]


function create_name(epsilon, f_res, rho_v, x_size, y_size)
    epsilon = replace(string(round(epsilon, digits = 2)), "." => "p")
    f_res = replace(string(round(f_res, digits = 2)), "." => "p")
    rho_v = replace(string(round(rho_v, digits = 2)), "." => "p")
    return "eps_" * epsilon * "_f_res_" * f_res * "_rho_v_" * rho_v * "_xsize_" * string(x_size) * "_ysize_" * string(y_size)
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

    elseif mode[array_ID] == "slap_geometry"
        x_size = 256
        y_size = 512
        s = (x_size, y_size) .* len_scale[array_ID]
        l_init = zeros(Int8, s)
        l_init[x_size ÷ 4 + 1 : 3 * x_size ÷ 4, :] .= 1
        lattice_params = Dict("lattice_size" => s,
                                "lattice_init" => l_init,
                                "upper_bound"  => "empty",
                                "lower_bound"  => "empty",
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
    z_I = rho_v[array_ID] / (1 - rho_v[array_ID]) * exp(f_res[array_ID]) / (1 + exp(f_res[array_ID]))
    z_B = rho_v[array_ID] / (1 - rho_v[array_ID]) / (1 + exp(f_res[array_ID]))

    model_params = Dict("epsilon" => epsilon[array_ID], 
                        "k_IB" => k_IB, 
                        "dμ" => dμ[array_ID], 
                        "z_I" => z_I, 
                        "z_B" => z_B, 
                        "f_res" => f_res[array_ID],
                        "rho_v" => rho_v[array_ID],
                        "D" => 1)

    simulation_params = Dict("t_max" => t_max[array_ID], 
                            "max_transitions" => max_transitions[array_ID],
                            "save_interval" => save_interval[array_ID])


    return lattice_params, model_params, simulation_params
end
