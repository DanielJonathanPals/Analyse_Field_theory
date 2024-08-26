using DelimitedFiles


function k_IB(df, dμ, u)
    return 0.1 * min(1,exp(-df-dμ-u))
end


ToInt64(str::String) = Int64(parse(Float64, str))

function read_params()
    data = readdlm("src_parameter_processing/processed_parameters.txt", ' ', String, '\n')

    epsilon = [parse(Float64, d) for d in data[1, :]]
    rho_v = [parse(Float64, d) for d in data[2, :]]
    dmu = [parse(Float64, d) for d in data[3, :]]
    t_max = [parse(Float64, d) for d in data[4, :]]
    save_interval = [parse(Float64, d) for d in data[5, :]]
    numb_of_simulations = ToInt64.(data[6, :])
    f_res = [parse(Float64, d) for d in data[7, :]]
    names = data[8, :]
    return epsilon, rho_v, dmu, t_max, save_interval, numb_of_simulations, f_res, names
end


# transforms Array_ID into param_ID (which parameter set is used) and run_ID (ID within the parameter set)
function transform_ID(Array_ID)
    _, _, _, _, _, numb_of_simulations, _, _ = read_params()
    param_ID = 1
    for sim in numb_of_simulations
        if Array_ID <= sim
            return param_ID, Array_ID
        end
        Array_ID -= sim
        param_ID += 1
    end
    return param_ID, Array_ID
end



struct lattice_prms
    lattice_size::Tuple{Int64, Int64}
    lattice_init::Array{Int8, 2}
    upper_bound::String
    lower_bound::String
    left_bound::String
    right_bound::String
end

struct model_prms
    epsilon::Float64
    k_IB::Function
    dμ::Float64
    z_I::Float64
    z_B::Float64
    f_res::Float64
    rho_v::Float64
    D::Float64
end

struct simulation_prms
    t_max::Float64
    max_transitions::Float64
    save_interval::Float64
end

function unpack_ARGS(array_ID)
    param_ID, run_ID = transform_ID(array_ID)
    epsilon, rho_v, dμ, t_max, save_interval, _, f_res, names = read_params()

    x_size = 256
    y_size = 512
    s = (x_size, y_size)
    l_init = zeros(Int8, s)
    l_init[x_size ÷ 4 + 1 : 3 * x_size ÷ 4, :] .= 1
    lattice_params = lattice_prms(s, l_init, "empty", "empty", "periodic", "periodic")

    z_I = rho_v[param_ID] / (1 - rho_v[param_ID]) * exp(f_res[param_ID]) / (1 + exp(f_res[param_ID]))
    z_B = rho_v[param_ID] / (1 - rho_v[param_ID]) / (1 + exp(f_res[param_ID]))

    model_params = model_prms(epsilon[param_ID], k_IB, dμ[param_ID], z_I, z_B, f_res[param_ID], rho_v[param_ID], 1)

    simulation_params = simulation_prms(t_max[param_ID], 1e17, save_interval[param_ID])


    return lattice_params, model_params, simulation_params, run_ID, names[param_ID]
end
