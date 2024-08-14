using DelimitedFiles

ToInt64(str::String) = Int64(parse(Float64, str))

function read_data(file_name::String, run_id::Int)
    data = readdlm("Data/$file_name/$run_id/states.txt", '\t', Int8, '\n')
    lattice_params = readdlm("Data/$file_name/$run_id/lattice_params.txt", '\t', String, '\n')
    x_size = ToInt64(String(split(lattice_params[2], ['(', ',', ')'])[2]))
    y_size = ToInt64(String(split(lattice_params[2], ['(', ',', ')'])[3]))
    epsilon = parse(Float64, split(lattice_params[9], '=')[2])
    dμ = parse(Float64, split(lattice_params[10], '=')[2])
    f_res = parse(Float64, split(lattice_params[13], '=')[2])
    rho_v = parse(Float64, split(lattice_params[14], '=')[2])
    t_max = ToInt64(String(split(lattice_params[17], '=')[2]))
    save_interval = ToInt64(String(split(lattice_params[19], '=')[2]))
    numb_of_snapshots = size(data)[1] ÷ x_size
    
    return data, x_size, y_size, epsilon, dμ, f_res, rho_v, t_max, save_interval, numb_of_snapshots
end