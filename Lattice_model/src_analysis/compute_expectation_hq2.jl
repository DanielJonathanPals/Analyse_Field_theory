import Pkg
Pkg.activate("../")

using HDF5
using DelimitedFiles

ToInt64(str::String) = Int64(parse(Float64, str))


function compute_expectation_hq2(file_name::String)
    isdir("Data/" * file_name * "/expectation_hq2") || mkdir("Data/" * file_name * "/expectation_hq2")

    lattice_params = readdlm("Data/$file_name/lattice_params.txt", '\t', String, '\n')
    L = ToInt64(string(split(lattice_params[2], [' ', ')'])[end-1]))
    t_max = parse(Float64, split(lattice_params[16], " = ")[end])
    save_interval = parse(Float64, split(lattice_params[18], " = ")[end])
    t = collect(0:save_interval:t_max)

    hq2 = zeros(Float64, L, length(t))

    simulation_file_names = readdir("Data/" * file_name)
    filter!(s->occursin(r"simulation_No", s),simulation_file_names)

    for simulation_file in simulation_file_names
        simulation_file = "Data/" * file_name * "/" * simulation_file
        fourier_data = h5read(simulation_file, "Fourier_Data")
        for i in 1:length(t)
            hq2[:,i] .+= (fourier_data["fourier_upper_real_" * string(i)] .^ 2 + fourier_data["fourier_upper_imag_" * string(i)] .^ 2 + fourier_data["fourier_lower_real_" * string(i)] .^ 2 + fourier_data["fourier_lower_imag_" * string(i)] .^ 2) ./ (2 * length(simulation_file_names))
        end
    end
    
    expec_file = h5open("Data/" * file_name * "/expectation_hq2/expectation_hq2", "w")
    create_group(expec_file, "Fixed_time")
    create_group(expec_file, "Fixed_q")

    for i in 1:length(t)
        expec_file["Fixed_time/" * string(i)] = hq2[:,i]
    end

    for i in 1:L
        expec_file["Fixed_q/" * string(i-1)] = hq2[i,:]
    end

    close(expec_file)
end


file_names = readdir("Data")
filter!(s->occursin(r"eps", s),file_names)

for file_name in file_names
    println("Woring on $file_name...")
    compute_expectation_hq2(file_name)
end