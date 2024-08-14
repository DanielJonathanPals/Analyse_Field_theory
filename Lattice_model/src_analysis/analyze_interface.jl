include("read_in_data.jl")
using Plots
using FFTW

function analyze_column(column::Vector{Int8}, rounded_interface_pos::Vector{Int})
    local_pos = zeros(Int, 2)

    if (column[rounded_interface_pos[1]] == 1) & (column[rounded_interface_pos[1] + 1] == 1)
        for j in 1:(rounded_interface_pos[1] - 1)
            if column[rounded_interface_pos[1] - j] != 1
                local_pos[1] = j
                break
            end
            if j == rounded_interface_pos[1] - 1
                local_pos[1] = j
            end
        end
    elseif (column[rounded_interface_pos[1]] != 1) & (column[rounded_interface_pos[1] + 1] != 1)
        for j in 1:(length(column) - rounded_interface_pos[1] - 1)
            if column[rounded_interface_pos[1] + 1 + j] == 1
                local_pos[1] = -j
                break
            end
            if j == length(column) - rounded_interface_pos[1] - 1
                local_pos[1] = -j
            end
        end
    end
    
    if (column[rounded_interface_pos[2]] == 1) & (column[rounded_interface_pos[2] + 1] == 1)
        for j in 1:(length(column) - rounded_interface_pos[1] - 1)
            if column[rounded_interface_pos[2] + 1 + j] != 1
                local_pos[2] = j
                break
            end
            if j == length(column) - rounded_interface_pos[2] - 1
                local_pos[2] = j
            end
        end
    elseif (column[rounded_interface_pos[2]] != 1) & (column[rounded_interface_pos[2] + 1] != 1)
        for j in 1:(rounded_interface_pos[2] - 1)
            if column[rounded_interface_pos[2] - j] == 1
                local_pos[2] = -j
                break
            end
            if j == rounded_interface_pos[2] - 1
                local_pos[2] = -j
            end
        end
    end
    return local_pos
end

function get_interface_profiles(data::Matrix{Int8}, x_size::Int, y_size::Int, numb_of_snapshot::Int, interface_positions::Vector{Float64})
    interface_profiles = zeros(Float64, 2, y_size)
    rounded_interface_positions = Int.(round.(interface_positions))

    for i in 1:y_size
        column = data[(numb_of_snapshot - 1) * x_size + 1:numb_of_snapshot * x_size, i]
        interface_profiles[:, i] = analyze_column(column, rounded_interface_positions)
    end
    new_interface_pos = rounded_interface_positions + (reshape(sum(interface_profiles, dims = 2), 2) ./ y_size)
    return interface_profiles, new_interface_pos
end

function run_analysis(file_name::String, run_id::Int)
    data, x_size, y_size, epsilon, dμ, f_res, rho_v, t_max, save_interval, numb_of_snapshots = read_data(file_name, run_id)

    interface_positions_init = [x_size / 4, 3 * x_size / 4]
    interface_positions = interface_positions_init

    fourier1 = zeros(Complex, numb_of_snapshots, y_size)
    fourier2 = zeros(Complex, numb_of_snapshots, y_size)
    save_interface_pos1 = zeros(Float64, numb_of_snapshots)
    save_interface_pos2 = zeros(Float64, numb_of_snapshots)

    for i in 1:numb_of_snapshots
        interface_profiles, new_interface_pos = get_interface_profiles(data, x_size, y_size, i, interface_positions)
        interface_positions = new_interface_pos
        save_interface_pos1[i] = interface_positions[1] - interface_positions_init[1]
        save_interface_pos2[i] = interface_positions[2] - interface_positions_init[2]
        fourier1[i, :] = fft(interface_profiles[1, :])
        fourier2[i, :] = fft(interface_profiles[2, :])
    end

    open("Data/" * file_name * "/" * string(run_id) * "/fourier_data_upper_boundary.txt", "w") do io
        writedlm(io, fourier1)
    end

    open("Data/" * file_name * "/" * string(run_id) * "/fourier_data_lower_boundary.txt", "w") do io
        writedlm(io, fourier2)
    end

    open("Data/" * file_name * "/" * string(run_id) * "/interface_pos_upper_boundary.txt", "w") do io
        writedlm(io, save_interface_pos1)
    end

    open("Data/" * file_name * "/" * string(run_id) * "/interface_pos_lower_boundary.txt", "w") do io
        writedlm(io, save_interface_pos2)
    end
end

function run_analysis(file_name::String)
    files = readdir("Data/" * file_name)
    for file in files
        run_analysis(file_name, ToInt64(file))
    end
end
