include("read_in_data.jl")
using FFTW

rounded_interface_positions = zeros(Int64, 2)

function analyze_column!(interface_profile::Matrix{Float64}, column::Vector{Int8}, rounded_interface_pos::Vector{Int}, column_idx::Int)

    if (column[rounded_interface_pos[1]] == 1) & (column[rounded_interface_pos[1] + 1] == 1)
        for j in 1:(rounded_interface_pos[1] - 1)
            if column[rounded_interface_pos[1] - j] != 1
                interface_profile[1, column_idx] = j
                break
            end
            if j == rounded_interface_pos[1] - 1
                interface_profile[1, column_idx] = j
            end
        end
    elseif (column[rounded_interface_pos[1]] != 1) & (column[rounded_interface_pos[1] + 1] != 1)
        for j in 1:(length(column) - rounded_interface_pos[1] - 1)
            if column[rounded_interface_pos[1] + 1 + j] == 1
                interface_profile[1, column_idx] = -j
                break
            end
            if j == length(column) - rounded_interface_pos[1] - 1
                interface_profile[1, column_idx] = -j
            end
        end
    end
    
    if (column[rounded_interface_pos[2]] == 1) & (column[rounded_interface_pos[2] + 1] == 1)
        for j in 1:(length(column) - rounded_interface_pos[1] - 1)
            if column[rounded_interface_pos[2] + 1 + j] != 1
                interface_profile[2, column_idx] = j
                break
            end
            if j == length(column) - rounded_interface_pos[2] - 1
                interface_profile[2, column_idx] = j
            end
        end
    elseif (column[rounded_interface_pos[2]] != 1) & (column[rounded_interface_pos[2] + 1] != 1)
        for j in 1:(rounded_interface_pos[2] - 1)
            if column[rounded_interface_pos[2] - j] == 1
                interface_profile[2, column_idx] = -j
                break
            end
            if j == rounded_interface_pos[2] - 1
                interface_profile[2, column_idx] = -j
            end
        end
    end
end

function get_interface_profiles!(interface_positions::Matrix{Float64}, interface_profile::Matrix{Float64}, state::Matrix{Int8}, idx::Int64)
    rounded_interface_positions .= Int.(round.(interface_positions[:, idx]))
    y_size = size(state, 2)
    for i in 1:y_size
        analyze_column!(interface_profile, state[:,i], rounded_interface_positions, i)
    end
    interface_positions[:, idx+1] .= rounded_interface_positions .+ (reshape(sum(interface_profile, dims = 2), 2) ./ y_size) .* [-1,1]
end

function fft_and_interface_pos!(fourier_arr::Matrix{Complex}, interface_positions::Matrix{Float64}, interface_profile::Matrix{Float64}, state::Matrix{Int8}, idx::Int64)
    get_interface_profiles!(interface_positions, interface_profile, state, idx)
    y_size = size(state, 2)
    fourier_arr[1, :] .= fft(interface_profile[1, :]) ./ √(y_size)
    fourier_arr[2, :] .= fft(interface_profile[2, :]) ./ √(y_size)
end


function expectation_hq2(file_name::String)
    files = read_dir("Data/" * file_name)

    # Test if all parameters in the files are the same
    x_size = zeros(Int, length(files))
    y_size = zeros(Int, length(files))
    epsilon = zeros(length(files))
    dμ = zeros(length(files))
    f_res = zeros(length(files))
    rho_v = zeros(length(files))
    t_max = zeros(length(files))
    save_interval = zeros(length(files))
    numb_of_snapshots = zeros(Int, length(files))
    for (i,file) in enumerate(files)
        x_size[i], y_size[i], epsilon[i], dμ[i], f_res[i], rho_v[i], t_max[i], save_interval[i], numb_of_snapshots[i] = read_params(file_name, ToInt64(file))
    end
    if !all(x_size .== x_size[1]) || !all(y_size .== y_size[1]) || !all(epsilon .== epsilon[1]) || !all(dμ .== dμ[1]) || !all(f_res .== f_res[1]) || !all(rho_v .== rho_v[1]) || !all(t_max .== t_max[1]) || !all(save_interval .== save_interval[1]) || !all(numb_of_snapshots .== numb_of_snapshots[1])
        throw(ErrorException("Not all parameters are the same in $file_name"))
        return nothing
    end

    all_fft_results = zeros(Float64, numb_of_snapshots[1], y_size[1], 4*length(files))
    for (i,file) in enumerate(files)
        all_fft_results[:, :, 4*i-3] = readdlm("Data/$file_name/$file/fourier_data_upper_boundary_real.txt", '\t', Float64, '\n')
        all_fft_results[:, :, 4*i-2] = readdlm("Data/$file_name/$file/fourier_data_upper_boundary_imag.txt", '\t', Float64, '\n')
        all_fft_results[:, :, 4*i-1] = readdlm("Data/$file_name/$file/fourier_data_lower_boundary_real.txt", '\t', Float64, '\n')
        all_fft_results[:, :, 4*i] = readdlm("Data/$file_name/$file/fourier_data_lower_boundary_imag.txt", '\t', Float64, '\n')
    end
    all_fft_results .= all_fft_results .^2
    average_hq2 = reshape(sum(all_fft_results, dims = 3), numb_of_snapshots[1], y_size[1]) ./ (2*length(files))

    open("Data/" * file_name * "/expectation_hq2.txt", "w") do io
        writedlm(io, average_hq2)
    end
end
