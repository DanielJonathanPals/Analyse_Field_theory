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
    else
        interface_profile[1, column_idx] = 0
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
    else
        interface_profile[2, column_idx] = 0
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