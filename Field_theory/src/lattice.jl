using Pkg
Pkg.activate(".")

using Random
using Kronecker
using LinearAlgebra
using Distributions
import Base.circshift


# Encodes the current state of the lattice in an Array of size Lx x Ly x 2 where Lx and Ly are the spacial 
# dimensions of the system and the first layer tracks the fraction of bonding state particles wheras the
# secon layer track the fraction of inert particles.
mutable struct lattice
    state::Array{Float64}
    boundaries::Dict{String, String}
    function lattice(state, boundaries)
        if length(size(state)) != 3 || size(state)[3] != 2
            throw(ArgumentError("The state must be a 3D array with the third dimension having size 2."))
        end
        if any([x < 0 for x in state]) 
            throw(ArgumentError("The state must be an array of non-negative integers."))
        end
        if any([x > 1 for x in state[:,:,1] + state[:,:,2]]) 
            throw(ArgumentError("At each lattice site the sum of the bonding and inert particles must be less than or equal to 1."))
        end
        if ((boundaries["upper_boundary"] == "periodic") ⊻ (boundaries["lower_boundary"] == "periodic")) || ((boundaries["left_boundary"] == "periodic") ⊻ (boundaries["right_boundary"] == "periodic"))
            @warn "You are using a periodic boundary without the opposing boundary being periodic. This is not recommended and can lead to wrong simulation results."
        end
        new(state, boundaries)
    end
end

# Alternative initiallization of the lattice
function lattice(;lattice_size::Tuple{Int64,Int64}=(10,10),
                    initialization::Union{String,Array}="random",    
                    upper_boundary::String="periodic",
                    lower_boundary::String="periodic",
                    left_boundary::String="periodic",
                    right_boundary::String="periodic")

    # initialize main lattice
    if initialization == "random"
        r = rand(Uniform(0,1),lattice_size...,3)
        s = sum(r,dims=3)
        l = (r ./ s)[:,:,1:2]  
    elseif initialization == "empty"
        l = zeros(Float64, lattice_size...,2)  
    else
        l = initialization         
        lattice_size = size(l[:,:,1])
    end

    boundaries = Dict("upper_boundary" => upper_boundary,
                        "lower_boundary" => lower_boundary,
                        "left_boundary" => left_boundary,
                        "right_boundary" => right_boundary)

    return lattice(l, boundaries)
end

(l::lattice)(i::Int64, j::Int64) = l.state[i,j,:]

function circshift(l::lattice, shift::Tuple{Int64,Int64})
    d = Dict("empty" => [0,0], "bonding" => [1,0], "inert" => [0,1])
    x_shift, y_shift = shift
    shifted = circshift(l.state, (x_shift, 0, 0))
    if x_shift > 0
        if l.boundaries["upper_boundary"] != "periodic"
            replace = zeros(Float64, x_shift, size(shifted,2), 2)
            replace[:,:,1] = d[l.boundaries["upper_boundary"]][1] * ones(size(replace[:,:,1]))
            replace[:,:,2] = d[l.boundaries["upper_boundary"]][2] * ones(size(replace[:,:,1]))
            shifted[1:x_shift, :, :] = replace
        end
    elseif x_shift < 0
        if l.boundaries["lower_boundary"] != "periodic"
            replace = zeros(Float64, -x_shift, size(shifted,2), 2)
            replace[:,:,1] = d[l.boundaries["lower_boundary"]][1] * ones(size(replace[:,:,1]))
            replace[:,:,2] = d[l.boundaries["lower_boundary"]][2] * ones(size(replace[:,:,1]))
            shifted[end+x_shift+1:end, :, :] = replace
        end
    end
    shifted = circshift(shifted, (0, y_shift, 0))
    if y_shift > 0
        if l.boundaries["left_boundary"] != "periodic"
            replace = zeros(Float64, size(shifted,1), y_shift, 2)
            replace[:,:,1] = d[l.boundaries["left_boundary"]][1] * ones(size(replace[:,:,2]))
            replace[:,:,2] = d[l.boundaries["left_boundary"]][2] * ones(size(replace[:,:,2]))
            shifted[:,1:y_shift,:] = replace
        end
    elseif y_shift < 0
        if l.boundaries["right_boundary"] != "periodic"
            replace = zeros(Float64, size(shifted,1), -y_shift, 2)
            replace[:,:,1] = d[l.boundaries["right_boundary"]][1] * ones(size(replace[:,:,2]))
            replace[:,:,2] = d[l.boundaries["right_boundary"]][2] * ones(size(replace[:,:,2]))
            shifted[:,end+y_shift+1:end,:] = replace
        end
    end
    return shifted
end


function update_lattice!(l::lattice, x_coord::Int64, y_coord::Int64, new_state::Vector{Float64})
    x_coord == 0 && (x_coord = size(l.state,1))
    y_coord == 0 && (y_coord = size(l.state,2))
    x_coord == size(l.state,1) + 1 && (x_coord = 1)
    y_coord == size(l.state,2) + 1 && (y_coord = 1)

    l.state[x_coord, y_coord,:] = new_state
end


function Δ(l::lattice)
    laplacian = -5 * l.state
    laplacian += 4/3 * (circshift(l, (1,0)) + circshift(l, (-1,0)) + circshift(l, (0,1)) + circshift(l, (0,-1)))
    laplacian += -1/12 * (circshift(l, (2,0)) + circshift(l, (-2,0)) + circshift(l, (0,2)) + circshift(l, (0,-2)))
    return laplacian[:,:,1]
end