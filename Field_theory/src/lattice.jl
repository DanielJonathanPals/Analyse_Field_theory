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
    state_with_boundaries::Array{Float64}
    boundaries::Dict{String, String}
    function lattice(state, state_with_boundaries, boundaries)
        if length(size(state)) != 3 || size(state)[3] != 2
            throw(ArgumentError("The state must be a 3D array with the third dimension having size 2."))
        end
        if any([x < 0 for x in state]) 
            throw(ArgumentError("The state must be an array of non-negative integers."))
        end
        if any([x > 1 for x in state[:,:,1] + state[:,:,2]]) 
            throw(ArgumentError("At each lattice site the sum of the bonding and inert particles must be less than or equal to 1."))
        end
        if state_with_boundaries[2:end-1, 2:end-1, :] != state
            throw(ArgumentError("The inside of `state_with_boundaries` must be equal to `state`."))
        end
        if ((boundaries["upper_boundary"] == "periodic") ⊻ (boundaries["lower_boundary"] == "periodic")) || ((boundaries["left_boundary"] == "periodic") ⊻ (boundaries["right_boundary"] == "periodic"))
            @warn "You are using a periodic boundary without the opposing boundary being periodic. This is not recommended and can lead to wrong simulation results."
        end
        new(state, state_with_boundaries, boundaries)
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

    lwb = zeros(Float64, ((lattice_size .+ 2)...,2))    # lattice with boundaries
    lwb[2:end-1,2:end-1,:] = l                          # set inside of lattice with boundaries to lattice

    d = Dict("empty" => [0,0], "bonding" => [1,0], "inert" => [0,1])

    # set upper boundary
    if upper_boundary == "periodic"
        lwb[1,2:end-1,:] = l[end,:,:]              
    else
        lwb[1,2:end-1,:] .= d[upper_boundary]' ⊗ ones(lattice_size[2])
    end

    # set lower boundary
    if lower_boundary == "periodic"
        lwb[end,2:end-1,:] = l[1,:,:]              
    else
        lwb[end,2:end-1,:] .= d[lower_boundary]' ⊗ ones(lattice_size[2])
    end     

    # set left boundary
    if left_boundary == "periodic"
        lwb[2:end-1,1,:] = l[:,end,:]              
    else
        lwb[2:end-1,1,:] .= d[left_boundary]' ⊗ ones(lattice_size[1])
    end

    # set right boundary
    if right_boundary == "periodic"
        lwb[2:end-1,end,:] = l[:,1,:]              
    else
        lwb[2:end-1,end,:] .= d[right_boundary]' ⊗ ones(lattice_size[1])
    end

    return lattice(l, lwb, boundaries)
end


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
            shifted[end-x_shift+1:end, :, :] = replace
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
            shifted[:,end-y_shift+1:end,:] = replace
        end
    end
    return shifted
end

# get the state of the lattice site at row i, column j``
(l::lattice)(i::Int64, j::Int64) = l.state_with_boundaries[i+1,j+1,:]

# returns the naighbours of the lattice site at row i, column j as an array [upper neighbor, right neighbor, lower neighbor, left neighbor]
function neighbours(l::lattice, i::Int64, j::Int64)
    return hcat(l(i-1,j), l(i,j+1), l(i+1,j), l(i,j-1))'
end


# returns true if the lattice site at row i, column j has a neighbour in the specified direction and returns the coordinates of that neighbour
function has_neighbour(l::lattice, i::Int64, j::Int64, rel_pos::String = "above")
    if rel_pos == "above"
        if i != 1
            return true, (i-1,j)
        elseif l.boundaries["upper_boundary"] == "periodic"
            return true, (size(l.state,1),j)
        else 
            return false, (0,0)
        end

    elseif rel_pos == "right"
        if j != size(l.state,2)
            return true, (i,j+1)
        elseif l.boundaries["right_boundary"] == "periodic"
            return true, (i,1)
        else 
            return false, (0,0)
        end

    elseif rel_pos == "lower"
        if i != size(l.state,1)
            return true, (i+1,j)
        elseif l.boundaries["lower_boundary"] == "periodic"
            return true, (1,j)
        else 
            return false, (0,0)
        end

    elseif rel_pos == "left"
        if j != 1
            return true, (i,j-1)
        elseif l.boundaries["left_boundary"] == "periodic"
            return true, (i,size(l.state,2))
        else 
            return false, (0,0)
        end
    end
end


# returns the coordinates of the neighbours of the lattice site at row i, column j as an array of tuples [(upper neighbor), (right neighbor), (lower neighbor), (left neighbor)]
function neighbour_coords(l::lattice, i::Int64, j::Int64)
    has_neighbor = Vector{Bool}(undef, 4)
    x_coords = Vector{Int64}(undef, 4)
    y_coords = Vector{Int64}(undef, 4)
    for (idx,rel_pos) in enumerate(["above", "right", "lower", "left"])
        has_neighbor[idx], (x_coords[idx], y_coords[idx]) = has_neighbour(l, i, j, rel_pos)
    end

    return has_neighbor, x_coords, y_coords
end


function update_lattice!(l::lattice, x_coord::Int64, y_coord::Int64, new_state::Vector{Float64})
    x_coord == 0 && (x_coord = size(l.state,1))
    y_coord == 0 && (y_coord = size(l.state,2))
    x_coord == size(l.state,1) + 1 && (x_coord = 1)
    y_coord == size(l.state,2) + 1 && (y_coord = 1)

    l.state[x_coord, y_coord,:] = new_state
    # update lattice with boundaries
    if x_coord == 1 && l.boundaries["lower_boundary"] == "periodic"
        l.state_with_boundaries[end, y_coord+1,:] = new_state
    elseif x_coord == size(l.state,1) && l.boundaries["upper_boundary"] == "periodic"
        l.state_with_boundaries[1, y_coord+1,:] = new_state
    end
    if y_coord == 1 && l.boundaries["right_boundary"] == "periodic"
        l.state_with_boundaries[x_coord+1, end,:] = new_state
    end
    if y_coord == size(l.state,2) && l.boundaries["left_boundary"] == "periodic"
        l.state_with_boundaries[x_coord+1, 1,:] = new_state
    end
    l.state_with_boundaries[x_coord+1, y_coord+1,:] = new_state
end


function Δ(l::lattice)
    laplacian = -5 * l.state
    laplacian += 4/3 * (circshift(l, (1,0)) + circshift(l, (-1,0)) + circshift(l, (0,1)) + circshift(l, (0,-1)))
    laplacian += -1/12 * (circshift(l, (2,0)) + circshift(l, (-2,0)) + circshift(l, (0,2)) + circshift(l, (0,-2)))
    return laplacian
end