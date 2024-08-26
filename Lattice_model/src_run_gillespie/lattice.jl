using Pkg
Pkg.activate(".")

using Random
using SparseArrays
using Kronecker
using LinearAlgebra


# Encodes the current state of the lattice
# E = 0, B = 1, I = 2
struct lattice
    state::Matrix{Int8}
    state_with_boundaries::Matrix{Int8}
    boundaries::Dict{String, String}
    function lattice(state, state_with_boundaries, boundaries)
        if !all([x in [0, 1, 2] for x in state])
            throw(ArgumentError("State must be a matrix of 0, 1, or 2. Here 0, 1, 2 corresponds to the lattice site being occupied by species E, B, or I respectively."))
        end
        if state_with_boundaries[2:end-1, 2:end-1] != state
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
                    initialization::Union{String,Matrix}="random",    
                    upper_boundary::String="periodic",
                    lower_boundary::String="periodic",
                    left_boundary::String="periodic",
                    right_boundary::String="periodic")

    # initialize main lattice
    if initialization == "random"
        l = rand(0:2,lattice_size...)    
    elseif initialization == "empty"
        l = zeros(Int8, lattice_size...)  
    else
        l = initialization         
        lattice_size = size(l)
    end

    boundaries = Dict("upper_boundary" => upper_boundary,
                        "lower_boundary" => lower_boundary,
                        "left_boundary" => left_boundary,
                        "right_boundary" => right_boundary)

    lwb = zeros(Int8, lattice_size .+ 2)       # lattice with boundaries
    lwb[2:end-1,2:end-1] = l                    # set inside of lattice with boundaries to lattice
    lwb[[1,1],[1,end],[end,1],[end,end]] .= 0   # set corner boundaries

    d = Dict("empty" => 0, "bonding" => 1, "inert" => 2)

    # set upper boundary
    if upper_boundary == "periodic"
        lwb[1,2:end-1] = l[end,:]              
    else
        lwb[1,2:end-1] .= d[upper_boundary]
    end

    # set lower boundary
    if lower_boundary == "periodic"
        lwb[end,2:end-1] = l[1,:]              
    else
        lwb[end,2:end-1] .= d[lower_boundary]
    end     

    # set left boundary
    if left_boundary == "periodic"
        lwb[2:end-1,1] = l[:,end]              
    else
        lwb[2:end-1,1] .= d[left_boundary]
    end

    # set right boundary
    if right_boundary == "periodic"
        lwb[2:end-1,end] = l[:,1]              
    else
        lwb[2:end-1,end] .= d[right_boundary]
    end

    return lattice(l, lwb, boundaries)
end

# get the state of the lattice site at row i, column j``
(l::lattice)(i::Int64, j::Int64) = l.state_with_boundaries[i+1,j+1]

# returns the naighbours of the lattice site at row i, column j as an array [upper neighbor, right neighbor, lower neighbor, left neighbor]
function neighbours(l::lattice, i::Int64, j::Int64)
    return [l(i-1,j), l(i,j+1), l(i+1,j), l(i,j-1)]
end


# returns the number of B-neighbours of the lattice site at row i, column j
function B_neighbours_count(l::lattice, i::Int64, j::Int64)
    count = 0
    # for some reason not using a for loop allocates less memory
    if l(i,j-1) == 1
        count += 1
    end
    if l(i,j+1) == 1
        count += 1
    end
    if l(i-1,j) == 1
        count += 1
    end
    if l(i+1,j) == 1
        count += 1
    end
    return count
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


function update_lattice!(l::lattice, x_coord::Int64, y_coord::Int64, new_state::Int8)
    x_coord == 0 && (x_coord = size(l.state,1))
    y_coord == 0 && (y_coord = size(l.state,2))
    x_coord == size(l.state,1) + 1 && (x_coord = 1)
    y_coord == size(l.state,2) + 1 && (y_coord = 1)

    l.state[x_coord, y_coord] = new_state
    # update lattice with boundaries
    if x_coord == 1 && l.boundaries["lower_boundary"] == "periodic"
        l.state_with_boundaries[end, y_coord+1] = new_state
    elseif x_coord == size(l.state,1) && l.boundaries["upper_boundary"] == "periodic"
        l.state_with_boundaries[1, y_coord+1] = new_state
    end
    if y_coord == 1 && l.boundaries["right_boundary"] == "periodic"
        l.state_with_boundaries[x_coord+1, end] = new_state
    end
    if y_coord == size(l.state,2) && l.boundaries["left_boundary"] == "periodic"
        l.state_with_boundaries[x_coord+1, 1] = new_state
    end
    l.state_with_boundaries[x_coord+1, y_coord+1] = new_state
end


# returns the local potential energies at each lattice site
function loc_energy(l::lattice; epsilon::Float64=-2.95)
    lwb = copy(l.state_with_boundaries)
    lwb = sparse(Int64.(iszero.(lwb .- 1)))
    lwb = circshift(lwb, (0,1)) + circshift(lwb, (0,-1)) + circshift(lwb, (1,0)) + circshift(lwb, (-1,0))
    u = epsilon * lwb[2:end-1, 2:end-1]
    return u
end


function loc_energy(l::lattice, x_coord::Int64, y_coord::Int64; epsilon::Float64=-2.95)
    n = neighbours(l, x_coord, y_coord)
    B_neighbours = sum(Int64.(iszero.(n .- 1)))
    return epsilon * B_neighbours
end