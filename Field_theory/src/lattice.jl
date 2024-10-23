using Pkg
Pkg.activate(".")

using Random
using Kronecker
using LinearAlgebra
using Distributions
import Base.circshift
using StaticArrays


# All object that serve as containers are a subtype of the abstract type container
abstract type Container end


# Matrix container which has the same size as the state matrix of the lattice
struct StateContainer <: Container
    state::Matrix{Float64}
end


# Container to store small matracies of size 3x3 and 2x2
struct MatrixContainer <: Container
    matrix::Matrix{Float64}
end


# Container to store small vectors
struct VectorContainer <: Container
    vector::Vector{Float64}
end


# Encodes the current state of the lattice in two matracies of size Lx x Ly where Lx and Ly are the spacial 
# dimensions of the system and the first matrix tracks the fraction of bonding state particles wheras the
# secon matrix track the fraction of inert particles.
struct Lattice
    B_state::Matrix{Float64}
    I_state::Matrix{Float64}
    upper_bound::String
    lower_bound::String
    left_bound::String
    right_bound::String
    function Lattice(B_state::Matrix{Float64}, I_state::Matrix{Float64}, upper_bound::String, lower_bound::String, left_bound::String, right_bound::String)
        if ((upper_bound == "periodic") ⊻ (lower_bound == "periodic")) || ((left_bound == "periodic") ⊻ (right_bound == "periodic"))
            @warn "You are using a periodic boundary without the opposing boundary being periodic. This is not recommended and can lead to wrong simulation results."
        end
        if size(B_state) != size(I_state)
            throw(ArgumentError("The size of the bonding and inert state matrices must be the same."))
        end
        new(B_state, I_state, upper_bound, lower_bound, left_bound, right_bound)
    end
end


(l::Lattice)(i::Int64, j::Int64) = l.B_state[i,j], l.I_state[i,j]

const d = Dict("empty" => [0.,0.], "bonding" => [1.,0.], "inert" => [0.,1.])

function circshift_B!(l::Lattice, shift::Tuple{Int64,Int64}, c_B::StateContainer)
    x_shift, y_shift = shift
    c_B.state[:,:] = l.B_state
    c_B.state[:,:] = circshift(c_B.state, (x_shift, 0))
    x_size, y_size = size(l.B_state)
    if x_shift > 0
        if l.upper_bound != "periodic"
            c_B.state[1:x_shift, :] = d[l.upper_bound][1] * ones(Float64, x_shift, y_size)
        end
    elseif x_shift < 0
        if l.lower_bound != "periodic"
            c_B.state[end+x_shift+1:end, :] = d[l.lower_bound][1] * ones(Float64, -x_shift, y_size)
        end
    end
    c_B.state[:,:] = circshift(c_B.state, (0, y_shift))
    if y_shift > 0
        if l.left_bound != "periodic"
            c_B.state[:,1:y_shift] = d[l.left_bound][1] * ones(Float64, x_size, y_shift)
        end
    elseif y_shift < 0
        if l.right_bound != "periodic"
            c_B.state[:,end+y_shift+1:end] = d[l.right_bound][1] * ones(Float64, x_size, -y_shift)
        end
    end
end


function update_lattice!(l::Lattice, x_coord::Int64, y_coord::Int64, new_state::Vector{Float64})
    x_coord == 0 && (x_coord = size(l.B_state,1))
    y_coord == 0 && (y_coord = size(l.B_state,2))
    x_coord == size(l.B_state,1) + 1 && (x_coord = 1)
    y_coord == size(l.B_state,2) + 1 && (y_coord = 1)

    l.B_state[x_coord, y_coord] = new_state[1]
    l.I_state[x_coord, y_coord] = new_state[2]
end

function update_lattice!(l::Lattice, x_coord::Int64, y_coord::Int64, new_state::SArray{Tuple{2},Float64,1,2})
    update_lattice!(l, x_coord, y_coord, Array(new_state))
end


# The laplace_container is modified in place to be the laplacian of the state matrix of the lattice
function Δ!(l::Lattice, c_B::StateContainer, laplace_container::StateContainer)
    laplace_container.state .= -5 .* l.B_state
    for tup in [(0,1), (0,-1), (1,0), (-1,0)]
        circshift_B!(l, tup, c_B)
        laplace_container.state .+= 4/3 .* c_B.state
    end
    for tup in [(0,2), (0,-2), (2,0), (-2,0)]
        circshift_B!(l, tup, c_B)
        laplace_container.state .+= -1/12 .* c_B.state
    end
end