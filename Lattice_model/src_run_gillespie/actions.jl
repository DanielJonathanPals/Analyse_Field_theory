using StaticArrays

include("lattice.jl")


struct action
    x_coord::Int64
    y_coord::Int64
    new_state::Int64
end


# Applies the action `a` to the lattice `l`
function (a::action)(l::lattice)
    update_lattice!(l, a.x_coord, a.y_coord, Int8(a.new_state))
end


# For the datastructure it is easier to encode the actions into handles which essentially are integers.
function action_to_handle(a::action, l::lattice)
    handle = a.new_state * length(l.state) + (a.x_coord-1)*size(l.state,2) + a.y_coord
    return Int64(handle)
end


# This function decodes the handle `handle` into the corresponding action for the lattice `l`. Sinc the handels cannot distinguish between different types of diffusion, the type of
# diffusion has to be specified in the argument `type_of_diffusion`. If the handle corresponds to a reaction, the argument `type_of_diffusion` is ignored.
function handle_to_action(handle::Int64, l::lattice)
    new_state = (handle-1) ÷ length(l.state)
    x_coord = ((handle-1) % length(l.state)) ÷ size(l.state,2) + 1
    y_coord = ((handle-1) % length(l.state)) % size(l.state,2) + 1
    return action(x_coord, y_coord, new_state)
end


# Contains all transition rates as functions of the local potential u
struct trans_rates
    BI::SVector{5, Float64}
    IB::SVector{5, Float64}
    BE::SVector{5, Float64}
    EB::SVector{5, Float64}
    IE::SVector{5, Float64}
    EI::SVector{5, Float64}
end


function trans_rates(args)
    df = log(args.z_I/args.z_B)
    Δμ = args.dμ
    D = args.D
    z_B = args.z_B
    z_I = args.z_I
    epsilon = args.epsilon

    k_IB(n) = D * args.k_IB(df, Δμ, n * epsilon)
    k_BI(n) = k_IB(n) * exp(df + Δμ + n * epsilon)
    k_BE(n) = D * exp(n * epsilon)
    k_EB(n) = D * z_B
    k_IE(n) = D
    k_EI(n) = D * z_I

    k_IB_arr = @SVector [k_IB(0), k_IB(1), k_IB(2), k_IB(3), k_IB(4)]
    k_BI_arr = @SVector [k_BI(0), k_BI(1), k_BI(2), k_BI(3), k_BI(4)]
    k_BE_arr = @SVector [k_BE(0), k_BE(1), k_BE(2), k_BE(3), k_BE(4)]
    k_EB_arr = @SVector [k_EB(0), k_EB(1), k_EB(2), k_EB(3), k_EB(4)]
    k_IE_arr = @SVector [k_IE(0), k_IE(1), k_IE(2), k_IE(3), k_IE(4)]
    k_EI_arr = @SVector [k_EI(0), k_EI(1), k_EI(2), k_EI(3), k_EI(4)]

    return trans_rates(k_BI_arr, k_IB_arr, k_BE_arr, k_EB_arr, k_IE_arr, k_EI_arr)
end

# returns the transition rate corresponding to the action `a` for the lattice `l`
function get_trans_rate(a::action, l::lattice, k::trans_rates)

    old_state = l(a.x_coord, a.y_coord)
    B_neighbours = B_neighbours_count(l, a.x_coord, a.y_coord)
    if old_state == 0
        if a.new_state == 1
            return k.EB[B_neighbours+1]
        elseif a.new_state == 2
            return k.EI[B_neighbours+1]
        elseif a.new_state == 0
            return 0.0
        end
    elseif old_state == 1
        if a.new_state == 2
            return k.BI[B_neighbours+1]
        elseif a.new_state == 0
            return k.BE[B_neighbours+1]
        elseif a.new_state == 1
            return 0.0
        end
    elseif old_state == 2
        if a.new_state == 1
            return k.IB[B_neighbours+1]
        elseif a.new_state == 0
            return k.IE[B_neighbours+1]
        elseif a.new_state == 2
            return 0.0
        end
    end
end


function get_trans_rate(handle::Int64, l::lattice, k::trans_rates)
    act = handle_to_action(handle, l)
    return get_trans_rate(act, l, k)
end