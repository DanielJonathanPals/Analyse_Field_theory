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
mutable struct trans_rates
    BI::Function
    IB::Function
    BE::Function
    EB::Function
    IE::Function
    EI::Function
end


function trans_rates(args)
    df = log(args["z_I"]/args["z_B"])
    Δμ = args["dμ"]
    D = args["D"]
    z_B = args["z_B"]
    z_I = args["z_I"]

    k_IB(u) = D * args["k_IB"](df, Δμ, u)
    k_BI(u) = k_IB(u) * exp(df + Δμ + u)
    k_BE(u) = D * exp(u)
    k_EB(u) = D * z_B
    k_IE(u) = D
    k_EI(u) = D * z_I

    return trans_rates(k_BI, k_IB, k_BE, k_EB, k_IE, k_EI)
end

# returns the transition rate corresponding to the action `a` for the lattice `l`
function get_trans_rate(a::action, l::lattice, k::trans_rates)
    old_state = l(a.x_coord, a.y_coord)
    u = loc_energy(l, a.x_coord, a.y_coord)
    if old_state == 0
        if a.new_state == 1
            return k.EB(u)
        elseif a.new_state == 2
            return k.EI(u)
        elseif a.new_state == 0
            return 0.0
        end
    elseif old_state == 1
        if a.new_state == 2
            return k.BI(u)
        elseif a.new_state == 0
            return k.BE(u)
        elseif a.new_state == 1
            return 0.0
        end
    elseif old_state == 2
        if a.new_state == 1
            return k.IB(u)
        elseif a.new_state == 0
            return k.IE(u)
        elseif a.new_state == 2
            return 0.0
        end
    end
end


function get_trans_rate(handle::Int64, l::lattice, k::trans_rates)
    act = handle_to_action(handle, l)
    return get_trans_rate(act, l, k)
end