include("lattice.jl")
include("unpack_ARGS.jl")


function deterministic_evolution_matrix(k_IB::Float64, k_BI::Float64, z_B::Float64, z_I::Float64, expon::Float64)
    return [-(k_BI + expon) k_IB z_B; k_BI -(k_IB + 1) z_I]    
end

function stochastic_evolution_matrix(k_IB::Float64, k_BI::Float64, z_B::Float64, z_I::Float64, expon::Float64, n_B::Float64, n_I::Float64)
    n_E = 1 - n_B - n_I
    a_11 = (k_BI + expon)*n_B + k_IB*n_I + z_B*n_E
    a_22 = k_BI*n_B + (k_IB + 1)*n_I + z_I*n_E
    a_33 = expon*n_B + n_I + (z_B + z_I)*n_E
    a_12 = -k_BI*n_B - k_IB*n_I
    a_13 = -z_B*n_E - expon*n_B
    a_23 = -z_I*n_E - n_I
    A = [a_11 a_12 a_13; a_12 a_22 a_23; a_13 a_23 a_33]
    C = hcat(zeros(3), sqrt(eigvals(A)[2]) .* eigvecs(A)[:,2],sqrt(eigvals(A)[3]) .* eigvecs(A)[:,3])
    return C[1:2,2:3]
end


function evolve_lattice!(l::lattice, dt::Float64, model_params::Dict, len_scale::Int64, modify_laplace::Bool = false)
    df = log(model_params["z_I"]/model_params["z_B"])
    dμ = model_params["dμ"]
    epsilon = model_params["epsilon"]
    D = model_params["D"]
    z_B = model_params["z_B"]
    z_I = model_params["z_I"]

    modify_laplace ? a = len_scale^2 : a = 1
    u_arr = epsilon * (l.state + a * Δ(l))

    for i in 1:size(l.state,1), j in 1:size(l.state,2)
        u = u_arr[i,j]
        n_B = l(i,j)[1]
        n_I = l(i,j)[2]
        n_E = 1 - n_B - n_I
        k_IB = model_params["k_IB"](df, dμ, u)
        k_BI = k_IB * exp(df + dμ + u)
        expon = exp(u)
        A = deterministic_evolution_matrix(k_IB, k_BI, z_B, z_I, expon)
        C = stochastic_evolution_matrix(k_IB, k_BI, z_B, z_I, expon, n_B, n_I) / len_scale
        dW = randn(2) * sqrt(dt)
        new_state = l.state[i,j,:] + dt * (D * A * vcat(n_B,n_I,n_E)) + √(D) * C * dW
        if (new_state[1] < 0) | (new_state[1] > 1) | (new_state[2] < 0) | (new_state[2] > 1) | (new_state[1] + new_state[2] > 1)
            new_state = l.state[i,j,:]
        end
        update_lattice!(l, i, j, new_state)
    end
end
