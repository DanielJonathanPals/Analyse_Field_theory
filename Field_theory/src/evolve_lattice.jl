include("lattice.jl")


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
    evals = eigvals(A)
    evecs = eigvecs(A)
    C = hcat(zeros(3), sqrt(evals[2]) .* evecs[:,2],sqrt(evals[3]) .* evecs[:,3])
    return C[1:2,2:3]
end


function evolve_lattice!(l::lattice, dt::Float64, model_params::Dict)
    df = log(model_params["z_I"]/model_params["z_B"])
    dμ = model_params["dμ"]
    epsilon = model_params["epsilon"]
    D = model_params["D"]
    z_B = model_params["z_B"]
    z_I = model_params["z_I"]
    modify_laplace = model_params["modify_laplace"]
    len_scale = model_params["len_scale"]

    modify_laplace ? a = len_scale^2 : a = 1
    u_arr = epsilon * (4 * l.state[:,:,1] + a * Δ(l) / len_scale^2)                 # need to devide laplace by len_scale^2 to get the correct units

    noise_ratios = zeros(Float64, size(l.state,1), size(l.state,2), 2)

    for idx in CartesianIndices(l.state)
        i, j = Tuple(idx)
        u = u_arr[i,j]
        n_B = l(i,j)[1]
        n_I = l(i,j)[2]
        n_E = 1 - n_B - n_I
        k_IB = model_params["k_IB"](df, dμ, u)
        k_BI = k_IB * exp(df + dμ + u)
        expon = exp(u)
        A = deterministic_evolution_matrix(k_IB, k_BI, z_B, z_I, expon)
        C = stochastic_evolution_matrix(k_IB, k_BI, z_B, z_I, expon, n_B, n_I) / len_scale
        det = dt * (D * A * vcat(n_B,n_I,n_E))
        stoch = √(D) * [C[1,1], C[2,2]]
        dW = randn(2) * sqrt(dt)
        u = l.state[i,j,:]
        v = det + stoch .* dW
        new_state = u + v
        # If the new state is outside the boundaries, project it back onto the boundary
        if (new_state[1] < 0) | (new_state[2] < 0) | (sum(new_state) > 1)
            λ_arr = [-u[2]/v[2], -u[1]/v[1], (1-u[1]-u[2])/(v[1]+v[2])]
            λ_arr[λ_arr .< 0] .= Inf
            λ = min(λ_arr...)
            new_state = u + 0.99 * λ * v
            if (new_state[1] < 0) | (new_state[2] < 0) | (sum(new_state) > 1)
                new_state = u
            end
        end
        update_lattice!(l, i, j, new_state)
        noise_ratios[i,j,:] = [det[i] / stoch[i] > 0.1 ? 1 : 0 for i in 1:2]
    end
    return reshape(sum(noise_ratios, dims = [1,2]), 2) ./ length(noise_ratios)
end
