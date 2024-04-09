include("evolve_lattice.jl")
include("unpack_ARGS.jl")
include("save_parameters.jl")


function run_simulation(array_ID::Int64)
    # Unpack the arguments
    lattice_params, model_params, simulation_params = unpack_ARGS(array_ID)
    name = create_name(model_params["z_I"], 
                        model_params["z_B"], 
                        lattice_params["lattice_size"][1], 
                        lattice_params["lattice_size"][2], 
                        model_params["len_scale"], 
                        model_params["modify_laplace"])
    x_size = lattice_params["lattice_size"][1]
    y_size = lattice_params["lattice_size"][2]

    # Save the parameters
    save_parameters(lattice_params, model_params, simulation_params, name)

    # Initialize the lattice
    l = lattice(lattice_size = lattice_params["lattice_size"], 
                initialization = lattice_params["lattice_init"], 
                upper_boundary = lattice_params["upper_bound"], 
                lower_boundary = lattice_params["lower_bound"], 
                left_boundary = lattice_params["left_bound"], 
                right_boundary = lattice_params["right_bound"])

    # Initialize the arrays to store the data
    times = collect(0:simulation_params["dt"]:simulation_params["t_max"])
    save_times = collect(0:simulation_params["save_interval"]:simulation_params["t_max"])
    numb_snapshots = length(save_times)
    B_states = Matrix{Float64}(undef, (x_size * numb_snapshots, y_size))
    I_states = Matrix{Float64}(undef, (x_size * numb_snapshots, y_size))
    B_ratios = Vector{Float64}(undef, numb_snapshots)
    I_ratios = Vector{Float64}(undef, numb_snapshots)
    save_count = 0

    # Run the simulation
    for t in times
        println("t = $t")
        if t >= save_times[save_count + 1]
            B_states[(save_count * x_size + 1):((save_count + 1) * x_size), :] = l.state[:,:,1]
            I_states[(save_count * x_size + 1):((save_count + 1) * x_size), :] = l.state[:,:,2]
        end
        B_rat, I_rat = evolve_lattice!(l, simulation_params["dt"], model_params)
        if t >= save_times[save_count + 1]
            B_ratios[save_count + 1] = B_rat
            I_ratios[save_count + 1] = I_rat
            save_count += 1
        end
    end

    # Save the data
    open("Data/" * name * "/B_states.txt", "w") do io
        writedlm(io, B_states)
    end
    open("Data/" * name * "/I_states.txt", "w") do io
        writedlm(io, I_states)
    end
    open("Data/" * name * "/B_ratios.txt", "w") do io
        writedlm(io, B_ratios)
    end
    open("Data/" * name * "/I_ratios.txt", "w") do io
        writedlm(io, I_ratios)
    end
end