using DelimitedFiles

function save_parameters(lattice_params, model_params, simulation_params, name)
    isdir("Data") || mkdir("Data")
    isdir("Data/" * name) || mkdir("Data/" * name)

    input = ["Lattice Parameters:",
            "Lattice size = $(lattice_params["lattice_size"])",
            "The upper boundary is $(lattice_params["upper_bound"])",
            "The lower boundary is $(lattice_params["lower_bound"])",
            "The left boundary is $(lattice_params["left_bound"])",
            "The right boundary is $(lattice_params["right_bound"])",
            "length scale = $(lattice_params["len_scale"])",
            "",
            "Model Parameters:",
            "epsilon = $(model_params["epsilon"])",
            "dμ = $(model_params["dμ"])",
            "z_I = $(model_params["z_I"])",
            "z_B = $(model_params["z_B"])",
            "D = $(model_params["D"])",
            "",
            "Simulation Parameters:",
            "t_max = $(simulation_params["t_max"])",
            "max_transitions = $(simulation_params["max_transitions"])",
            "save_interval = $(simulation_params["save_interval"])"]

    open("Data/" * name * "/lattice_params.txt", "w") do io
        writedlm(io, input)
    end
end