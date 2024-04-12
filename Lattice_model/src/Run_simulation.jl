using DelimitedFiles

include("actions.jl")
include("Initialize_objects.jl")
include("unpack_ARGS.jl")
include("Save_parameters.jl")


function run_simulation(array_ID)
    # Unpack the arguments
    lattice_params, model_params, simulation_params = unpack_ARGS(array_ID)
    k = trans_rates(model_params)
    name = create_name(model_params["z_I"], model_params["z_B"], lattice_params["lattice_size"][1], lattice_params["lattice_size"][2])
    x_size = lattice_params["lattice_size"][1]
    y_size = lattice_params["lattice_size"][2]

    # Save the parameters
    save_parameters(lattice_params, model_params, simulation_params, name)

    # Initialize the lattice
    l = lattice(lattice_size=lattice_params["lattice_size"], 
                initialization=lattice_params["lattice_init"],  
                upper_boundary = lattice_params["upper_bound"], 
                lower_boundary = lattice_params["lower_bound"], 
                left_boundary = lattice_params["left_bound"], 
                right_boundary = lattice_params["right_bound"])

    # Initialize the heap
    heap = init_heap(l, k)
    dep_graph = init_dependency_graph(l)

    # Initialize the arrays to store the data
    trans_count = 0
    t = 0
    save_times = collect(0:simulation_params["save_interval"]:simulation_params["t_max"])
    numb_snapshots = length(save_times)
    states = Matrix{Int64}(undef, (x_size * numb_snapshots, y_size))
    save_count = 0
    
    # Run the simulation
    for i in 1:simulation_params["max_transitions"]
        trans_count += 1
        t = heap.nodes[1].time

        # Save the state at given times
        if t > save_times[save_count + 1]
            states[(save_count * x_size + 1):((save_count + 1) * x_size), :] = l.state
            save_count += 1
            save_count == numb_snapshots && break
        end

        # update the lattice
        top_handle = heap.nodes[1].handle
        act = handle_to_action(heap.nodes[1].handle, l)
        act(l)

        dep_handles = filter(!iszero,dep_graph[top_handle,:])
        old_rates = [heap.nodes[heap.node_map[handle]].trans_rate for handle in dep_handles]
        new_rates = Float64.([get_trans_rate(handle, l, k) for handle in dep_handles])

        # update the drawn node
        new_top_rate = get_trans_rate(top_handle, l, k)
        t_new = t + randexp() / new_top_rate
        update!(heap, top_handle, t_new, new_top_rate)

        # update the dependent nodes
        for (j,handle) in enumerate(dep_handles)
            if handle == 0 
                break
            elseif handle != top_handle
                heap_idx = heap.node_map[handle]
                if old_rates[j] == 0.0
                    t_new = t + randexp() / new_rates[j]
                else
                    t_new = old_rates[j]/new_rates[j] * (heap.nodes[heap_idx].time - t) + t
                end
                update!(heap, handle, t_new, new_rates[j])
            end
        end

        # Break if the simulation is finished
        if t > simulation_params["t_max"] 
            println("Simulation finished at t = $t since the maximum number of transitions was reached.")
            break
        end
    end

    # Save the data
    open("Data/" * name * "/states.txt", "w") do io
        writedlm(io, states)
    end
end