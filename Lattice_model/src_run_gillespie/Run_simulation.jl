using DelimitedFiles
using HDF5

include("actions.jl")
include("Initialize_objects.jl")
include("unpack_ARGS.jl")
include("../src_analysis/analyze_interface.jl")


function run_simulation(array_ID)
    # Unpack the arguments
    lattice_params, model_params, simulation_params, run_id, name = unpack_ARGS(array_ID)
    k = trans_rates(model_params)
    x_size = lattice_params.lattice_size[1]
    y_size = lattice_params.lattice_size[2]

    # Initialize the lattice
    l = lattice(lattice_size=lattice_params.lattice_size, 
                initialization=lattice_params.lattice_init,  
                upper_boundary = lattice_params.upper_bound, 
                lower_boundary = lattice_params.lower_bound, 
                left_boundary = lattice_params.left_bound, 
                right_boundary = lattice_params.right_bound)

    # Initialize the heap
    heap = init_heap(l, k)
    dep_graph = init_dependency_graph(l)
    dep_handles = zeros(Int64, length(dep_graph[1,:]))
    old_rates = zeros(Float64, length(dep_graph[1,:]))
    new_rates = zeros(Float64, length(dep_graph[1,:]))

    # Initialize the arrays to store the data
    trans_count = 0
    t = 0
    save_times = collect(0:simulation_params.save_interval:simulation_params.t_max)
    numb_snapshots = length(save_times)
    save_count = 0
    interface_pos = zeros(Float64, 2, numb_snapshots + 1)
    interface_pos[:,1] = [x_size / 4, 3 * x_size / 4]
    fourier_arr = zeros(Complex, 2, y_size)
    fourier_dummy = zeros(Float64, y_size)
    interface_profiles = zeros(Float64, 2, y_size)          # Dummy array for the interface profiles

    # set up file for saving the data
    dirname = "/scratch-local/Daniel.Pals"
    isdir(dirname) || mkdir(dirname)
    file_name = dirname * "/" * name * "_simulation_No_" * string(run_id)
    
    file = h5open(file_name, "w")

    create_group(file, "States")
    create_group(file, "Fourier_Data")
    create_group(file, "Interface_position")
    
    
    # Run the simulation
    max_transitions = simulation_params.max_transitions
    for i in 1:max_transitions
        trans_count += 1
        t = heap.nodes[1].time

        # Save the state at given times
        if t > save_times[save_count + 1]
            save_count += 1

            fft_and_interface_pos!(fourier_arr, interface_pos, interface_profiles, l.state, save_count)

            file["States/" * string(save_count)] = l.state
            fourier_dummy .= real.(fourier_arr[1,:])
            file["Fourier_Data/fourier_upper_real_" * string(save_count)] = fourier_dummy
            fourier_dummy .= imag.(fourier_arr[1,:])
            file["Fourier_Data/fourier_upper_imag_" * string(save_count)] = fourier_dummy
            fourier_dummy .= real.(fourier_arr[2,:])
            file["Fourier_Data/fourier_lower_real_" * string(save_count)] = fourier_dummy
            fourier_dummy .= imag.(fourier_arr[2,:])
            file["Fourier_Data/fourier_lower_imag_" * string(save_count)] = fourier_dummy

            save_count == numb_snapshots && break
        end

        # update the lattice
        top_handle = heap.nodes[1].handle
        act = handle_to_action(heap.nodes[1].handle, l)
        act(l)

        dep_handles .= @view dep_graph[top_handle,:]
        old_rates .= [(handle == 0) ? 0. : heap.nodes[heap.node_map[handle]].trans_rate for handle in dep_handles]
        new_rates .= [(handle == 0) ? 0. : Float64(get_trans_rate(handle, l, k)) for handle in dep_handles]

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
        if t > simulation_params.t_max 
            break
        end
    end

    file["Interface_position/positions"] = interface_pos[:, 2:end]

    close(file)

    mv(file_name, "Data/" * name * "/simulation_No_" * string(run_id), force=true)
    println(readdir(dirname))

end