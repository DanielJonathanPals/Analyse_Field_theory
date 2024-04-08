include("Heap.jl")


# Initialize nodes.
function init_nodes(l::lattice, k::trans_rates)
    handles = []
    handles = collect(1:length(l.state) * 3)

    nodes = Vector{MutableBinaryHeapNode}(undef, length(handles))
    times = zeros(length(handles))
    for (i,handle) in enumerate(handles)
        rate = get_trans_rate(handle, l, k)
        times[i] = randexp() / rate
        nodes[i] = MutableBinaryHeapNode(times[i], handle, rate)
    end
    return nodes
end


function init_heap(l::lattice, k::trans_rates = trans_rates(default_params))
    nodes = init_nodes(l, k)
    heap = _make_mutable_binary_heap(nodes)
    return heap
end


function find_dependent_handles(handle::Int64, l::lattice)
    act = handle_to_action(handle, l)
    handles = []
    x = act.x_coord
    y = act.y_coord

    has_nbr, x_coord, y_coord = neighbour_coords(l, x, y)
    for c in 0:2
        push!(handles, action_to_handle(action(x, y, c), l))
        for i in 1:4
            if has_nbr[i]
                push!(handles, action_to_handle(action(x_coord[i],y_coord[i], c), l))
            end
        end
    end

    return handles
end


# returns an array where the i-th row contains the handles of the actions that depend on the i-th handle
function init_dependency_graph(l::lattice)
    
    graph = zeros(Int64, length(l.state) * 3, 15)                   # Each handle can have at most 15 dependent handles
    
    for handle in 1:length(l.state) * 3
        dependent_handles = find_dependent_handles(handle, l)
        graph[handle, :] = vcat(dependent_handles, zeros(Int64, 15-length(dependent_handles)))
    end
    return graph
end