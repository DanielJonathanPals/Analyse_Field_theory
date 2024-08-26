struct MutableBinaryHeapNode
    time::Float64
    handle::Int
    trans_rate::Float64
end


mutable struct MutableBinaryHeap
    nodes::Vector{MutableBinaryHeapNode}
    node_map::Vector{Int}
end


function _heap_bubble_up!(nodes::Vector{MutableBinaryHeapNode}, nodemap::Vector{Int}, nd_id::Int)

    nd = nodes[nd_id]
    t = nd.time

    swapped = true  # whether swap happens at last step
    i = nd_id

    while swapped && i > 1  # nd is not root
        p = i >> 1          # parent index
        nd_p = nodes[p]

        if t < nd_p.time
            # move parent downward
            nodes[i] = nd_p
            nodemap[nd_p.handle] = i
            i = p
        else
            swapped = false
        end
    end

    if i != nd_id
        nodes[i] = nd
    end
    nodemap[nd.handle] = i
end


function _heap_bubble_down!(nodes::Vector{MutableBinaryHeapNode}, nodemap::Vector{Int}, nd_id::Int)

    nd = nodes[nd_id]
    t = nd.time

    n = length(nodes)
    last_parent = n >> 1

    swapped = true
    i = nd_id

    while swapped && i <= last_parent
        il = i << 1     # left child index

        if il < n   # contains both left and right children
            ir = il + 1     # right child index

            # determine the better child
            nd_l = nodes[il]
            nd_r = nodes[ir]

            if nd_r.time < nd_l.time
                # consider right child
                if nd_r.time < t
                    nodes[i] = nd_r
                    nodemap[nd_r.handle] = i
                    i = ir
                else
                    swapped = false
                end
            else
                # consider left child
                if nd_l.time < t
                    nodes[i] = nd_l
                    nodemap[nd_l.handle] = i
                    i = il
                else
                    swapped = false
                end
            end

        else  # contains only left child
            nd_l = nodes[il]
            if nd_l.time < t
                nodes[i] = nd_l
                nodemap[nd_l.handle] = i
                i = il
            else
                swapped = false
            end
        end
    end

    if i != nd_id
        nodes[i] = nd
        nodemap[nd.handle] = i
    end
end


function _make_mutable_binary_heap(nodes::Vector{MutableBinaryHeapNode})
    # make a static binary index tree from a list of values

    n = length(nodes)
    nodemap = zeros(Int, max([n.handle for n in nodes]...))

    for i = 1 : n
        _heap_bubble_up!(nodes, nodemap, i)
    end
    return MutableBinaryHeap(nodes, nodemap)
end


function Base.show(io::IO, h::MutableBinaryHeap)
    print(io, "MutableBinaryHeap(")
    nodes = h.nodes
    n = length(nodes)
    if n > 0
        print(io, string(nodes[1].time))
        for i = 2 : n
            print(io, ", $(nodes[i].time)")
        end
    end
    print(io, ")")
end


function update!(h::MutableBinaryHeap, handle::Int64, t_new::Float64, new_rate::Float64) 
    nodes = h.nodes
    nodemap = h.node_map

    nd_id = nodemap[handle]
    t0 = nodes[nd_id].time
    nodes[nd_id] = MutableBinaryHeapNode(t_new, handle, new_rate)
    if t_new < t0
        _heap_bubble_up!(nodes, nodemap, nd_id)
    else
        _heap_bubble_down!(nodes, nodemap, nd_id)
    end
end