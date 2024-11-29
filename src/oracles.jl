# Copyright 2024 Mathieu Besançon
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Spanning tree on undirected graph
struct SpanningTreeOracle{G<:Graphs.AbstractGraph,M<:AbstractMatrix} <: FrankWolfe.LinearMinimizationOracle
    g::G
    weight_matrix::M
end

SpanningTreeOracle(g) = SpanningTreeOracle(g, similar(weights(g), Float64))

function FrankWolfe.compute_extreme_point(lmo::SpanningTreeOracle, direction::AbstractVector; kwargs...)
    # direction represents edges
    @assert length(edges(lmo.g)) == length(direction)
    for (idx, edge) in enumerate(edges(lmo.g))
        (src, dst) = Tuple(edge)
        # ensure matrix is symmetric
        lmo.weight_matrix[src, dst] = lmo.weight_matrix[dst, src] = direction[idx] 
    end
    tree_edges = Graphs.kruskal_mst(lmo.g, lmo.weight_matrix)
    tree_vector = BitVector(e in tree_edges for e in edges(lmo.g))
    return tree_vector
end

struct TSPOracle{G<:Graphs.AbstractGraph,M<:AbstractMatrix} <: FrankWolfe.LinearMinimizationOracle
    g::G
    weight_matrix::M
end

TSPOracle(g) = TSPOracle(g, similar(weights(g), Float64))

function subtour(edge_matrix::Matrix, n)
    return subtour(selected_edges(edge_matrix, n), n)
end

function subtour(edges::Vector{Tuple{Int,Int}}, n)
    shortest_subtour, unvisited = collect(1:n), Set(collect(1:n))
    while !isempty(unvisited)
        this_cycle, neighbors = Int[], unvisited
        while !isempty(neighbors)
            current = pop!(neighbors)
            push!(this_cycle, current)
            if length(this_cycle) > 1
                pop!(unvisited, current)
            end
            neighbors =
                [j for (i, j) in edges if i == current && j in unvisited]
        end
        if length(this_cycle) < length(shortest_subtour)
            shortest_subtour = this_cycle
        end
    end
    return shortest_subtour
end

function FrankWolfe.compute_extreme_point(lmo::TSPOracle, direction::AbstractVector; kwargs...)
    # direction represents edges
    @assert length(edges(lmo.g)) == length(direction)
    for (idx, edge) in enumerate(edges(lmo.g))
        (src, dst) = Tuple(edge)
        # ensure matrix is symmetric
        lmo.weight_matrix[src, dst] = lmo.weight_matrix[dst, src] = direction[idx] 
    end

    for i in 1:nv(lmo.g)
        lmo.weight_matrix[i,i] = 1000000
    end

    d = lmo.weight_matrix
    n = nv(lmo.g)
    #println(d)
    lazy_model = Model(GLPK.Optimizer)
    #set_optimizer_attribute(lazy_model, "msg_lev", GLPK.GLP_MSG_ALL)
    @variable(lazy_model, x[1:n, 1:n], Bin, Symmetric)
    @objective(lazy_model, Min, dot(d, x) / 2)
    @constraint(lazy_model, [i in 1:n], sum(x[i, :]) == 2)
    @constraint(lazy_model, [i in 1:n], x[i, i] == 0)
    
    function subtour_elimination_callback(cb_data)
        status = callback_node_status(cb_data, lazy_model)
        if status != MOI.CALLBACK_NODE_STATUS_INTEGER
            return  # Only run at integer solutions
        end
        x_value = callback_value.(cb_data, lazy_model[:x])
        cycle = subtour(x_value, n)
        if !(1 < length(cycle) < n)
            return  # Only add a constraint if there is a cycle
        end
        #println("Found cycle of length $(length(cycle))")
        S = [(i, j) for (i, j) in Iterators.product(cycle, cycle) if i < j]
        con = @build_constraint(
            sum(lazy_model[:x][i, j] for (i, j) in S) <= length(cycle) - 1,
        )
        MOI.submit(lazy_model, MOI.LazyConstraint(cb_data), con)
        return
    end
    set_attribute(
        lazy_model,
        MOI.LazyConstraintCallback(),
        subtour_elimination_callback,
    )
    optimize!(lazy_model)
    if termination_status(lazy_model) == OPTIMAL
        objective_value(lazy_model)
    else
        @warn("Not optimal $(termination_status(lazy_model))")
    end
    @assert is_solved_and_feasible(lazy_model)

    x_ret = Float64[]
    for edge in edges(lmo.g)
        push!(x_ret,value(x[Tuple(edge)[1],Tuple(edge)[2]]))
    end

    return x_ret
end

function selected_edges(x::Matrix{Float64}, n)
    return Tuple{Int,Int}[(i, j) for i in 1:n, j in 1:n if x[i, j] > 0.5]
end

struct MinCostFlowOracle{G<:Graphs.AbstractGraph,M<:AbstractMatrix,VD<:AbstractVector{<:Real}} <: FrankWolfe.LinearMinimizationOracle
    g::G
    weight_matrix::M
    node_demand::VD
    silent::Bool
end

MinCostFlowOracle(g, node_demand::AbstractVector) = MinCostFlowOracle(g, similar(weights(g), Float64), node_demand, true)

function FrankWolfe.compute_extreme_point(lmo::MinCostFlowOracle, direction::AbstractVector; kwargs...)
    for (idx, edge) in enumerate(edges(lmo.g))
        (src, dst) = Tuple(edge)
        lmo.weight_matrix[src, dst] = direction[idx] 
    end
    optimizer = optimizer_with_attributes(
        HiGHS.Optimizer, MOI.Silent() => lmo.silent,
    )
    flow = GraphsFlows.mincost_flow(
        lmo.g, lmo.node_demand,
        Ones(nv(lmo.g), nv(lmo.g)),
        lmo.weight_matrix, optimizer,
    )
    v = SparseArrays.spzeros(size(direction))
    for (idx, edge) in enumerate(edges(lmo.g))
        (src, dst) = Tuple(edge)
        f = flow[src, dst]
        if ≉(f, 0, atol=1e-6)
            v[idx] = flow[src, dst]
        end
    end
    return v
end
