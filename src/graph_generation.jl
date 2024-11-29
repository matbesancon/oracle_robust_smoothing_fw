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

using LinearAlgebra
using Random

using Graphs
using SimpleWeightedGraphs

function build_graph_shortest_path(n_points; threshold_keep=0.3, rng=Random.seed!(42))
    points = map(1:n_points) do _
        10 * rand(rng, 2)
    end
    edge_distances = []
    for i in 2:n_points
        for j in 1:(i-1)
            push!(
                edge_distances,
                (i, j, norm(points[i] - points[j])),
            )
        end
    end
    bottom_left_corner = findmin(p -> norm(p, Inf), points)[2]
    top_right_corner = findmax(p -> norm(p, Inf), points)[2]

    # sort edges from shortest to longest
    sort!(edge_distances, by=t->t[3])
    edge_distances_filtered = edge_distances[1:round(Int, length(edge_distances) * threshold_keep)]
    g = SimpleWeightedGraph(n_points)
    for (i, j, dist) in edge_distances_filtered
        Graphs.add_edge!(g, i, j, dist)
    end
    for idx in (round(Int, length(edge_distances) * threshold_keep)):length(edge_distances)
        if Graphs.is_connected(g)
            break
        end
        (i, j, dist) = edge_distances[idx]
        Graphs.add_edge!(g, i, j, dist)
    end
    @assert Graphs.is_connected(g)
    g_dir = SimpleWeightedDiGraph(g)
    return g_dir, bottom_left_corner, top_right_corner
end

function generate_shortest_path_problem(n_points; threshold_keep=0.3, rng=Random.seed!(42), deviation=0.5)
    g, bottom_left_corner, top_right_corner = build_graph_shortest_path(n_points, threshold_keep=threshold_keep, rng=rng)
    c0 = [SimpleWeightedGraphs.weight(e) for e in edges(g)]
    d = deviation * c0
    node_demand = spzeros(nv(g))
    node_demand[bottom_left_corner] = -1
    node_demand[top_right_corner] = 1
    lmo = MinCostFlowOracle(SimpleDiGraph(g), collect(weights(g)), node_demand, true)
    return (lmo, c0, d)
end
