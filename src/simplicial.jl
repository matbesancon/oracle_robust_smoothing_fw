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

using JuMP

# function simplicial_convexhull_oracle(vertices, c_vectors; optimizer=HiGHS.Optimizer)
#     m = Model(optimizer)
#     set_silent(m)
#     n = length(vertices[1])
#     nv = length(vertices)
#     nc = length(c_vectors)
#     @variable(m, x[1:n])
#     @variable(m, z)
#     @variable(m, α[1:nv] >= 0)
#     @constraint(m, sum(α) == 1)
#     @constraint(m, sum(α[i] * collect(vertices[i]) for i in 1:nv) .== x)
#     @constraint(
#         m,
#         underestimation[j=1:nc],
#         z ≥ dot(c_vectors[j], x),
#     )
#     @objective(m, Min, z)
#     optimize!(m)
#     @assert termination_status(m) == MOI.OPTIMAL
#     λ = JuMP.dual.(underestimation)
#     subgradient = sum(λ[j] * c_vectors[j] for j in 1:nc)
#     return JuMP.value.(α), JuMP.value.(x), subgradient
# end

function simplicial_robust_oracle(lmo, x0, c0, d, gamma; epsilon=0.0001, max_iteration=100, optimizer=HiGHS.Optimizer, true_function=(x -> 1.0))
    x_t = copy(x0)
    vertices = [x_t]
    weights = [1.0]
    c_t = copy(c0)
    success = false
    t = 0
    vals = Float64[]
    println("Start Simplicial Algorithm")
    while t < max_iteration
        t += 1
        x_t, c_t, val_t = simplicial_convexhull_oracle_budgeted(vertices, c0, d, gamma; optimizer=HiGHS.Optimizer)
        v_t = FrankWolfe.compute_extreme_point(lmo, c_t)
        if dot(c_t, v_t) ≥ dot(c_t, x_t)-epsilon
            success = true
            break
        end
        push!(vertices, v_t)
        push!(vals, val_t)
    end
    if !success
        @warn "did not terminate"
    end
    return vertices, vals
end

# vertices for the budgeted uncertainty set [0,1] ∩ {∑ c == Γ}
function vertices_budgeted_uncertainty(Gamma, n)
    @assert isinteger(Gamma)
    @assert Gamma ≤ n
    map(Combinatorics.combinations(1:n, Int(Gamma))) do idxs
        v = falses(n)
        for idx in idxs
            v[idx] = 1
        end
        v
    end
end
