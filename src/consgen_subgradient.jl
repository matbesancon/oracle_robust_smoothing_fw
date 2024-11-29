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


using Graphs
using LinearAlgebra
using JuMP
using HiGHS

function oracle_ST(G::Graphs.AbstractGraph, c::AbstractVector)

    @assert length(edges(G)) == length(c)
    weight_matrix = zeros(Float64, nv(G), nv(G))
    for (idx, edge) in enumerate(edges(G))
        (src, dst) = Tuple(edge)
        # ensure matrix is symmetric
        weight_matrix[src, dst] = weight_matrix[dst, src] = c[idx] 
    end
    tree_edges = Graphs.kruskal_mst(G, weight_matrix)
    tree_vector = BitVector(e ∈ tree_edges for e in edges(G))
    return tree_vector
end


function runCG_budgeted(lmo, c0, d, gamma; epsilon=0.0001, max_iterations=1000, optimizer=HiGHS.Optimizer, model = Model(optimizer), time_limit=3600)
    println("Start Constraint Generation")
    start_time = time()
    n = length(c0)
    empty!(model)
    @variable(model, 0 <= delta[1:n] <= 1)
    @variable(model, z >=0)
    @objective(model, Max, z)
    @constraint(model,budget, sum(delta[i] for i in 1:n) <= gamma)
    x_new = FrankWolfe.compute_extreme_point(lmo, c0)

    list_solutions = []
    list_objectives = Float64[]
    list_times = []

    set_silent(model)

    optimal = false
    iteration = 0
    while !optimal && iteration < max_iterations
        iteration += 1
        rhs_val = dot(c0, x_new)
        @constraint(model, sum(d[i] * x_new[i] * delta[i] for i in 1:n) - z >= -rhs_val)
        optimize!(model)

        if termination_status(model) == OPTIMAL
            c_new = c0 + value.(delta) .* d
            push!(list_solutions, copy(x_new))
            push!(list_objectives, value(z))
            time_iter = time()-start_time
            push!(list_times, time_iter)
            if time_iter >= time_limit
                @warn("Time limit")
            end
            x_new = FrankWolfe.compute_extreme_point(lmo, c_new)
            opt_oracle = dot(x_new, c_new)

            if mod(iteration,100)==0
                println("CG: Iteration " * string(iteration) * " Objective: " * string(value(z)))
            end

            if opt_oracle >= value(z)-epsilon
                optimal = true
            end
        else
            @warn("Not optimal $(termination_status(model))")
        end
    end

    opt_sol, opt_val = simplicial_convexhull_oracle_budgeted(list_solutions, c0, d, gamma; optimizer)

    return opt_val, list_solutions, iteration, opt_sol, list_objectives, list_times
end

function runCG_simplex(lmo, list_c, epsilon; max_iterations=1000, optimizer=HiGHS.Optimizer)
    n = length(list_c[1])
    m = length(list_c)
    model = Model(optimizer)
    @variable(model, 0 <= lambda[1:m] <= 1)
    @variable(model, 0 <= c[1:n])
    @variable(model, z >= 0)
    @objective(model, Max, z)
    @constraint(model, sum(lambda[i] for i in 1:m) == 1)

    x_new = FrankWolfe.compute_extreme_point(lmo, list_c[1])

    list_solutions = [x_new]

    set_silent(model)

    optimal = false
    iteration = 0
    while !optimal && iteration < max_iterations
        iteration += 1

        @constraint(model, sum(c[i]*x_new[i] for i in 1:n) - z >= 0)
        @constraint(model, sum(lambda[i] * collect(list_c[i]) for i in 1:m) .== c)
        
        optimize!(model)

        if termination_status(model) == OPTIMAL
            c_new = value.(c)
            x_new = FrankWolfe.compute_extreme_point(lmo, c_new)
            opt_oracle = dot(x_new, c_new)
            #println(value(z))

            push!(list_solutions, copy(x_new))

            if opt_oracle >= value(z) - epsilon
                optimal = true
            end
        else
            @warn("Not optimal")
        end
    end

    opt_sol, opt_val = simplicial_convexhull_oracle_simplex(list_solutions,list_c; optimizer)

    return opt_val, list_solutions, iteration, opt_sol
end


function projectOn_U(cref, d, gamma, c_out, optimizer=HiGHS.Optimizer)
    n = length(cref)

    model = Model(optimizer)
    @variable(model, 0<=delta[1:n]<=1)
    @variable(model, 0<=c[1:n])
    @objective(model, Min, sum((c[i]-c_out[i])^2 for i in 1:n))

    @constraint(model, sum(delta[i] for i in 1:n) <= gamma)
    @constraint(
        model,
        cons_budget[i=1:n],
        c[i] - delta[i] * d[i] == cref[i],
    )
    set_silent(model)
    optimize!(model)

    c_proj = cref + value.(delta) .* d
    return c_proj
end

function projectOn_simplex(list_c,c_out, optimizer=HiGHS.Optimizer)
    m = length(list_c)
    n = length(c_out)

    model = Model(optimizer)
    @variable(model, 0<=lambda[1:m]<=1)
    @variable(model, 0<=c[1:n])
    @objective(model, Min, sum((c[i]-c_out[i])^2 for i in 1:n))

    @constraint(model, sum(lambda[i] for i in 1:m) == 1)
    for i in 1:n
        @constraint(model,c[i] + sum((-list_c[j][i])*lambda[j] for j in 1:m)==0)
    end

    set_silent(model)

    optimize!(model)

    c_proj = zeros(n)
    for j in 1:m
        c_proj = c_proj + value(lambda[j]) .*list_c[j]
    end

    return c_proj
end

function simplicial_convexhull_oracle_budgeted(vertices, c0, d, gamma; optimizer=HiGHS.Optimizer)
    m = Model(optimizer)
    set_silent(m)
    n = length(vertices[1])
    nv = length(vertices)
    @variable(m, x[1:n])
    @variable(m, a>=0)
    @variable(m, b[1:n]>=0)
    @variable(m, α[1:nv] >= 0)
    @constraint(m, sum(α) == 1)
    @constraint(m, sum(α[i] * vertices[i] for i in 1:nv) .== x)

    @constraint(m, [i=1:n], a + b[i] - d[i]*x[i]>= 0)

    @objective(m, Min, gamma * a + sum(b[i] for i in 1:n) + sum(c0[i]*x[i] for i in 1:n))
    optimize!(m)

    obj = objective_value(m)
    @assert termination_status(m) == MOI.OPTIMAL

    devs = value.(x) .* d
    indices = sortperm(-devs)
    c_ret = copy(c0)
    for i in 1:gamma
        c_ret[indices[i]] = c_ret[indices[i]] + d[indices[i]]
    end

    x_ret = value.(x)

    return x_ret, c_ret, obj, JuMP.value.(α)
end

function simplicial_convexhull_oracle_simplex(vertices, c_vectors; optimizer=HiGHS.Optimizer)
    m = Model(optimizer)
    set_silent(m)
    n = length(vertices[1])
    nv = length(vertices)
    nc = length(c_vectors)
    @variable(m, x[1:n])
    @variable(m, z)
    @variable(m, α[1:nv] >= 0)
    @constraint(m, sum(α) == 1)
    @constraint(m, sum(α[i] * collect(vertices[i]) for i in 1:nv) .== x)
    @constraint(
        m,
        underestimation[j=1:nc],
        z ≥ dot(c_vectors[j], x),
    )
    @objective(m, Min, z)
    optimize!(m)
    @assert termination_status(m) == MOI.OPTIMAL

    return JuMP.value.(x), objective_value(m)
end

function isInU(c0, d, gamma,c_out)
    budget = sum( (c_out[i] - c0[i]) / d[i] for i in eachindex(c0))
    inU = budget <= gamma + 0.00001
    return inU
end

function run_subgradient_budgeted(lmo, cref, d, gamma; epsilon=0.0001, max_iterations=1000, optimizer=HiGHS.Optimizer)
    println("Start Subgradient Ascent")
    start_time = time()
    c_proj = copy(cref)
    n = length(cref)
    objective_value = 0

    diamU = norm(d-cref)
    max_gradnorm = sqrt(n)
    objective_values = Float64[]
    iter = 0
    list_solutions = []
    list_times = []
    last_value = -10000000.0
    countSameObjective = 0
    while iter < max_iterations && countSameObjective < 50
        iter += 1
        x_new = FrankWolfe.compute_extreme_point(lmo, c_proj)
        if x_new ∉ list_solutions
            push!(list_solutions, x_new)
        end

        objective_value = dot(c_proj, x_new)
        push!(objective_values, objective_value)
        push!(list_times,time()-start_time)
        if mod(iter,100)==0
            println("Gradient Ascent: Iteration " * string(iter) * " Objective: " * string(objective_value))
        end

        if abs(objective_value-last_value)>epsilon
            countSameObjective = 0
        else
            countSameObjective = countSameObjective + 1
        end
        last_value = objective_value

        #step_length = 1/(1 + iteration)                   #1/1+t step-length
        #step_length = 0.01                                    #constant step-length
        step_length = diamU / (max_gradnorm * sqrt(iter))  #step-length from paper

        # collect necessary because of type conversion behavior. See issue on SparseArrays
        c_new = c_proj + step_length * collect(x_new)
        
        # LP version, used for debugging only
        # c_proj = projectOn_U(cref, d, gamma, c_new, optimizer)
        Δ = solve_quadratic_knapsack(
            d * 0, # lower bounds
            d, # upper bound
            gamma, # rhs
            inv.(d), # coefficient knapsack inequality
            ones(size(d)), # quadratic coefficients
            c_new - cref, # quadratic reference point
        )
        c_proj = cref + Δ
    end
    x_opt, opt_val_BT = simplicial_convexhull_oracle_budgeted(list_solutions, cref, d, gamma; optimizer)

    return opt_val_BT, objective_values, list_solutions, x_opt, list_times
end
