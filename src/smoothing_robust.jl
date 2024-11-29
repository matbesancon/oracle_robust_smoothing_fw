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

function evaluate_function(x, c, mu, c0)
    return dot(x, c) - mu/2 * sum((c[i] - c0[i])^2 for i in eachindex(c))
end

"""
Given a model encoding the adversarial problem constraints:
```
c ∈ U
```
sets the objective: MAX `⟨c,x⟩ - μ / 2 * ||c - c0||`,
optimizes and returns the optimal c and the optimal value f_μ(x)
"""
function solve_inner_problem_objective(mu, x, model::JuMP.AbstractModel, c; c0=zeros(size(c)...))
    f = evaluate_function(x, c, mu, c0)
    fref = objective_function(model)
    if f != fref
        # evaluating for a new objective
        MOI.set(model, MOI.ObjectiveSense(), MOI.FEASIBILITY_SENSE)
        @objective(model, Max, f)
        JuMP.optimize!(model)
        st = termination_status(model)
        if st != MOI.OPTIMAL
            println(x)
        end
        write_to_file(model, "error.lp")
        @assert st == MOI.OPTIMAL "Unexpected status $(st)"
    end
    cvalues = JuMP.value.(c)
    opt_val = JuMP.objective_value(model)
    return (cvalues, opt_val)
end

function solve_budgeted_inner_problem_jump(d, gamma, x, mu, c0, cref; optimizer=HiGHS.Optimizer)
    n = length(c0)
    model = Model(optimizer)
    @variable(model, 0<=delta[1:n]<=1)
    @variable(model, 0<=c[1:n])
    @objective(model, Max, dot(x,c) - mu/2 * sum((c[i]-c0[i])^2 for i in 1:n))
    @constraint(model, sum(delta[i] for i in 1:n) <= gamma)
    @constraint(
        model,
        cons_budget[i=1:n],
        c[i] - delta[i] * d[i] == cref[i],
    )
    set_silent(model)
    optimize!(model)
    @assert termination_status(model) == MOI.OPTIMAL
    c_proj = cref + value.(delta) .* d
    return c_proj
end

# c = cref + ∑ δ_i d_i
# ∑ δ_i ≤ Γ
# δ_i ∈ [0,1]
# change of variable
# Δ_i = δ_i d_i
# c = cref + ∑ Δ_i
# ∑ Δ_i / d_i ≤ Γ
# Δ_i ∈ [0, d]
function solve_budgeted_inner_problem(d, Γ, x, mu, c0, cref; storage=similar(cref))
    n = length(d)
    @assert all(>=(0), d)
    copyto!(storage, cref)
    if mu == 0
        sorted_idx = sortperm(x .* d, rev=true)
        for idx in 1:floor(Int, Γ)
            storage[sorted_idx[idx]] += d[sorted_idx[idx]]
        end
        return storage
    end
    lower_bounds = zeros(n)
    quad_coeff = mu * ones(n)
    point_to_project = c0 .+ x ./ mu - cref
    Δ = solve_quadratic_knapsack(lower_bounds, d, Γ, inv.(d), quad_coeff, point_to_project)
    @. storage += Δ
    return storage
end

"""
Computes an epsilon-approximate solution to the robust problem:
```
min_{x ∈ X} max_{c ∈ U} ⟨x, c⟩
```
"""
function solve_robust_problem(
        cref, c0, d, Γ, lmo;
        epsilon=1e-4, max_lmo_calls=2500, max_iteration=4 * max_lmo_calls,
        standard_fw = false, automatic_tolerance=false, D=sqrt(length(cref)), smoothing_parameter=nothing,
        update_bound_convhull=false,
        best_val_vector=nothing,
        best_solution=nothing,
        kwargs...
    )
    # max distance from a point to c0 in the set
    maxed_negative_indices = [i for i in eachindex(cref) if c0[i] - cref[i] > d[i] + cref[i] - c0[i]]
    maxed_positive_indices = Int[]
    left_indices = setdiff(eachindex(cref), maxed_negative_indices)
    @info "Γ=$Γ"
    for _ in 1:Γ
        setdiff!(left_indices, maxed_positive_indices)
        if isempty(left_indices)
            break
        end
        # max index not taken yet
        imax = argmax(i -> d[i] + cref[i] - c0[i], left_indices)
        @assert imax !== nothing
        @assert d[imax] + cref[imax] - c0[imax] > 0
        push!(maxed_positive_indices, imax)
    end
    @assert !isempty(maxed_positive_indices)
    max_uncertain_distance_squared = sum((c0[i] - cref[i])^2 for i in maxed_negative_indices; init=0.0) + sum((d[i] + cref[i] - c0[i])^2 for i in maxed_positive_indices)
    
    # fixed mu for this method
    μ = epsilon / max_uncertain_distance_squared
    @info "Smoothing parameter: $μ"
    @info "Should run $(ceil(4 * D^2 * max_uncertain_distance_squared / epsilon^2)) iterations for the provided target accuracy"
    if automatic_tolerance
        epsilon = max(
            epsilon,
            2 * D * sqrt(max_uncertain_distance_squared / max_iteration),
        )
        μ = epsilon / max_uncertain_distance_squared
        @info "Setting epsilon to value $(epsilon) corresponding to maxiter, μ=$μ"
    end
    if smoothing_parameter !== nothing
        μ = smoothing_parameter
    end
    function f(x)
        c = solve_budgeted_inner_problem(d, Γ, x, μ, c0, cref)
        opt_val = evaluate_function(x, c, μ, c0)
        return opt_val
    end
    function grad!(storage, x)
        c = solve_budgeted_inner_problem(d, Γ, x, μ, c0, cref)
        storage .= c
        return nothing 
    end
    x0 = FrankWolfe.compute_extreme_point(lmo, c0)

    function worst_case_value(x)
        c = solve_budgeted_inner_problem(d, Γ, x, 0, c0, cref)
        dot(c, x)
    end

    # LMO counting the number of calls
    tracking_lmo = FrankWolfe.TrackingLMO(lmo)
    trajectory_vector = []
    callback = make_callback_maximum_oracle_calls(tracking_lmo, max_lmo_calls, trajectory_vector, worst_case_value, max_uncertain_distance_squared, D, μ)
    if update_bound_convhull
        callback = make_callback_update_convexhull(worst_case_value, lmo, c0, d, Γ, epsilon, best_val_vector, best_solution, callback=callback)
    end
    # if we automatically set the epsilon value, still force to run enough iterations
    fw_epislon = automatic_tolerance ? 1e-7 : epsilon
    if standard_fw
        x, v, primal_val, dual_gap, trajectory = FrankWolfe.frank_wolfe(f, grad!, tracking_lmo, x0; epsilon=fw_epislon, verbose=true, callback=callback, max_iteration=max_iteration, gradient=1.0 * x0, kwargs...)
    else
        x, v, primal_val, dual_gap, trajectory, _ = FrankWolfe.blended_pairwise_conditional_gradient(f, grad!, tracking_lmo, x0; epsilon=fw_epislon, verbose=true, callback=callback, max_iteration=max_iteration, kwargs...)
    end
    # recompute c one last time
    cfinal = similar(c0)
    grad!(cfinal, x)
    return x, cfinal, v, primal_val, dual_gap, trajectory_vector
end

function build_adaptive_function_gradient(d, cref, Γ, L_g, D, start_multiplication; verbose=true)
    iter_counter = Ref(0)
    μ0 = start_multiplication * D / L_g
    c0 = 0 * cref
    verbose && @info "μ0: $μ0"
    function f(x)
        μ = μ0 / sqrt(iter_counter[] + 1)
        c = solve_budgeted_inner_problem(d, Γ, x, μ, c0, cref)
        return evaluate_function(x, c, μ, c0)
    end
    # only increment counter on gradient call to avoid reducing the smoothing parameter during line search
    function grad!(storage, x)
        μ = μ0 / sqrt(iter_counter[] + 1)
        c = solve_budgeted_inner_problem(d, Γ, x, μ, c0, cref)
        iter_counter[] += 1
        storage .= c
    end
    return f, grad!
end

function solve_robust_problem_adaptive(cref,c0, d, Γ, lmo;
        max_lmo_calls=2500, max_iteration=4 * max_lmo_calls, D = sqrt(length(cref)), line_search=FrankWolfe.Agnostic(), standard_fw=false, start_multiplication=1,
        update_bound_convhull=false,
        best_val_vector=nothing,
        best_solution=nothing,
        epsilon=1e-4,
        kwargs...
    )
    n = length(cref)

    # max distance from a point to c0 in the set
    maxed_negative_indices = [i for i in eachindex(cref) if c0[i] - cref[i] > d[i] + cref[i] - c0[i]]
    maxed_positive_indices = Int[]
    left_indices = setdiff(eachindex(cref), maxed_negative_indices)
    @info "Γ=$Γ"
    for _ in 1:Γ
        setdiff!(left_indices, maxed_positive_indices)
        if isempty(left_indices)
            break
        end
        # max index not taken yet
        imax = argmax(i -> d[i] + cref[i] - c0[i], left_indices)
        @assert imax !== nothing
        @assert d[imax] + cref[imax] - c0[imax] > 0
        push!(maxed_positive_indices, imax)
    end
    @assert !isempty(maxed_positive_indices)
    max_uncertain_distance_squared = sum((c0[i] - cref[i])^2 for i in maxed_negative_indices; init=0.0) + sum((d[i] + cref[i] - c0[i])^2 for i in maxed_positive_indices)

    # compute the max norm ||c||
    @assert all(>(0), d)
    # negative indices not taken, since the max norm is already reached.
    maxed_indices = BitSet(i for i in 1:n if cref[i] < 0)
    for _ in 1:Γ
        left_indices = setdiff(1:n, maxed_indices)
        if isempty(left_indices)
            break
        end
        # max index not taken yet
        imax = argmax(i -> d[i], left_indices)
        @assert imax !== nothing
        push!(maxed_indices, imax)
    end
    cmaxed = 1.0 * cref
    for idx in maxed_indices
        if cmaxed[idx] >= 0
            cmaxed[idx] += d[idx]
        end
    end
    L_g = norm(cmaxed)
    f, grad! = build_adaptive_function_gradient(d, cref, Γ, L_g, D, start_multiplication)
    x0 = FrankWolfe.compute_extreme_point(lmo, cref)

    function worst_case_value(x)
        c = solve_budgeted_inner_problem(d, Γ, x, 0, cref, cref)
        dot(c, x)
    end

    # LMO counting the number of calls
    tracking_lmo = FrankWolfe.TrackingLMO(lmo)
    trajectory_vector = []
    callback = make_callback_maximum_oracle_calls(tracking_lmo, max_lmo_calls, trajectory_vector, worst_case_value, max_uncertain_distance_squared, D, 1)
    if update_bound_convhull
        callback = make_callback_update_convexhull(worst_case_value, lmo, c0, d, Γ, epsilon, best_val_vector, best_solution, callback=callback)
    end

    if standard_fw
        x, v, primal_val, dual_gap, _ = FrankWolfe.frank_wolfe(f, grad!, tracking_lmo, x0, line_search=line_search, verbose=true, max_iteration=max_iteration, gradient=1.0*x0, callback=callback, epsilon=epsilon)
    else
        x, v, primal_val, dual_gap, _, _ = FrankWolfe.blended_pairwise_conditional_gradient(f, grad!, tracking_lmo, x0, line_search=line_search, verbose=true, max_iteration=max_iteration, callback=callback, epsilon=epsilon)
    end
    # recompute c one last time
    cfinal = similar(cref)
    grad!(cfinal, x)
    return x, cfinal, v, primal_val, dual_gap, trajectory_vector
end

# callback to interrupt FW with a budget in number of oracle calls
function make_callback_maximum_oracle_calls(lmo::FrankWolfe.TrackingLMO, maxnumber_oracles::Int, trajectory_vector, compute_worst_objective::Function, max_uncertain_distance_squared, diameter, mu)
    # error = A / (t+3) + B
    A = 2 * diameter^2 / mu
    B = mu * max_uncertain_distance_squared / 2
    function callback_maximum_oracle_calls(state, args...)
        val = compute_worst_objective(state.x)
        theoretical_bound = A / (state.t + 3) + B
        push!(trajectory_vector, (state.t, val, lmo.counter, state.time, state.primal, state.dual, theoretical_bound))
        return lmo.counter < maxnumber_oracles
    end
end

function make_callback_update_convexhull(f, lmo, c0, d, gamma, epsilon, best_vals::Vector, best_solution; callback=nothing, start_iteration=1, every_k_iteration=10)
    vertex_list = []
    best_candidate = Ref(copy(d))
    f_best = Ref(Inf)
    n_iter = Ref(0)
    function callback_update_convhull(state, args...)
        n_iter[] += 1
        if callback(state, args...) === false
            @info "terminating from inner callback"
            return false
        end
        # vertex already seen, nothing new
        if any(v -> v ≈ state.v, vertex_list)
            f_fw = f(state.x)
            if f_best[] > f_fw
                best_candidate[] .= state.x
                best_solution .= state.x
            end
            f_best[] = f(best_candidate[])
            push!(best_vals, f_best[])
            if n_iter[] < 2
                @info "best from FW"
            end
            return true
        end
        push!(vertex_list, state.v)
        # skip computation before first iteration
        if state.t <= start_iteration
            f_fw = f(state.x)
            if f_best[] > f_fw
                best_candidate[] .= state.x
                best_solution .= state.x
            end
            f_best[] = f(best_candidate[])
            push!(best_vals, f_best[])
            if n_iter[] < 2
                @info "best from FW"
            end
            return true
        end
        x_opt, c_t, val_t, α = simplicial_convexhull_oracle_budgeted(vertex_list, c0, d, gamma)
        # if count(>=(1e-3), α) >= length(d) + 1
        #     @info "Optimum in interior of full-dimensional active set, terminating, $(minimum(α))"
        #     return false
        # end
        v_sub = FrankWolfe.compute_extreme_point(lmo, c_t)
        if dot(c_t, v_sub) >= dot(c_t, x_opt) - epsilon
            @info "Terminating with gap: $(dot(c_t, x_opt) - dot(c_t, v_sub))"
            # TODO correct?
            return false
        end
        f_fw = f(state.x)
        f_conv = f(x_opt)
        if f_best[] > f_fw >= f_conv
            best_candidate[] .= x_opt
            best_solution .= x_opt
            if n_iter[] < 2
                @info "best from convhull"
            end
        elseif f_best[] > f_conv >= f_fw
            best_candidate[] .= state.x
            best_solution .= state.x
            if n_iter[] < 2
                @info "best from FW"
            end
        end
        f_best[] = f(best_candidate[])
        push!(best_vals, f_best[])
        return true
    end
end
