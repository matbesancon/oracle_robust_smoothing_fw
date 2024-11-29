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

"""
    solve_quadratic_knapsack(lower_bounds, upper_bounds, b, a, d, q)

Solve the quadratic continuous knapsack problem of the form:
```
min_x ∑_i d_i / 2 * (x_i - q_i)^2
∑ a_i x_i ≤ b
l ≤ x ≤ u
```

If `equality`, then the knapsack constraint becomes `∑ a_i x_i = b`.

Specialized version of the 2-pegging breakpoint algorithm from
> M. Patriksson, C. Strömberg,
Algorithms for the continuous nonlinear resource allocation problem New implementations and numerical studies,
European Journal of Operational Research (2015), http://dx.doi.org/10.1016/j.ejor.2015.01.02
"""
function solve_quadratic_knapsack(lower_bounds, upper_bounds, b, a, d, q; equality=false, tol=10eps())
    # start by projecting q onto the bounds and verifying the knapsack constraint
    # Step 0
    n = length(lower_bounds)
    xsol = clamp.(q, lower_bounds, upper_bounds)
    base_activity = dot(xsol, a)
    if base_activity ≈ b || !equality && base_activity <= b + tol
        return xsol
    end
    mu_l = - d .* (lower_bounds .- q) ./ a
    mu_u = - d .* (upper_bounds .- q) ./ a
    mu_k = vcat(-Inf, collect(mu_l), collect(mu_u), Inf)
    N = collect(1:n)
    b_k = b
    fixed_to_lower = BitSet()
    fixed_to_upper = BitSet()
    while true
        # step 1
        if isempty(mu_k)
            return compute_iterate_relaxed_after_fixings!(xsol, a, q, d, b_k, lower_bounds, upper_bounds, fixed_to_lower, fixed_to_upper)
        end
        mu_med = median(mu_k)
        # step 2
        compute_active_sets_from_multiplier!(xsol, mu_med, lower_bounds, upper_bounds, d, q, a)
        beta_l = beta_u = 0.0
        for j in N
            if mu_l[j] ≤ mu_med
                beta_l += lower_bounds[j] * a[j]
            end
            if mu_u[j] ≥ mu_med
                beta_u += upper_bounds[j] * a[j]
            end
        end
        δ = beta_l + beta_u
        for j in N
            if mu_u[j] < mu_med < mu_l[j]
                δ += a[j] * xsol[j]
            end
        end
        index_to_delete = BitSet()
        if δ > b_k + tol
            # Step 3.1
            for j in N
                if mu_med ≥ mu_l[j]
                    push!(index_to_delete, j)
                    push!(fixed_to_lower, j)
                end
            end
            setdiff!(N, index_to_delete)
            filter!(>(mu_med), mu_k)
            b_k -= beta_l
        elseif δ < b_k - tol
            # Step 3.2
            index_to_delete = BitSet()
            for j in N
                if mu_med ≤ mu_u[j]
                    push!(index_to_delete, j)
                    push!(fixed_to_upper, j)
                end
            end
            setdiff!(N, index_to_delete)
            filter!(<(mu_med), mu_k)
            b_k -= beta_u
        else # δ = b_k ± tol 
            return xsol
        end
    end
end

function compute_active_sets_from_multiplier!(x, mu, lower_bounds, upper_bounds, d, q, a)
    for j in eachindex(x)
        if mu ≥ - d[j] * (lower_bounds[j] - q[j]) / a[j]
            x[j] = lower_bounds[j]
        elseif mu ≤ - d[j] * (upper_bounds[j] - q[j]) / a[j]
            x[j] = upper_bounds[j]
        else
            x[j] = -mu * a[j] / d[j] + q[j]
        end
    end
    return x
end

"""
Solve the relaxed model ignoring the bound constraints on all unfixed variables.
One can obtain a closed-form solution for μ from `∇_x L(x,μ) = 0`, resulting in an expression for each `x_j`,
which can then be substituted back into `∑_j a_j x_j = b_k`.
"""
function compute_iterate_relaxed_after_fixings!(x, a, q, d, b_k, lower_bounds, upper_bounds, fixed_to_lower, fixed_to_upper)
    # free_variables = [n] \ fixed_variables
    free_variables = collect(eachindex(x))
    setdiff!(free_variables, fixed_to_lower, fixed_to_upper)
    if !isempty(free_variables)
        μ = (b_k - sum(a[j] * q[j] for j in free_variables)) / sum(a[j]^2 / d[j] for j in free_variables)
    end
    for j in eachindex(x)
        if j in fixed_to_lower
            x[j] = lower_bounds[j]
        elseif j in fixed_to_upper
            x[j] = upper_bounds[j]
        else
            x[j] = (μ * a[j] + d[j] * q[j]) / d[j]
        end
    end
    return x
end
