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
import FrankWolfeRobustOpt as FWR
using LinearAlgebra
using JuMP
using FrankWolfe
using Random
using Plots
using Printf
using JSON
using HiGHS

const results = []
const eps_tol = 1e-3
const smoothing_parameter = 0.05
const iterations = 2500

# spanning tree instances
for extreme_scenarios in (true,false)
    for n in (10,)
        g = Graphs.complete_graph(n)
        lmo = FWR.SpanningTreeOracle(g)
        for seed in 1:3
            Random.seed!(seed)
            
            #uniform parameters
            #c0 = rand(ne(g))*100
            #normally distributed parameters
            c0 = randn(ne(g))
            c0 = c0 .+ (-minimum(c0)+1)
            dim = length(c0)
            if extreme_scenarios
                d = 2*c0 + rand(dim).*(10*c0.-2*c0)
            else
                d = 0.3*c0 + rand(dim).*(1*c0.-0.3*c0)
            end
            
            for Γ in (max(1,floor(Int,0.1*n)), max(1,floor(Int,0.2*n)), max(1,floor(Int,0.3*n)))
                function worst_case_value(x)
                    c = FWR.solve_budgeted_inner_problem(d, Γ, x, 0, c0, c0)
                    dot(c, x)
                end
                res_consgen = FWR.runCG_budgeted(lmo, c0, d, Γ; epsilon = eps_tol * n, max_iterations=iterations)
                iterates_consgen = res_consgen[2]
                obj_consgen = res_consgen[5]
                times_consgen = res_consgen[6]
                #val_consgen = worst_case_value.(iterates_consgen)

                # res_subgradient = FWR.run_subgradient_budgeted(lmo, c0, d, Γ; epsilon = eps_tol * n, max_iterations=iterations)
                # iterates_subgradient = res_subgradient[3]
                # obj_subgradient = res_subgradient[2]
                # times_subgradient = res_subgradient[5]
                
                res_static = FWR.solve_robust_problem(c0, c0, d, Γ, lmo, max_lmo_calls = iterations, max_iteration=iterations, automatic_tolerance=false, line_search=FrankWolfe.Agnostic(), lazy=true, smoothing_parameter=smoothing_parameter)
                trajectory_static = res_static[end]
                val_trajectory_static = getindex.(trajectory_static, 2)
                times_trajectory_static = getindex.(trajectory_static, 4)
                n_lmos_static = getindex.(trajectory_static, 3)
                res_adaptive = FWR.solve_robust_problem_adaptive(c0,c0, d, Γ, lmo, max_lmo_calls = iterations, line_search=FrankWolfe.Agnostic(), lazy=true)
                trajectory_adaptive = res_adaptive[end]
                val_trajectory_adaptive = getindex.(trajectory_adaptive, 2)
                times_trajectory_adaptive = getindex.(trajectory_adaptive, 4)
                n_lmos_adaptive = getindex.(trajectory_adaptive, 3)

                best_val_vector = Float64[]
                best_solution = 0 * d
                res_static_convhull_bound = FWR.solve_robust_problem(c0, c0, d, Γ, lmo, max_iteration=iterations, max_lmo_calls = iterations, automatic_tolerance=false, line_search=FrankWolfe.Agnostic(), lazy=true, smoothing_parameter=smoothing_parameter, update_bound_convhull=true, best_val_vector=best_val_vector, best_solution=best_solution)
                # res_static_convhull_iterate = FWR.solve_robust_problem(c0, c0, d, Γ, lmo, max_lmo_calls = iterations, automatic_tolerance=false, line_search=FrankWolfe.Agnostic(), lazy=true, smoothing_parameter=smoothing_parameter, update_bound_convhull=true, update_iterate_convhull=true)

                trajectory_bound = res_static_convhull_bound[end]
                val_trajectory_bound = getindex.(trajectory_bound, 2)
                times_trajectory_bound = getindex.(trajectory_bound, 4)
                n_lmos_bound = getindex.(trajectory_bound, 3)

                best_val_vector_adaptive = Float64[]
                best_solution_adaptive = 0 * d
                res_adaptive_convhull_bound = FWR.solve_robust_problem(c0, c0, d, Γ, lmo, max_iteration=iterations, max_lmo_calls = iterations, automatic_tolerance=false, line_search=FrankWolfe.Agnostic(), lazy=true, update_bound_convhull=true, best_val_vector=best_val_vector_adaptive, best_solution=best_solution_adaptive)
                # res_static_convhull_iterate = FWR.solve_robust_problem(c0, c0, d, Γ, lmo, max_lmo_calls = iterations, automatic_tolerance=false, line_search=FrankWolfe.Agnostic(), lazy=true, smoothing_parameter=smoothing_parameter, update_bound_convhull=true, update_iterate_convhull=true)

                trajectory_bound_adaptive = res_adaptive_convhull_bound[end]
                val_trajectory_bound_adaptive = getindex.(trajectory_bound_adaptive, 2)
                times_trajectory_bound_adaptive = getindex.(trajectory_bound_adaptive, 4)
                n_lmos_bound_adaptive = getindex.(trajectory_bound_adaptive, 3)

                # best_val_vector_nlazy = Float64[]
                # best_solution_nlazy = 0 * d
                # res_static_convhull_bound_nlazy = FWR.solve_robust_problem(c0, c0, d, Γ, lmo, max_iteration=iterations, max_lmo_calls = iterations, automatic_tolerance=false, line_search=FrankWolfe.Agnostic(), lazy=false, smoothing_parameter=smoothing_parameter, update_bound_convhull=true, best_val_vector=best_val_vector_nlazy, best_solution=best_solution_nlazy)
                # # res_static_convhull_iterate = FWR.solve_robust_problem(c0, c0, d, Γ, lmo, max_lmo_calls = iterations, automatic_tolerance=false, line_search=FrankWolfe.Agnostic(), lazy=true, smoothing_parameter=smoothing_parameter, update_bound_convhull=true, update_iterate_convhull=true)

                # trajectory_bound_nlazy = res_static_convhull_bound_nlazy[end]
                # val_trajectory_bound_nlazy = getindex.(trajectory_bound_nlazy, 2)
                # times_trajectory_bound_nlazy = getindex.(trajectory_bound_nlazy, 4)
                # n_lmos_bound_nlazy = getindex.(trajectory_bound_nlazy, 3)
                

                # plot(times_trajectory_bound, best_val_vector, label="best", yaxis=:log)
                # plot!(times_trajectory_bound, val_trajectory_bound, label="bound")
                # plot!(times_trajectory_static, val_trajectory_static, label="classic FW")

                # plot(val_trajectory_iterate[1:1000], width=3)
                # plot!(val_trajectory_bound[1:1000])
                # plot!(val_trajectory_static[1:1000])

                # plot(val_trajectory_iterate, yaxis=:log)
                # plot!(val_trajectory_bound, width=4)
                # plot!(val_trajectory_static)

                push!(results, Dict(
                    "n" => n,
                    "seed" => seed,
                    "gamma" => Γ,
                    "val_consgen" => obj_consgen,
                    "times_consgen" => times_consgen,
                    "val_static" => val_trajectory_static,
                    "times_static" => times_trajectory_static,
                    "lmos_static" => n_lmos_static,
                    "val_adaptive" => val_trajectory_adaptive,
                    "times_adaptive" => times_trajectory_adaptive,
                    "lmos_adaptive" => n_lmos_adaptive,
                    # "val_subgradient" => obj_subgradient,
                    # "times_subgradient" => times_subgradient,
                    "times_convhull_bound" => times_trajectory_bound,
                    "val_convhull_bound" => best_val_vector,
                    "lmos_convhull_bound" => n_lmos_bound,
                    "times_convhull_bound_adaptive" => times_trajectory_bound_adaptive,
                    "val_convhull_bound_adaptive" => best_val_vector_adaptive,
                    "lmos_convhull_bound_adaptive" => n_lmos_bound_adaptive,
                    # "times_convhull_bound_nlazy" => times_trajectory_bound_nlazy,
                    # "val_convhull_bound_nlazy" => best_val_vector_nlazy,
                    # "lmos_convhull_bound_nlazy" => n_lmos_bound_nlazy,
                ))
            end
        end
        extreme_str = extreme_scenarios ? "_ext" : ""
        filename = "results_ST_n_$(n)$(extreme_str).json"
        open(filename, "w") do f
            write(f, JSON.json(results, 4))
        end
        empty!(results)
    end
end
