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

import FrankWolfeRobustOpt as FWR
using Test
using HiGHS
using JuMP
using LinearAlgebra
using Graphs
using FrankWolfe

@testset "simple projection" begin
    n = 32
    b = 10 * rand()
    x = FWR.solve_quadratic_knapsack(zeros(n), ones(n), b, ones(n), ones(n), ones(n))
    @test all(≈(b / n), x)
end

@testset "projection with tight lower bound" begin
    n = 32
    b = 10 * rand()
    x = FWR.solve_quadratic_knapsack(0.5 * ones(n), ones(n), b, ones(n), ones(n), ones(n))
    @test all(≈(max(b / n, 0.5)), x)
end

@testset "projection with tight upper bound" begin
    n = 32
    b = 10 * rand()
    x = FWR.solve_quadratic_knapsack(zeros(n), 0.9 * (b / n) * ones(n), b, ones(n), ones(n), ones(n))
    @test all(≈(0.9 * (b / n)), x)
end

@testset "projection with tight upper bound $n" for n in (20, 30, 500)
    lower_bounds = rand(n)
    upper_bounds = lower_bounds + rand(n)
    b = (sum(lower_bounds) + sum(upper_bounds)) / 2
    d = 2 * rand(n)
    a = ones(n) + 0.05 * randn(n)
    q = ones(n)
    xsol = FWR.solve_quadratic_knapsack(lower_bounds, upper_bounds, b, a, d, q)

    # JuMP comparison
    model = Model(HiGHS.Optimizer)
    @variable(model, lower_bounds[i] ≤ x[i=1:n] ≤ upper_bounds[i])
    @constraint(model, dot(x, a) ≤ b)
    @objective(model, Min, sum(d[i]/2 * (x[i] - q[i])^2 for i in 1:n))
    optimize!(model)
    xval = JuMP.value.(x)
    @test norm(xval - xsol) / n ≤ 0.001
end

@testset "knapsack points" begin
    xref = [0.9675041973621596, 0.9179963080319086, 0.012761091920961212, 1.0, 0.014636426884945535, 0.032455807624169615, 0.03249580263784048, 0.03831827066913893, 0.029048994414007052, 0.9547831004548694]
    c0 = cref = [10.867872878972538, 11.608308095789033, 11.292927349765773, 10.832193220428326, 12.635075035544508, 12.41511068484226, 11.953316945432459, 14.013812756399865, 13.609913675203423, 11.861733146652188]
    gamma = 5
    d = [1.0958574242622292, 3.727029909601489, 10.948663147342861, 0.13971689683488275, 9.545833688315703, 4.3048350068120484, 4.299536724542579, 3.646221355897698, 4.809698223753765, 0.7836927412876049]
    c_closedform = FWR.solve_budgeted_inner_problem(d, gamma, xref, 0, c0, c0)
    c_jump = FWR.solve_budgeted_inner_problem_jump(d, gamma, xref, 0, c0, c0)
    @test dot(c_jump, xref) ≈ dot(c_closedform, xref) atol=1e-4

    xref = [0.983918098643819, 0.804114221875988, 0.804114221875988, 1.0, 0.016081901356181115, 0.0, 0.016081901356181115, 0.0, 0.1798038767678309, 0.19588577812401203]
    c0 = cref = [10.867872878972538, 11.608308095789033, 11.292927349765773, 10.832193220428326, 12.635075035544508, 12.41511068484226, 11.953316945432459, 14.013812756399865, 13.609913675203423, 11.861733146652188]
    d = [1.0958574242622292, 3.727029909601489, 10.948663147342861, 0.13971689683488275, 9.545833688315703, 4.3048350068120484, 4.299536724542579, 3.646221355897698, 4.809698223753765, 0.7836927412876049]
    c_closedform = FWR.solve_budgeted_inner_problem(d, gamma, xref, 0, c0, c0)
    c_jump = FWR.solve_budgeted_inner_problem_jump(d, gamma, xref, 0, c0, c0)
    @test dot(c_jump, xref) ≈ dot(c_closedform, xref) atol=1e-4
end
