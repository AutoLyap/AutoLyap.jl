using Test, Suppressor, JuMP

@suppress begin
    using AutoLyap # to turn off annoying solver info
end

@info "[💻 ] Running tests for AutoLyap.jl"

solver_val = :clarabel # options are :mosek, :clarabel, :cosmo, :scs, :copt

show_output_val = false # options are true or false

if solver_val == :mosek || solver_val == :clarabel
    if solver_val == :mosek
        @info "[🏀 ] Using Mosek solver for SDP, using 1e-5 for tolerance in ≈"
    elseif solver_val == :clarabel
        @info "[🏀 ] Using Clarabel solver for SDP, using 1e-5 for tolerance in ≈"
    end
    atol_input = 1e-5
elseif solver_val == :copt || solver_val == :scs || solver_val == :cosmo
    if solver_val == :copt
        @info "[🎮 ] Using COPT solver for SDP, using 1e-3 for tolerance in ≈"
    elseif solver_val == :scs
        @info "[🎮 ] Using SCS solver for SDP, using 1e-3 for tolerance in ≈"
    elseif solver_val == :cosmo
        @info "[🎮 ] Using COSMO solver for SDP, using 1e-3 for tolerance in ≈"
    end
    atol_input = 1e-3
else
    error("Unsupported solver: $solver_val. Supported solvers are :mosek, :clarabel, :copt, :scs, and :cosmo.")
end
    

# ## TEST CASE 1: MaximallyMonotone + (StronglyMonotone + Lipschitz)

@testset "DRS Linear Convergence (StronglyMonotone + Lipschitz)" begin

    # ----------------------------------------------------------------------
    # Step 1: Defining the Mathematical Problem
    # ----------------------------------------------------------------------

    # We can use the exported types directly
    g1_conditions = MaximallyMonotone()
    g2_conditions = [
        StronglyMonotone(mu = 1.0),
        LipschitzOperator(L = 2.0)
    ]
    components_list = [g1_conditions, g2_conditions]
    problem = InclusionProblem(components = components_list)

    # ----------------------------------------------------------------------
    # Step 2: Defining the Optimization Algorithm
    # ----------------------------------------------------------------------

    algorithm = DouglasRachford(gamma = 1.0, lambda_value = 2.0, operator_version=true)

    # ----------------------------------------------------------------------
    # Step 3: Defining the Performance Metric
    # ----------------------------------------------------------------------

    (P, T) = AutoLyap.IterationIndependent.get_parameters_distance_to_solution(algorithm)

    # ----------------------------------------------------------------------
    # Step 4: Finding the Performance Rate via an SDP
    # ----------------------------------------------------------------------

    rho = AutoLyap.IterationIndependent.bisection_search_rho(
        problem,
        algorithm,
        P,
        T;
        lower_bound=0.0,
        upper_bound=1.0,
        tol=1e-8,
        solver = solver_val,
        show_output = show_output_val
    )

    # ----------------------------------------------------------------------
    # Step 5: Add Tests
    # ----------------------------------------------------------------------

    println("[🎎 ] Computed DRS convergence rate (rho) for (StronglyMonotone + Lipschitz): $rho") #$

    @test rho ≈ 0.4285714266488867 atol = atol_input

end

@testset "DRS Linear Convergence (StronglyMonotone + Cocoercive)" begin
    # Step 1: Defining the Mathematical Problem
    g1_conditions = MaximallyMonotone()
    # The second component is now StronglyMonotone and Cocoercive
    g2_conditions = [
        StronglyMonotone(mu = 1.0),
        Cocoercive(beta = 0.5)
    ]
    components_list = [g1_conditions, g2_conditions]
    problem = InclusionProblem(components = components_list)

    # Step 2: Defining the Optimization Algorithm
    algorithm = DouglasRachford(gamma = 1.0, lambda_value = 2.0, operator_version=true)

    # Step 3: Defining the Performance Metric
    (P, T) = AutoLyap.IterationIndependent.get_parameters_distance_to_solution(algorithm)

    # Step 4: Finding the Performance Rate
    rho = AutoLyap.IterationIndependent.bisection_search_rho(
        problem,
        algorithm,
        P,
        T;
        lower_bound=0.0,
        upper_bound=1.0,
        tol=1e-8,
        solver = solver_val,
        show_output = show_output_val
    )

    # Step 5: Add Tests
    println("[💽 ] Computed ")
    println("DRS convergence rate (rho) for (StronglyMonotone + Cocoercive): $rho") #$

    @test rho ≈ 0.1999999878462404 atol = atol_input
end


# # ==============================================================================
# # NEW TEST CASE: GradientDominated + Smooth
# # ==============================================================================
@testset "Gradient Method Linear Convergence (GradientDominated + Smooth)" begin
    # Step 1: Defining the Mathematical Problem
    # A single component that is both GradientDominated and Smooth
    f_conditions = [
        GradientDominated(mu_gd = 0.5),
        Smooth(L = 1.0)
    ]
    components_list = [f_conditions]
    problem = InclusionProblem(components = components_list)

    # Step 2: Defining the Optimization Algorithm
    algorithm = GradientMethod(gamma = 1.0)

    # Step 3: Defining the Performance Metric
    # This metric requires p and t, so we expect 4 return values
    (P, p, T, t) = AutoLyap.IterationIndependent.get_parameters_linear_function_value_suboptimality(algorithm)

    # Step 4: Finding the Performance Rate
    # Note that p and t are passed as positional arguments
    rho = AutoLyap.IterationIndependent.bisection_search_rho(
        problem,
        algorithm,
        P,
        T,
        p,
        t;
        lower_bound=0.0,
        upper_bound=1.0,
        tol=1e-8,
        solver = solver_val,
        show_output = show_output_val
    )

    # Step 5: Add Tests
    println("[💾 ] Computed GM convergence rate (rho) for (GradientDominated + Smooth): $rho")

    @test rho ≈ 0.3999999387888238 atol = atol_input
end

# # ==============================================================================
# # NEW TEST CASE: Heavy-Ball on Smooth Convex
# # ==============================================================================
@testset "Heavy-Ball Sublinear Convergence (SmoothConvex)" begin
    # Step 1: Defining the Mathematical Problem
    components_list = [
        SmoothConvex(L = 1.0)
    ]
    problem = InclusionProblem(components = components_list)

    # Step 2: Defining the Optimization Algorithm
    algorithm = HeavyBallMethod(gamma = 1.0, delta = 0.5)

    # Step 3: Defining the Performance Metric
    # Use the new function for sublinear convergence
    (P, p, T, t) = AutoLyap.IterationIndependent.get_parameters_sublinear_function_value_suboptimality(algorithm)

    # Step 4: Verifying the Performance Rate
    # Directly call the verify function with rho=1 and remove_C4=false
    successful = AutoLyap.IterationIndependent.verify_iteration_independent_Lyapunov(
        problem,
        algorithm,
        P,
        T,
        p,
        t;
        rho=1.0,
        remove_C4=false,
        solver = solver_val,
        show_output = show_output_val
    )

    # Step 5: Add Tests
    println("[💮 ] Verified Heavy-Ball sublinear convergence: $successful")

    # The Python script's output `successful` is a boolean.
    @test successful == true
end

# ==============================================================================
# NEW TEST CASE: Heavy-Ball on GradientDominated + Smooth
# ==============================================================================
@testset "Heavy-Ball Linear Convergence (GradientDominated + Smooth)" begin
    # Step 1: Defining the Mathematical Problem
    # A single component that is both GradientDominated and Smooth
    f_conditions = [
        GradientDominated(mu_gd = 0.5),
        Smooth(L = 1.0)
    ]
    components_list = [f_conditions]
    problem = InclusionProblem(components = components_list)

    # Step 2: Defining the Optimization Algorithm
    algorithm = HeavyBallMethod(gamma = 1.0, delta = 0.5)

    # Step 3: Defining the Performance Metric
    # This is a linear convergence test for function-value suboptimality
    (P, p, T, t) = AutoLyap.IterationIndependent.get_parameters_linear_function_value_suboptimality(algorithm)

    # Step 4: Finding the Performance Rate
    # Note that p and t are passed as positional arguments
    rho = AutoLyap.IterationIndependent.bisection_search_rho(
        problem,
        algorithm,
        P,
        T,
        p,
        t;
        lower_bound=0.0,
        upper_bound=1.0,
        tol=1e-8,
        solver = solver_val,
        show_output = show_output_val
    )

    # Step 5: Add Tests
    println("[💧 ] Computed Heavy-Ball convergence rate (rho) for (GradientDominated + Smooth): $rho")

    # The expected value is obtained by running the Python script.
    @test rho ≈ 0.7090351881970491 atol = atol_input
end

# ==============================================================================
# NEW TEST CASE: Nesterov Momentum on GradientDominated + Smooth
# ==============================================================================
@testset "Nesterov Momentum Linear Convergence (GradientDominated + Smooth)" begin
    # Step 1: Defining the Mathematical Problem
    # A single component that is both GradientDominated and Smooth
    f_conditions = [
        GradientDominated(mu_gd = 0.5),
        Smooth(L = 1.0)
    ]
    components_list = [f_conditions]
    problem = InclusionProblem(components = components_list)

    # Step 2: Defining the Optimization Algorithm
    algorithm = GradientNesterovMomentum(gamma = 1.0, delta = 0.5)

    # Step 3: Defining the Performance Metric
    # This is a linear convergence test for function-value suboptimality
    (P, p, T, t) = AutoLyap.IterationIndependent.get_parameters_linear_function_value_suboptimality(algorithm)

    # Step 4: Finding the Performance Rate
    # Note that p and t are passed as positional arguments
    rho = AutoLyap.IterationIndependent.bisection_search_rho(
        problem,
        algorithm,
        P,
        T,
        p,
        t;
        lower_bound=0.0,
        upper_bound=1.0,
        tol=1e-8,
        solver = solver_val,
        show_output = show_output_val
    )

    # Step 5: Add Tests
    println("[👻 ] Computed Nesterov Momentum convergence rate (rho) for (GradientDominated + Smooth): $rho")

    # The expected value is obtained by running the Python script.
    @test rho ≈ 0.49999997817303665 atol = atol_input
end



# ==============================================================================
# NEW TEST CASE: Nesterov FGM on Smooth Convex (Iteration Dependent)
# ==============================================================================
@testset "Nesterov FGM Sublinear Convergence (SmoothConvex)" begin
    # Step 1: Defining the Mathematical Problem
    components_list = [
        SmoothConvex(L = 1.0)
    ]
    problem = InclusionProblem(components = components_list)

    # Step 2: Defining the Optimization Algorithm
    algorithm = NesterovFastGradientMethod(gamma = 1.0)

    # Step 3: Defining the Initial and Final Lyapunov Function Parameters
    # Get (Q_0, q_0) for the initial condition (distance to solution at k=0)
    (Q_0, q_0) = AutoLyap.IterationDependent.get_parameters_distance_to_solution(
        algorithm, 0; i=1, j=2
    )

    # Get (Q_K, q_K) for the final condition (function suboptimality at k=10)
    (Q_K, q_K) = AutoLyap.IterationDependent.get_parameters_function_value_suboptimality(
        algorithm, 10; j=2
    )

    # Step 4: Verifying the Performance Rate
    # Directly call the verify function for K=10 iterations
    (successful, c) = AutoLyap.IterationDependent.verify_iteration_dependent_Lyapunov(
        problem,
        algorithm,
        10, # Iteration budget K
        Q_0,
        Q_K,
        q_0,
        q_K;
        solver = solver_val,
        show_output = show_output_val
    )

    # Step 5: Add Tests
    println("[👓 ] Computed Nesterov FGM sublinear rate constant (c): $c")

    # The Python script outputs the constant `c`.
    @test successful == true
    @test !isnothing(c)
    # The expected value is obtained by running the Python script.
    @test c ≈ 0.011026816265736821 atol = atol_input
end

# ==============================================================================
# NEW TEST CASE: OGM on Smooth Convex (Iteration Dependent)
# ==============================================================================
@testset "OGM Sublinear Convergence (SmoothConvex)" begin
    # Step 1: Defining the Mathematical Problem
    components_list = [
        SmoothConvex(L = 1.0)
    ]
    problem = InclusionProblem(components = components_list)

    # Step 2: Defining the Optimization Algorithm
    algorithm = OptimizedGradientMethod(L = 1.0, K = 10) # L=1, K=10

    # Step 3: Defining the Initial and Final Lyapunov Function Parameters
    # Get (Q_0, q_0) for the initial condition (distance to solution at k=0)
    (Q_0, q_0) = AutoLyap.IterationDependent.get_parameters_distance_to_solution(
        algorithm, 0
    )

    # Get (Q_K, q_K) for the final condition (function suboptimality at k=K=10)
    (Q_K, q_K) = AutoLyap.IterationDependent.get_parameters_function_value_suboptimality(
        algorithm, 10
    )

    # Step 4: Verifying the Performance Rate
    # Directly call the verify function for K=10 iterations
    (successful, c) = AutoLyap.IterationDependent.verify_iteration_dependent_Lyapunov(
        problem,
        algorithm,
        10, # Iteration budget K
        Q_0,
        Q_K,
        q_0,
        q_K;
        solver = solver_val,
        show_output = show_output_val
    )

    # Step 5: Add Tests
    println("[👟 ] Computed OGM sublinear rate constant (c): $c")

    # The Python script outputs the constant `c`.
    @test successful == true
    @test !isnothing(c)
    # The expected value is obtained by running the Python script.
    @test c ≈ 0.0062865073712335415 atol = atol_input
end

# ==============================================================================
# TEST CASE: Chambolle-Pock on Convex + Convex
# ==============================================================================
@testset "Chambolle-Pock Sublinear Convergence (Convex + Convex)" begin
    # Step 1: Defining the Mathematical Problem
    # Problem is a sum of two convex functions
    components_list = [
        Convex(),
        Convex()
    ]
    problem = InclusionProblem(components = components_list)

    # Step 2: Defining the Optimization Algorithm
    algorithm = ChambollePock(tau = 1.0, sigma = 1.0, theta = 1.0)

    # Step 3: Defining the Performance Metric
    # Use the fixed_point_residual metric with h=1, alpha=1
    (P, p, T, t) = AutoLyap.IterationIndependent.get_parameters_fixed_point_residual(
        algorithm; h=1, alpha=1
    )

    # Step 4: Verifying the Performance Rate
    # Directly call the verify function with rho=1.0, h=1, alpha=1
    successful = AutoLyap.IterationIndependent.verify_iteration_independent_Lyapunov(
        problem,
        algorithm,
        P,
        T,
        p,
        t;
        rho=1.0,
        h=1,
        alpha=1,
        solver = solver_val,
        show_output = show_output_val
    )

    # Step 5: Add Tests
    println("[💐 ] Verified Chambolle-Pock sublinear convergence: $successful")

    # The Python script's output `successful` is a boolean.
    @test successful == true
end


# ==============================================================================
# NEW TEST CASE: Full-Model Return (Iteration-Independent)
# ==============================================================================
@testset "Iteration-Independent Full Model Return" begin
    f_conditions = [
        GradientDominated(mu_gd = 0.5),
        Smooth(L = 1.0)
    ]
    problem = InclusionProblem(components = [f_conditions])
    algorithm = GradientMethod(gamma = 1.0)

    (P, p, T, t) = AutoLyap.IterationIndependent.get_parameters_linear_function_value_suboptimality(algorithm)

    result = AutoLyap.IterationIndependent.verify_iteration_independent_Lyapunov(
        problem,
        algorithm,
        P,
        T,
        p,
        t;
        rho=1.0,
        solver=solver_val,
        show_output=show_output_val,
        return_full_model=true
    )

    @test result.successful == true
    @test result.status in (JuMP.OPTIMAL, JuMP.ALMOST_OPTIMAL)
    @test result.objective_value === nothing
    @test result.model isa JuMP.Model

    dv = result.decision_values
    @test dv.Q.is_decision == true
    @test dv.S.is_decision == true
    @test dv.q !== nothing
    @test dv.s !== nothing
    @test dv.q.is_decision == true
    @test dv.s.is_decision == true
    @test size(dv.Q.value) == size(P)
    @test size(dv.S.value) == size(T)
    @test length(dv.q.value) == length(p)
    @test length(dv.s.value) == length(t)
    @test all(isfinite, dv.Q.value)
    @test all(isfinite, dv.S.value)
    @test all(isfinite, dv.q.value)
    @test all(isfinite, dv.s.value)
    @test !isempty(dv.multipliers)
    @test all(m -> m.is_decision && isfinite(m.value), dv.multipliers)
end


# ==============================================================================
# NEW TEST CASE: Full-Model Return Fixed-Variable Metadata (Iteration-Independent)
# ==============================================================================
@testset "Iteration-Independent Full Model Fixed Metadata" begin
    f_conditions = [
        GradientDominated(mu_gd = 0.5),
        Smooth(L = 1.0)
    ]
    problem = InclusionProblem(components = [f_conditions])
    algorithm = GradientMethod(gamma = 1.0)

    (P, p, T, t) = AutoLyap.IterationIndependent.get_parameters_linear_function_value_suboptimality(algorithm)

    result = AutoLyap.IterationIndependent.verify_iteration_independent_Lyapunov(
        problem,
        algorithm,
        P,
        T,
        p,
        t;
        rho=1.0,
        Q_equals_P=true,
        S_equals_T=true,
        q_equals_p=true,
        s_equals_t=true,
        solver=solver_val,
        show_output=show_output_val,
        return_full_model=true
    )

    dv = result.decision_values
    @test dv.Q.is_decision == false
    @test dv.S.is_decision == false
    @test dv.q !== nothing
    @test dv.s !== nothing
    @test dv.q.is_decision == false
    @test dv.s.is_decision == false
    @test dv.Q.value ≈ Float64.(P) atol = atol_input
    @test dv.S.value ≈ Float64.(T) atol = atol_input
    @test dv.q.value ≈ Float64.(p) atol = atol_input
    @test dv.s.value ≈ Float64.(t) atol = atol_input
end


# ==============================================================================
# NEW TEST CASE: Full-Model Return (Iteration-Dependent)
# ==============================================================================
@testset "Iteration-Dependent Full Model Return" begin
    problem = InclusionProblem(components = [SmoothConvex(L = 1.0)])
    algorithm = OptimizedGradientMethod(L = 1.0, K = 10)

    (Q_0, q_0) = AutoLyap.IterationDependent.get_parameters_distance_to_solution(algorithm, 0)
    (Q_K, q_K) = AutoLyap.IterationDependent.get_parameters_function_value_suboptimality(algorithm, 10)

    result = AutoLyap.IterationDependent.verify_iteration_dependent_Lyapunov(
        problem,
        algorithm,
        10,
        Q_0,
        Q_K,
        q_0,
        q_K;
        solver=solver_val,
        show_output=show_output_val,
        return_full_model=true
    )

    @test result.successful == true
    @test result.status in (JuMP.OPTIMAL, JuMP.ALMOST_OPTIMAL)
    @test result.objective_value isa Float64
    @test isfinite(result.objective_value)
    @test result.model isa JuMP.Model

    dv = result.decision_values
    @test dv.c.is_decision == true
    @test dv.c.value ≈ result.objective_value atol = atol_input
    @test dv.Q[0].is_decision == false
    @test dv.Q[10].is_decision == false
    @test all(dv.Q[k].is_decision for k in 1:9)
    @test dv.Q[0].value ≈ Float64.(Q_0) atol = atol_input
    @test dv.Q[10].value ≈ Float64.(Q_K) atol = atol_input
    @test all(isfinite, dv.Q[5].value)
    @test dv.q !== nothing
    @test dv.q[0].is_decision == false
    @test dv.q[10].is_decision == false
    @test all(dv.q[k].is_decision for k in 1:9)
    @test dv.q[0].value ≈ Float64.(q_0) atol = atol_input
    @test dv.q[10].value ≈ Float64.(q_K) atol = atol_input
    @test all(isfinite, dv.q[5].value)
    @test !isempty(dv.multipliers)
    @test all(m -> m.is_decision && isfinite(m.value), dv.multipliers)
end


# ==============================================================================
# NEW TEST CASE: Full-Model Status Gate Helpers
# ==============================================================================
@testset "Full-Model Status Gate Helpers" begin
    @test AutoLyap.IterationIndependent._assert_full_model_status("tmp_fn", JuMP.OPTIMAL) === nothing
    @test AutoLyap.IterationIndependent._assert_full_model_status("tmp_fn", JuMP.ALMOST_OPTIMAL) === nothing
    @test_throws ErrorException AutoLyap.IterationIndependent._assert_full_model_status("tmp_fn", JuMP.INFEASIBLE)
    @test_throws ErrorException AutoLyap.IterationIndependent._assert_full_model_status("tmp_fn", JuMP.INFEASIBLE_OR_UNBOUNDED)

    @test AutoLyap.IterationDependent._assert_full_model_status("tmp_fn", JuMP.OPTIMAL) === nothing
    @test AutoLyap.IterationDependent._assert_full_model_status("tmp_fn", JuMP.ALMOST_OPTIMAL) === nothing
    @test_throws ErrorException AutoLyap.IterationDependent._assert_full_model_status("tmp_fn", JuMP.INFEASIBLE)
    @test_throws ErrorException AutoLyap.IterationDependent._assert_full_model_status("tmp_fn", JuMP.INFEASIBLE_OR_UNBOUNDED)
end
