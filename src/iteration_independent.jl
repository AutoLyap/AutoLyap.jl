# AutoLyap.jl/src/iteration_independent.jl

module IterationIndependent

using JuMP, Mosek, MosekTools, Clarabel, COSMO, SCS, LinearAlgebra, Combinatorics, Suppressor

@suppress begin
    using COPT
end



# Import our own modules
using ..ProblemClass
using ..Algorithms

export search_lyapunov, bisection_search_rho,
    # Performance metric generators
    get_parameters_distance_to_solution,
    get_parameters_linear_function_value_suboptimality,    
    get_parameters_sublinear_function_value_suboptimality, 
    get_parameters_fixed_point_residual,
    get_parameters_duality_gap,
    get_parameters_optimality_measure

# Internal helpers (not exported)

#=  
    _compute_Thetas(algo::Algorithm, h::Int, alpha::Int, condition::String)

Compute the Theta matrices (capital $\Theta$) using the X matrices.

For **condition "C1"**:

- Set $k_{\min} = 0$ and $k_{\max} = h+\alpha+1$.
- Retrieve $X = X_{\alpha+1}$ from `get_Xs(k_min, k_max)`.
- $\Theta_0$ is of size $[n + (h+1)\bar{m} + m] \times [n + (h+\alpha+2)\bar{m} + m]$.
- $\Theta_1$ is formed by vertically stacking $X$ with a block row consisting of a zero block and an identity matrix.

For **condition "C4"**:

- Set $k_{\min} = 0$ and $k_{\max} = h+\alpha+2$.
- Retrieve $X = X_1$ from `get_Xs(k_min, k_max)`.
- $\Theta_0$ is of size $[n + (h+\alpha+2)\bar{m} + m] \times [n + (h+\alpha+3)\bar{m} + m]$.
- $\Theta_1$ is formed similarly by stacking $X$ with an appropriate block row.

Input to the function:
==================      
algo: An instance of Algorithm (providing n, m_bar, m), Algorithm
h: A nonnegative integer, Int
alpha: A nonnegative integer, Int
condition: Either "C1" or "C4", String

Output to the function
=================== 
A tuple $(\Theta_0, \Theta_1)$.

=#
function _compute_Thetas(alg::Algorithm, h::Int, alpha::Int, condition::String)
    n, m_bar, m = alg.n, alg.m_bar, alg.m

    if condition == "C1"
        k_min, k_max = 0, h + alpha + 1
        Xs = get_Xs(alg, k_min, k_max)
        X_mat = Xs[alpha+1]

        Theta0 = [I(n + (h + 1) * m_bar) zeros(n + (h + 1) * m_bar, (alpha + 1) * m_bar + m);
            zeros(m, n + (h + 1) * m_bar + (alpha + 1) * m_bar) I(m)]

        lower_block = [zeros((h + 1) * m_bar + m, n + (alpha + 1) * m_bar) I((h + 1) * m_bar + m)]
        Theta1 = vcat(X_mat, lower_block)

        return Theta0, Theta1
    elseif condition == "C4"
        k_min, k_max = 0, h + alpha + 2
        Xs = get_Xs(alg, k_min, k_max)
        X_mat = Xs[1]

        Theta0 = [I(n + (h + alpha + 2) * m_bar) zeros(n + (h + alpha + 2) * m_bar, m_bar + m);
            zeros(m, n + (h + alpha + 2) * m_bar + m_bar) I(m)]

        lower_block = [zeros((h + alpha + 2) * m_bar + m, n + m_bar) I((h + alpha + 2) * m_bar + m)]
        Theta1 = vcat(X_mat, lower_block)

        return Theta0, Theta1
    else
        throw(ArgumentError("Condition must be 'C1' or 'C4'."))
    end
end

#=  
    _compute_thetas(algo::Algorithm, h::Int, alpha::Int, condition::String)

Compute the theta matrices (lowercase $\theta$) for functional evaluations.

For **condition "C1"**:

- $\theta_0 \in \mathbb{R}^{((h+1)\bar{m}_{\text{func}}+m_{\text{func}}) \times ((h+\alpha+2)\bar{m}_{\text{func}}+m_{\text{func}})}$ is given by a block matrix with an identity in the upper left and lower right.
- $\theta_1$ is formed as a horizontal block consisting of a zero block and an identity matrix.

For **condition "C4"**:

- $\theta_0 \in \mathbb{R}^{((h+\alpha+2)\bar{m}_{\text{func}}+m_{\text{func}}) \times ((h+\alpha+3)\bar{m}_{\text{func}}+m_{\text{func}})}$ is defined similarly.
- $\theta_1$ is a horizontal block with a zero block and an identity matrix.

Input to the function:
==================      
algo: An instance of Algorithm (providing $\bar{m}_{\text{func}}$ and $m_{\text{func}}$), Algorithm
h: A nonnegative integer, Int
alpha: A nonnegative integer, Int
condition: Either "C1" or "C4", String

Output to the function
=================== 
A tuple $(\theta_0, \theta_1)$.

=#

function _compute_thetas(alg::Algorithm, h::Int, alpha::Int, condition::String)
    m_bar_func, m_func = alg.m_bar_func, alg.m_func
    m_func > 0 || throw(ArgumentError("Theta matrices require at least one functional component."))

    if condition == "C1"
        theta0 = [I((h + 1) * m_bar_func) zeros((h + 1) * m_bar_func, (alpha + 1) * m_bar_func + m_func);
            zeros(m_func, (h + 1) * m_bar_func + (alpha + 1) * m_bar_func) I(m_func)]

        theta1 = [zeros((h + 1) * m_bar_func + m_func, (alpha + 1) * m_bar_func) I((h + 1) * m_bar_func + m_func)]

        return theta0, theta1
    elseif condition == "C4"
        theta0 = [I((h + alpha + 2) * m_bar_func) zeros((h + alpha + 2) * m_bar_func, m_bar_func + m_func);
            zeros(m_func, (h + alpha + 2) * m_bar_func + m_bar_func) I(m_func)]

        theta1 = [zeros((h + alpha + 2) * m_bar_func + m_func, m_bar_func) I((h + alpha + 2) * m_bar_func + m_func)]

        return theta0, theta1
    else
        throw(ArgumentError("Condition must be 'C1' or 'C4'."))
    end
end

function _assert_full_model_status(function_name::String, status)
    if status in (OPTIMAL, ALMOST_OPTIMAL)
        return nothing
    end
    throw(ErrorException("`$function_name` with `return_full_model=true` requires termination status OPTIMAL or ALMOST_OPTIMAL, got $(status)."))
end

function _to_float_matrix(data)
    mat = Matrix(data)
    if eltype(mat) <: VariableRef
        return Float64.(value.(mat))
    elseif eltype(mat) <: Number
        return Float64.(mat)
    end
    return Float64.(value.(mat))
end

function _to_float_vector(data)
    vec_data = collect(data)
    if eltype(vec_data) <: VariableRef
        return Float64.(value.(vec_data))
    elseif eltype(vec_data) <: Number
        return Float64.(vec_data)
    end
    return Float64.(value.(vec_data))
end

_to_float_scalar(x::VariableRef) = Float64(value(x))
_to_float_scalar(x::Number) = Float64(x)

function _is_mosek_license_error(e)
    return isdefined(Mosek, :LicenseError) && e isa getfield(Mosek, :LicenseError)
end

const _SEARCH_STATUS_FEASIBLE = "feasible"
const _SEARCH_STATUS_INFEASIBLE = "infeasible"
const _SEARCH_STATUS_NOT_SOLVED = "not_solved"

function _classify_search_status(status)
    if status in (OPTIMAL, ALMOST_OPTIMAL)
        return _SEARCH_STATUS_FEASIBLE
    elseif status in (INFEASIBLE, INFEASIBLE_OR_UNBOUNDED)
        return _SEARCH_STATUS_INFEASIBLE
    end
    return _SEARCH_STATUS_NOT_SOLVED
end

function _serialize_pair(pair::Tuple)
    j, k = pair
    j_val = (j == "star") ? "star" : Int(j)
    k_val = (k == "star") ? "star" : Int(k)
    return Dict{String,Any}("j" => j_val, "k" => k_val)
end

function _serialize_iteration_independent_certificate(
    Q,
    S,
    q,
    s,
    multiplier_records,
    algo::Algorithm
)
    operator_lambda = Vector{Dict{String,Any}}()
    function_lambda = Vector{Dict{String,Any}}()
    function_nu = Vector{Dict{String,Any}}()

    for rec in multiplier_records
        entry = Dict{String,Any}(
            "condition" => String(rec.condition),
            "component" => Int(rec.component),
            "interpolation_index" => Int(rec.interpolation_entry) - 1,
            "pairs" => [_serialize_pair(p) for p in rec.pairs],
            "value" => _to_float_scalar(rec.variable),
        )

        if rec.component in algo.I_op
            push!(operator_lambda, entry)
        elseif rec.is_equality_multiplier
            push!(function_nu, entry)
        else
            push!(function_lambda, entry)
        end
    end

    return Dict{String,Any}(
        "Q" => _to_float_matrix(Q),
        "S" => _to_float_matrix(S),
        "q" => (q === nothing ? nothing : _to_float_vector(q)),
        "s" => (s === nothing ? nothing : _to_float_vector(s)),
        "multipliers" => Dict{String,Any}(
            "operator_lambda" => operator_lambda,
            "function_lambda" => function_lambda,
            "function_nu" => function_nu,
        ),
    )
end

#=  
    search_lyapunov(prob::InclusionProblem, algo::Algorithm, P::Matrix, T::Matrix, p::Union{Vector, Nothing}=nothing, t::Union{Vector, Nothing}=nothing; rho::Float64=1.0, h::Int=0, alpha::Int=0, Q_equals_P=false, S_equals_T=false, q_equals_p=false, s_equals_t=false, remove_C2=false, remove_C3=false, remove_C4=true, solver=:clarabel, show_output=false, return_full_model=false)

Verify an iteration-independent Lyapunov inequality via an SDP.

This method sets up and solves a semidefinite program (SDP) using JuMP to verify a Lyapunov inequality for a given inclusion problem and algorithm.

Input to the function:
==================      
prob: An InclusionProblem instance containing interpolation conditions, InclusionProblem
algo: An Algorithm instance providing dimensions and methods to compute matrices, Algorithm
P: A symmetric matrix of dimension $n + (h+1)\bar{m} + m$ by $n + (h+1)\bar{m} + m$, Matrix
T: A symmetric matrix of dimension $n + (h+\alpha+2)\bar{m} + m$ by $n + (h+\alpha+2)\bar{m} + m$, Matrix
p: A vector for functional components (if any) with appropriate dimensions, Union{Vector, Nothing} (optional, default=nothing)
t: A vector for functional components (if any) with appropriate dimensions, Union{Vector, Nothing} (optional, default=nothing)
rho: A scalar contraction parameter used in forming the Lyapunov inequality, Float64 (keyword, default=1.0)
h: Nonnegative integer defining history, Int (keyword, default=0)
alpha: Nonnegative integer defining overlap, Int (keyword, default=0)
Q_equals_P: If true, sets Q equal to P, Bool (keyword, default=false)
S_equals_T: If true, sets S equal to T, Bool (keyword, default=false)
q_equals_p: For functional components, if true, sets q equal to p, Bool (keyword, default=false)
s_equals_t: For functional components, if true, sets s equal to t, Bool (keyword, default=false)
remove_C2: Flag to remove constraint C2, Bool (keyword, default=false)
remove_C3: Flag to remove constraint C3, Bool (keyword, default=false)
remove_C4: Flag to remove constraint C4, Bool (keyword, default=true)
solver: Solver to use (:mosek or :clarabel), Symbol (keyword, default=:clarabel)
show_output: Whether to show solver output (:on or :off), Symbol (keyword, default=:on)
return_full_model: If true, returns a full solution snapshot with status, decision-variable values, and model. If false, returns the legacy boolean output, Bool (keyword, default=false)

Output to the function
=================== 
Returns a dictionary with keys `"status"`, `"solve_status"`, `"rho"`, and `"certificate"`.
If `return_full_model == true`, the dictionary additionally contains `"full_model"` with Julia-specific debug payload.

=#

function search_lyapunov(
    prob::InclusionProblem,
    algo::Algorithm,
    P::Matrix, T::Matrix,
    p::Union{Vector,Nothing}=nothing, t::Union{Vector,Nothing}=nothing;
    rho::Float64=1.0, h::Int=0, alpha::Int=0,
    Q_equals_P=false, S_equals_T=false,
    q_equals_p=false, s_equals_t=false,
    remove_C2=false, remove_C3=false, remove_C4=true,
    solver=:mosek, # options are :mosek, :clarabel, :cosmo, :scs, :copt may add more later
    show_output=false, # options are true and false
    return_full_model=false
)
    # --- 1. Validation ---
    (prob.m == algo.m && Set(prob.I_func) == Set(algo.I_func) && Set(prob.I_op) == Set(algo.I_op)) || throw(ArgumentError("Problem and Algorithm definitions are inconsistent."))
    h >= 0 || throw(ArgumentError("h must be non-negative."))
    alpha >= 0 || throw(ArgumentError("alpha must be non-negative."))

    # --- 2. Dimensions ---
    n, m, m_bar, m_func, m_bar_func = algo.n, algo.m, algo.m_bar, algo.m_func, algo.m_bar_func
    dim_P = n + (h + 1) * m_bar + m
    dim_T = n + (h + alpha + 2) * m_bar + m
    size(P) == (dim_P, dim_P) || throw(DimensionMismatch("P has incorrect dimensions."))
    size(T) == (dim_T, dim_T) || throw(DimensionMismatch("T has incorrect dimensions."))

    # --- 3. JuMP Model Setup ---
    if solver == :mosek
        model = Model(Mosek.Optimizer)
    elseif solver == :clarabel
        model = Model(Clarabel.Optimizer)
    elseif solver == :cosmo
        @warn "[💀 ] Using COSMO solver. First-order SDP solvers such as COSMO, SCS are not as efficient as the second-order solvers (Mosek, Clarabel) based on our tests. We recommend the open-source Clarabel solver if you do not want to use any commercial solver such as Mosek. Use COSMO at your own risk."
        model = Model(COSMO.Optimizer)
    elseif solver == :scs
        @warn "[💀 ] Using SCS solver. First-order SDP solvers (eg COSMO, SCS) are not as efficient as the second-order solvers (eg Mosek, Clarabel) based on our tests.  We recommend the open-source Clarabel solver if you do not want to use any commercial solver such as Mosek. Use SCS at your own risk."
        model = Model(SCS.Optimizer)
    elseif solver == :copt
        @suppress begin
            model = Model(COPT.ConeOptimizer)
            set_attribute(model, "RelGap", 1e-6)
            set_attribute(model, "AbsGap", 1e-8)
            set_attribute(model, "FeasTol", 1e-8)
            set_attribute(model, "SDPMethod", 0) # will use the primal-dual interior-point method
        end
    else
        throw(ArgumentError("Unsupported solver: $solver. Supported solvers are :mosek, :clarabel, :copt, :scs, and :cosmo."))
    end

    if show_output == false
        set_silent(model)
    end


    # --- 4. JuMP Variables ---
    Q = Q_equals_P ? P : @variable(model, Q[1:dim_P, 1:dim_P], Symmetric)
    S = S_equals_T ? T : @variable(model, S[1:dim_T, 1:dim_T], Symmetric)

    q, s = nothing, nothing
    if m_func > 0
        dim_p = (h + 1) * m_bar_func + m_func
        dim_t = (h + alpha + 2) * m_bar_func + m_func
        isnothing(p) || size(p) == (dim_p,) || throw(DimensionMismatch("p has incorrect dimensions."))
        isnothing(t) || size(t) == (dim_t,) || throw(DimensionMismatch("t has incorrect dimensions."))
        q = q_equals_p ? p : @variable(model, q[1:dim_p])
        s = s_equals_t ? t : @variable(model, s[1:dim_t])
    end

    # --- 5. Build Main Constraints ---
    Ws, ws = Dict(), Dict()
    k_maxs = Dict()

    conds = ["C1"]
    if !remove_C2
        push!(conds, "C2")
    end
    if !remove_C3
        push!(conds, "C3")
    end
    if !remove_C4
        push!(conds, "C4")
    end

    for cond in conds
        if cond == "C1"
            Theta0_C1, Theta1_C1 = _compute_Thetas(algo, h, alpha, "C1")
            Ws[cond] = Theta1_C1' * Q * Theta1_C1 - rho * Theta0_C1' * Q * Theta0_C1 + S
            k_maxs[cond] = h + alpha + 1
            if m_func > 0
                theta0_C1, theta1_C1 = _compute_thetas(algo, h, alpha, "C1")
                ws[cond] = (theta1_C1' - rho * theta0_C1') * q + s
            end
        elseif cond == "C2"
            Ws[cond] = P - Q
            k_maxs[cond] = h
            if m_func > 0
                ws[cond] = p - q
            end
        elseif cond == "C3"
            Ws[cond] = T - S
            k_maxs[cond] = h + alpha + 1
            if m_func > 0
                ws[cond] = t - s
            end
        elseif cond == "C4"
            Theta0_C4, Theta1_C4 = _compute_Thetas(algo, h, alpha, "C4")
            Ws[cond] = Theta1_C4' * S * Theta1_C4 - Theta0_C4' * S * Theta0_C4
            k_maxs[cond] = h + alpha + 2
            if m_func > 0
                theta0_C4, theta1_C4 = _compute_thetas(algo, h, alpha, "C4")
                ws[cond] = (theta1_C4' - theta0_C4') * s
            end
        end
    end

    # --- 6. Add Interpolation Constraints ---
    PSD_sum = Dict{String, Any}(c => -Ws[c] for c in conds)
    EQ_sum = m_func > 0 ? Dict{String, Any}(c => -ws[c] for c in conds) : Dict{String, Any}()

    multiplier_records = NamedTuple[]

    for cond in conds, i in 1:m
        interp_data_list = get_component_data(prob, i)
        for (o, interp_data) in enumerate(interp_data_list)
            k_max = k_maxs[cond]
            # Generate pairs of (j,k) indices
            all_pairs = Vector{Any}()
            for j in 1:algo.m_bar_is[i], k in 0:k_max
                push!(all_pairs, (j, k))
            end
            push!(all_pairs, ("star", "star"))

            idx, M, a, eq = if i in algo.I_op
                interp_data[2], interp_data[1], nothing, false
            else
                interp_data[4], interp_data[1], interp_data[2], interp_data[3]
            end

            # Determine which combinations of pairs to iterate over
            pair_combos = if idx == idx_i_lt_j
                combinations(all_pairs, 2)
            elseif idx == idx_i_ne_j
                # `collect` the lazy iterator before filtering
                filter(p -> p[1] != p[2], collect(Base.product(all_pairs, all_pairs)))
            elseif idx == idx_i_only
                [[p] for p in all_pairs]
            elseif idx == idx_i_ne_star
                j_k_pairs = filter(p -> p != ("star", "star"), all_pairs)
                [[p, ("star", "star")] for p in j_k_pairs]
            end

            # Add constraints for each combination
            for pairs in pair_combos
                pairs_vec = [p for p in pairs]
                W_mat = compute_W(algo, i, pairs_vec, 0, k_max, M)
                mult = eq ? @variable(model) : @variable(model, lower_bound = 0)
                push!(multiplier_records, (
                    id=length(multiplier_records) + 1,
                    condition=cond,
                    component=i,
                    interpolation_entry=o,
                    pairs=copy(pairs_vec),
                    is_equality_multiplier=eq,
                    variable=mult
                ))

                PSD_sum[cond] += mult * W_mat
                if i in algo.I_func
                    F_vec = compute_F_aggregated(algo, i, pairs_vec, 0, k_max, a)
                    EQ_sum[cond] += mult * F_vec
                end
            end
        end
    end

    # --- 7. Finalize and Solve ---
    for cond in conds
        dim = size(PSD_sum[cond], 1)
        @constraint(model, PSD_sum[cond] in PSDCone())
        if m_func > 0
            @constraint(model, EQ_sum[cond] .== 0)
        end
    end

    try
        optimize!(model)
        status = termination_status(model)
        if show_output == true
           @info "[🎯 ] termination_status is $(status)"
        end

        classified_status = _classify_search_status(status)
        solve_status = string(status)

        if classified_status != _SEARCH_STATUS_FEASIBLE
            return Dict{String,Any}(
                "status" => classified_status,
                "solve_status" => solve_status,
                "rho" => Float64(rho),
                "certificate" => nothing,
            )
        end

        certificate = _serialize_iteration_independent_certificate(Q, S, q, s, multiplier_records, algo)

        result = Dict{String,Any}(
            "status" => _SEARCH_STATUS_FEASIBLE,
            "solve_status" => solve_status,
            "rho" => Float64(rho),
            "certificate" => certificate,
        )

        if return_full_model
            _assert_full_model_status("search_lyapunov", status)

            q_entry = nothing
            s_entry = nothing
            if m_func > 0
                q_entry = q === nothing ? nothing : (value=_to_float_vector(q), is_decision=!q_equals_p)
                s_entry = s === nothing ? nothing : (value=_to_float_vector(s), is_decision=!s_equals_t)
            end

            multipliers = [(
                id=rec.id,
                condition=rec.condition,
                component=rec.component,
                interpolation_entry=rec.interpolation_entry,
                pairs=rec.pairs,
                is_equality_multiplier=rec.is_equality_multiplier,
                value=_to_float_scalar(rec.variable),
                is_decision=true
            ) for rec in multiplier_records]

            decision_values = (
                Q=(value=_to_float_matrix(Q), is_decision=!Q_equals_P),
                S=(value=_to_float_matrix(S), is_decision=!S_equals_T),
                q=q_entry,
                s=s_entry,
                multipliers=multipliers
            )

            result["full_model"] = (
                successful=true,
                status=status,
                objective_value=nothing,
                decision_values=decision_values,
                model=model
            )
        end

        return result
    catch e
        # Preserve existing behavior for explicit license errors.
        if _is_mosek_license_error(e)
            rethrow(e)
        end

        return Dict{String,Any}(
            "status" => _SEARCH_STATUS_NOT_SOLVED,
            "solve_status" => "solver_error",
            "rho" => Float64(rho),
            "certificate" => nothing,
        )
    end
end

function verify_iteration_independent_Lyapunov(args...; kwargs...)
    throw(ErrorException(
        "IterationIndependent.verify_iteration_independent_Lyapunov was removed. " *
        "Use IterationIndependent.search_lyapunov instead."
    ))
end

#=  
    bisection_search_rho(prob::InclusionProblem, algo::Algorithm, P::Matrix, T::Matrix, p::Union{Vector, Nothing}=nothing, t::Union{Vector, Nothing}=nothing; lower_bound::Float64=0.0, upper_bound::Float64=1.0, tol::Float64=1e-12, h::Int=0, alpha::Int=0, Q_equals_P::Bool=false, S_equals_T::Bool=false, q_equals_p::Bool=false, s_equals_t::Bool=false, remove_C2::Bool=false, remove_C3::Bool=false, remove_C4::Bool=true)

Perform a bisection search to find the minimal contraction parameter $\rho$.

This method performs a bisection search over $\rho$ in the interval $[\text{lower\_bound}, \text{upper\_bound}]$ to find the minimal value for which the iteration-independent Lyapunov inequality holds. At each iteration it calls `search_lyapunov` until the interval size is below $\text{tol}$.

Input to the function:
==================      
prob: An InclusionProblem instance containing interpolation conditions, InclusionProblem
algo: An Algorithm instance providing dimensions and methods, Algorithm
P: A symmetric matrix of dimension $n + (h+1)\bar{m} + m$ by $n + (h+1)\bar{m} + m$, Matrix
T: A symmetric matrix of dimension $n + (h+\alpha+2)\bar{m} + m$ by $n + (h+\alpha+2)\bar{m} + m$, Matrix
p: A vector for functional components (if applicable), Union{Vector, Nothing} (optional, default=nothing)
t: A vector for functional components (if applicable), Union{Vector, Nothing} (optional, default=nothing)
lower_bound: Lower bound for $\rho$, Float64 (keyword, default=0.0)
upper_bound: Upper bound for $\rho$, Float64 (keyword, default=1.0)
tol: Tolerance for the bisection search stopping criterion, Float64 (keyword, default=1e-12)
h: Nonnegative integer defining the history for the matrices, Int (keyword, default=0)
alpha: Nonnegative integer for extending the horizon, Int (keyword, default=0)
Q_equals_P: If true, set Q equal to P, Bool (keyword, default=false)
S_equals_T: If true, set S equal to T, Bool (keyword, default=false)
q_equals_p: For functional components, if true, set q equal to p, Bool (keyword, default=false)
s_equals_t: For functional components, if true, set s equal to t, Bool (keyword, default=false)
remove_C2: Flag to remove constraint C2, Bool (keyword, default=false)
remove_C3: Flag to remove constraint C3, Bool (keyword, default=false)
remove_C4: Flag to remove constraint C4, Bool (keyword, default=true)

Output to the function
=================== 
Returns a dictionary with keys `"status"`, `"solve_status"`, `"rho"`, and `"certificate"`.
If `"status" == "feasible"`, then `"rho"` contains the terminal bisection value and `"certificate"` contains the feasibility certificate.
Otherwise, `"rho"` and `"certificate"` are `nothing`.

=#

function bisection_search_rho(
    prob, algo, P, T, p=nothing, t=nothing;
    lower_bound=0.0, upper_bound=1.0, tol=1e-12,
    # Pass-through keyword arguments for search_lyapunov
    h=0, alpha=0, Q_equals_P=false, S_equals_T=false,
    q_equals_p=false, s_equals_t=false,
    remove_C2=false, remove_C3=false, remove_C4=true,
    solver = :clarabel, # options are :mosek, :clarabel, :cosmo, :scs, :copt may add more later
    show_output=false # options are true and false
)

    # Define the pass-through arguments in a NamedTuple for clarity
    search_kwargs = (; h, alpha, Q_equals_P, S_equals_T, q_equals_p, s_equals_t,
        remove_C2, remove_C3, remove_C4)

    # Check if a solution exists at the upper bound
    # We splat the search_kwargs into the call
    upper_result = search_lyapunov(prob, algo, P, T, p, t; rho=upper_bound, solver = solver, show_output = show_output, search_kwargs...)
    if upper_result["status"] != _SEARCH_STATUS_FEASIBLE
        return Dict{String,Any}(
            "status" => upper_result["status"],
            "solve_status" => upper_result["solve_status"],
            "rho" => nothing,
            "certificate" => nothing,
        )
    end

    l, u = lower_bound, upper_bound
    while (u - l) > tol
        mid = (l + u) / 2
        mid_result = search_lyapunov(prob, algo, P, T, p, t; rho=mid, solver = solver, show_output = show_output, search_kwargs...)
        if mid_result["status"] == _SEARCH_STATUS_FEASIBLE
            u = mid
        else
            l = mid
        end
    end

    terminal_result = search_lyapunov(
        prob, algo, P, T, p, t;
        rho=u, solver = solver, show_output = show_output, search_kwargs...
    )

    if terminal_result["status"] == _SEARCH_STATUS_FEASIBLE
        return Dict{String,Any}(
            "status" => _SEARCH_STATUS_FEASIBLE,
            "solve_status" => terminal_result["solve_status"],
            "rho" => Float64(u),
            "certificate" => terminal_result["certificate"],
        )
    end

    return Dict{String,Any}(
        "status" => terminal_result["status"],
        "solve_status" => terminal_result["solve_status"],
        "rho" => nothing,
        "certificate" => nothing,
    )
end


# ===========================================================================
# PERFORMANCE METRIC GENERATORS
# ===========================================================================

# --- Linear Convergence Metrics ---

#=  
    get_parameters_distance_to_solution(algo::Algorithm; h=0, alpha=0, i=1, j=1, tau=0)

Compute the matrices for the distance to solution.

This method computes the following matrix:

$$P = \left( P_{(i,j)}\, Y_\tau^{0,h} - P_{(i,\star)}\, Y_\star^{0,h} \right)^T
    \left( P_{(i,j)}\, Y_\tau^{0,h} - P_{(i,\star)}\, Y_\star^{0,h} \right),$$

and constructs $T$ (and, if functional components exist, the vectors $p$ and $t$) as zero.

**Definitions**

- $Y_\tau^{0,h}$ is the Y matrix at iteration $\tau$ over the horizon $[0, h]$.
- $Y_\star^{0,h}$ is the "star" Y matrix over $[0, h]$.
- $P_{(i,j)}$ and $P_{(i,\star)}$ are the projection matrices for component $i$.

**Dimensions**

- $\dim(P) = n + (h+1)\bar{m} + m,$
- $\dim(T) = n + (h+\alpha+2)\bar{m} + m,$
- If $\text{algo.m\_func} > 0$:
  - $\mathrm{len}(p) = (h+1)\bar{m}_{\text{func}} + m_{\text{func}},$
  - $\mathrm{len}(t) = (h+\alpha+2)\bar{m}_{\text{func}} + m_{\text{func}}.$

Input to the function:
==================      
algo: An instance of Algorithm. It must provide: algo.m (total number of components), algo.m_bar_is (a list where the i-th entry gives the number of evaluations for component i), Methods get_Ys(k_min, k_max) and get_Ps(), Algorithm
h: A nonnegative integer defining the time horizon [0, h] for Y matrices, integer (keyword, default=0)
alpha: A nonnegative integer for extending the horizon for T (and t), integer (keyword, default=0)
i: Component index (1-indexed). Must satisfy 1 ≤ i ≤ algo.m, integer (keyword, default=1)
j: Evaluation index for component i. Must satisfy 1 ≤ j ≤ algo.m_bar_is[i], integer (keyword, default=1)
tau: Iteration index. Must satisfy 0 ≤ tau ≤ h, integer (keyword, default=0)

Output to the function
=================== 
If algo.m_func == 0: A tuple (P, T)
Otherwise: A tuple (P, p, T, t), where P is computed as above and T, p, and t are zero arrays with the appropriate dimensions.

=#

function get_parameters_distance_to_solution(algo::Algorithm; h=0, alpha=0, i=1, j=1, tau=0)
    Ys = get_Ys(algo, 0, h)
    Ps = get_Ps(algo)
    diff = Ps[(i, j)] * Ys[tau] - Ps[(i, "star")] * Ys["star"]
    P_mat = diff' * diff

    dim_T = algo.n + (h + alpha + 2) * algo.m_bar + algo.m
    T_mat = zeros(dim_T, dim_T)

    if algo.m_func > 0
        dim_p = (h + 1) * algo.m_bar_func + algo.m_func
        dim_t = (h + alpha + 2) * algo.m_bar_func + algo.m_func
        p_vec, t_vec = zeros(dim_p), zeros(dim_t)
        return P_mat, p_vec, T_mat, t_vec
    else
        return P_mat, T_mat
    end
end

#=  
    get_parameters_linear_function_value_suboptimality(algo::Algorithm; h=0, alpha=0, j=1, tau=0)

Compute the matrices/vectors for function-value suboptimality.

This function is only applicable when $m = m_{\text{func}} = 1$.

It returns a tuple $(P, p, T, t)$ where:

- $p$ is computed as

  $$p = \left( F_{(1,j,\tau)}^{0,h} - F_{(1,\star,\star)}^{0,h} \right)^T,$$

  with $p$ returned as a 1D array of length $(h+1)\bar{m}_{\text{func}} + m_{\text{func}}$.
- $P$ is a zero matrix of dimension

  $$n + (h+1)\bar{m} + m,$$

- $T$ is a zero matrix of dimension

  $$n + (h+\alpha+2)\bar{m} + m,$$

- $t$ is a zero vector of dimension

  $$(h+\alpha+2)\bar{m}_{\text{func}} + m_{\text{func}}.$$

Input to the function:
==================      
algo: An instance of Algorithm. It must satisfy algo.m = 1 and algo.m_func = 1, Algorithm
h: A nonnegative integer defining the horizon [0, h] for F matrices, integer (keyword, default=0)
alpha: A nonnegative integer for extending the horizon for T and t, integer (keyword, default=0)
j: Evaluation index for component 1. Must satisfy 1 ≤ j ≤ algo.m_bar_is[1], integer (keyword, default=1)
tau: Iteration index. Must satisfy 0 ≤ tau ≤ h, integer (keyword, default=0)

Output to the function
=================== 
A tuple (P, p, T, t), where p is computed as above (a 1D array) and P, T, and t are zero arrays with appropriate dimensions.

=#

function get_parameters_linear_function_value_suboptimality(algo::Algorithm; h=0, alpha=0, j=1, tau=0)
    (algo.m == 1 && algo.m_func == 1) || throw(ArgumentError("Only applicable when m=m_func=1."))

    Fs = get_Fs(algo, 0, h)
    # p_vec = vec((Fs[(1, j, tau)] - Fs[(1, "star", "star")])')
    p_vec = Vector(vec((Fs[(1, j, tau)] - Fs[(1, "star", "star")])'))

    dim_P = algo.n + (h + 1) * algo.m_bar + algo.m
    dim_T = algo.n + (h + alpha + 2) * algo.m_bar + algo.m
    dim_t = (h + alpha + 2) * algo.m_bar_func + algo.m_func

    return zeros(dim_P, dim_P), p_vec, zeros(dim_T, dim_T), zeros(dim_t)
end

#=  
    get_parameters_sublinear_function_value_suboptimality(algo::Algorithm; h=0, alpha=0, j=1, tau=0)

Compute the matrices/vectors for sublinear function-value suboptimality.

This function is only applicable when $m = m_{\text{func}} = 1$.

It returns a tuple $(P, p, T, t)$ where:

- $t$ is computed as

  $$t = \left( F_{(1,j,\tau)}^{0,h+\alpha+1} - F_{(1,\star,\star)}^{0,h+\alpha+1} \right)^T,$$

  with $t$ returned as a 1D array of length $(h+\alpha+2)\bar{m}_{\text{func}} + m_{\text{func}}$.
- $P$ is a zero matrix of dimension

  $$n + (h+1)\bar{m} + m,$$

- $p$ is a zero vector of dimension

  $$(h+1)\bar{m}_{\text{func}} + m_{\text{func}},$$

- $T$ is a zero matrix of dimension

  $$n + (h+\alpha+2)\bar{m} + m.$$

Input to the function:
==================      
algo: An instance of Algorithm. It must satisfy algo.m = 1 and algo.m_func = 1, Algorithm
h: A nonnegative integer defining the base horizon for the extended horizon [0, h+alpha+1] for F matrices, integer (keyword, default=0)
alpha: A nonnegative integer for extending the horizon for T and t, integer (keyword, default=0)
j: Evaluation index for component 1. Must satisfy 1 ≤ j ≤ algo.m_bar_is[1], integer (keyword, default=1)
tau: Iteration index. Must satisfy 0 ≤ tau ≤ h+alpha+1, integer (keyword, default=0)

Output to the function
=================== 
A tuple (P, p, T, t), where t is computed as above (a 1D array) and P, p, and T are zero arrays with appropriate dimensions.

=#

function get_parameters_sublinear_function_value_suboptimality(algo::Algorithm; h=0, alpha=0, j=1, tau=0)
    (algo.m == 1 && algo.m_func == 1) || throw(ArgumentError("Only applicable when m=m_func=1."))

    k_max = h + alpha + 1
    Fs = get_Fs(algo, 0, k_max)

    t_vec_row = Fs[(1, j, tau)] - Fs[(1, "star", "star")]
    t_vec = Vector(vec(t_vec_row'))

    dim_P = algo.n + (h + 1) * algo.m_bar + algo.m
    dim_p = (h + 1) * algo.m_bar_func + algo.m_func
    dim_T = algo.n + (k_max + 1) * algo.m_bar + algo.m

    return zeros(dim_P, dim_P), zeros(dim_p), zeros(dim_T, dim_T), t_vec
end

# --- Sublinear Convergence Metrics ---

#=  
    get_parameters_fixed_point_residual(algo::Algorithm; h=0, alpha=0, tau=0)

Compute the matrices for the fixed-point residual.

For a given iteration index $\tau$ (with $0 \le \tau \le h+\alpha+1$), define

$$T = \left( X_{\tau+1}^{0, h+\alpha+1} - X_{\tau}^{0, h+\alpha+1} \right)^T
    \left( X_{\tau+1}^{0, h+\alpha+1} - X_{\tau}^{0, h+\alpha+1} \right),$$

where $X_{\tau}^{0, h+\alpha+1}$ is the X matrix computed over the horizon $[0, h+\alpha+1]$ (via `get_Xs`).

**Dimensions**

- $P$ is a zero matrix of dimension $n + (h+1)\bar{m} + m$.
- $T$ is computed as above and has dimension $n + (h+\alpha+2)\bar{m} + m$.
- If $\text{algo.m\_func} > 0$:
  - $p$ is a zero vector of length $(h+1)\bar{m}_{\text{func}} + m_{\text{func}},$
  - $t$ is a zero vector of length $(h+\alpha+2)\bar{m}_{\text{func}} + m_{\text{func}}.$

Input to the function:
==================      
algo: An instance of Algorithm, Algorithm
h: A nonnegative integer defining the time horizon [0, h] for P, integer (keyword, default=0)
alpha: A nonnegative integer for extending the horizon for T (and t), integer (keyword, default=0)
tau: Iteration index for computing the fixed-point residual. Must satisfy 0 ≤ tau ≤ h+alpha+1, integer (keyword, default=0)

Output to the function
=================== 
If algo.m_func == 0: A tuple (P, T)
Otherwise: A tuple (P, p, T, t)

=#

function get_parameters_fixed_point_residual(algo::Algorithm; h=0, alpha=0, tau=0)
    k_max = h + alpha + 1
    Xs = get_Xs(algo, 0, k_max)
    diff = Xs[tau+1] - Xs[tau]
    T_mat = diff' * diff

    dim_P = algo.n + (h + 1) * algo.m_bar + algo.m
    P_mat = zeros(dim_P, dim_P)

    if algo.m_func > 0
        dim_p = (h + 1) * algo.m_bar_func + algo.m_func
        dim_t = (k_max + 1) * algo.m_bar_func + algo.m_func
        p_vec, t_vec = zeros(dim_p), zeros(dim_t)
        return P_mat, p_vec, T_mat, t_vec
    else
        return P_mat, T_mat
    end
end

#=  
    get_parameters_duality_gap(algo::Algorithm; h=0, alpha=0, tau=0)

Compute the matrices for the duality gap.

For a given iteration index $\tau$ (with $0 \le \tau \le h+\alpha+1$), define

$$T = -\frac{1}{2} \sum_{i=1}^{m} \begin{bmatrix}
P_{(i,\star)}\, U_\star^{0,h+\alpha+1} \\
P_{(i,1)}\, Y_\tau^{0,h+\alpha+1}
\end{bmatrix}^T
\begin{bmatrix}
0 & 1 \\
1 & 0 
\end{bmatrix}
\begin{bmatrix}
P_{(i,\star)}\, U_\star^{0,h+\alpha+1} \\
P_{(i,1)}\, Y_\tau^{0,h+\alpha+1}
\end{bmatrix},$$

and

$$t = \sum_{i=1}^{m} \left( F_{(i,1,\tau)}^{0,h+\alpha+1} - F_{(i,\star,\star)}^{0,h+\alpha+1} \right)^T.$$

All other matrices are set to zero.

**Requirements**

It is required that $m = m_{\text{func}}$ (i.e. all components are functional).

**Dimensions**

- $P$ is a zero matrix of dimension $n + (h+1)\bar{m} + m$.
- $p$ is a zero vector of length $(h+1)\bar{m}_{\text{func}} + m_{\text{func}}$.
- $T$ is computed as above and has dimension $n + (h+\alpha+2)\bar{m} + m$.
- $t$ is computed as above and has length $(h+\alpha+2)\bar{m}_{\text{func}} + m_{\text{func}}$.

Input to the function:
==================      
algo: An instance of Algorithm (with m = m_func), Algorithm
h: A nonnegative integer defining the time horizon [0, h] for P, integer (keyword, default=0)
alpha: A nonnegative integer for extending the horizon for T and t, integer (keyword, default=0)
tau: Iteration index for computing the duality gap. Must satisfy 0 ≤ tau ≤ h+alpha+1, integer (keyword, default=0)

Output to the function
=================== 
A tuple (P, p, T, t), where t is a one-dimensional array.

=#

function get_parameters_duality_gap(algo::Algorithm; h=0, alpha=0, tau=0)
    algo.m == algo.m_func || throw(ArgumentError("Only applicable when m=m_func."))
    k_max = h + alpha + 1

    U_dict, Y_dict, Fs_dict = get_Us(algo, 0, k_max), get_Ys(algo, 0, k_max), get_Fs(algo, 0, k_max)
    Ps = get_Ps(algo)
    mid = [0 1; 1 0]

    dim_T = algo.n + (k_max + 1) * algo.m_bar + algo.m
    T_sum = zeros(dim_T, dim_T)
    for i in 1:algo.m
        block = vcat(Ps[(i, "star")] * U_dict["star"], Ps[(i, 1)] * Y_dict[tau])
        T_sum += block' * mid * block
    end
    T_mat = -0.5 * T_sum

    dim_t = (k_max + 1) * algo.m_bar_func + algo.m_func
    t_sum = zeros(dim_t)
    for i in 1:algo.m
        t_sum .+= vec((Fs_dict[(i, 1, tau)] - Fs_dict[(i, "star", "star")])')
    end

    dim_P = algo.n + (h + 1) * algo.m_bar + algo.m
    dim_p = (h + 1) * algo.m_bar_func + algo.m_func

    return zeros(dim_P, dim_P), zeros(dim_p), T_mat, t_sum
end

#=  
    get_parameters_optimality_measure(algo::Algorithm; h=0, alpha=0, tau=0)

Compute the matrices for the optimality measure.

For a given iteration index $\tau$ (with $0 \le \tau \le h+\alpha+1$), define

$$T =
\begin{cases}
  \left( P_{(1,1)}\, U_\tau^{0,h+\alpha+1} \right)^T \left( P_{(1,1)}\, U_\tau^{0,h+\alpha+1} \right)
  & \text{if } m = 1, \\[1em]
  \left( \left( \sum_{i=1}^{m} P_{(i,1)}\, U_\tau^{0,h+\alpha+1} \right)^T \left( \sum_{i=1}^{m} P_{(i,1)}\, U_\tau^{0,h+\alpha+1} \right)
  + \sum_{i=2}^{m} \left( \left( P_{(1,1)} - P_{(i,1)} \right) Y_\tau^{0,h+\alpha+1} \right)^T \left( \left( P_{(1,1)} - P_{(i,1)} \right) Y_\tau^{0,h+\alpha+1} \right) \right)
  & \text{if } m > 1.
\end{cases}$$

All other matrices are set to zero.

**Dimensions**

- $P$ is a zero matrix of dimension $n + (h+1)\bar{m} + m$.
- If $\text{algo.m\_func} > 0$:
  - $p$ is a zero vector of length $(h+1)\bar{m}_{\text{func}} + m_{\text{func}},$
  - $t$ is a zero vector of length $(h+\alpha+2)\bar{m}_{\text{func}} + m_{\text{func}}.$

Input to the function:
==================      
algo: An instance of Algorithm, Algorithm
h: A nonnegative integer defining the time horizon [0, h] for P, integer (keyword, default=0)
alpha: A nonnegative integer for extending the horizon for T (and t), integer (keyword, default=0)
tau: Iteration index for computing the optimality measure. Must satisfy 0 ≤ tau ≤ h+alpha+1, integer (keyword, default=0)

Output to the function
=================== 
If algo.m_func == 0: A tuple (P, T)
Otherwise: A tuple (P, p, T, t)

=#

function get_parameters_optimality_measure(algo::Algorithm; h=0, alpha=0, tau=0)
    k_max = h + alpha + 1
    U_tau = get_Us(algo, 0, k_max)[tau]
    Y_tau = get_Ys(algo, 0, k_max)[tau]
    Ps = get_Ps(algo)

    dim_T = algo.n + (k_max + 1) * algo.m_bar + algo.m
    if algo.m == 1
        block = Ps[(1, 1)] * U_tau
        T_mat = block' * block
    else
        sum_U = sum(Ps[(i, 1)] * U_tau for i in 1:algo.m)
        term1 = sum_U' * sum_U

        sum_Y = sum(((Ps[(1, 1)] - Ps[(i, 1)]) * Y_tau)' * ((Ps[(1, 1)] - Ps[(i, 1)]) * Y_tau) for i in 2:algo.m)
        T_mat = term1 + sum_Y
    end

    dim_P = algo.n + (h + 1) * algo.m_bar + algo.m
    P_mat = zeros(dim_P, dim_P)

    if algo.m_func > 0
        dim_p = (h + 1) * algo.m_bar_func + algo.m_func
        dim_t = (k_max + 1) * algo.m_bar_func + algo.m_func
        p_vec, t_vec = zeros(dim_p), zeros(dim_t)
        return P_mat, p_vec, T_mat, t_vec
    else
        return P_mat, T_mat
    end
end


end # module IterationIndependent
