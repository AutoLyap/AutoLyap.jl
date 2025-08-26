# AutoLyap.jl/src/iteration_dependent.jl

module IterationDependent



using JuMP, Mosek, MosekTools, Clarabel, COSMO, SCS, LinearAlgebra, Combinatorics, Suppressor

@suppress begin
    using COPT
end

using ..ProblemClass
using ..Algorithms

export verify_iteration_dependent_Lyapunov,
    # Performance metric generators
    get_parameters_distance_to_solution,
    get_parameters_function_value_suboptimality,
    get_parameters_fixed_point_residual,
    get_parameters_optimality_measure


# Internal helpers (not exported)

#=  
    _compute_Thetas(algo::Algorithm, k::Int)

Compute the capital Theta matrices for the iteration-dependent Lyapunov context.

The matrices are defined as follows:

$$\Theta_{0} =
\begin{bmatrix}
I_{n+\bar{m}} & 0_{(n+\bar{m})\times\bar{m}} & 0_{(n+\bar{m})\times m} \\
0_{m\times(n+\bar{m})} & 0_{m\times\bar{m}} & I_{m}
\end{bmatrix}$$

and

$$\Theta_{1}^{(k)} =
\begin{bmatrix}
X_{k+1}^{k,k+1} \\
0_{(\bar{m}+m)\times(n+\bar{m})} \quad I_{(\bar{m}+m)}
\end{bmatrix}$$

where:

- $n = \text{algo.n}$
- $\bar{m} = \text{algo.m\_bar}$
- $m = \text{algo.m}$
- $X_{k+1}^{k,k+1}$ is retrieved via `get_Xs(k, k+1)` using key $(k+1)$.

Input to the function:
==================      
algo: An instance of Algorithm, Algorithm
k: A non-negative integer iteration index used to select the appropriate X matrix, Int

Output to the function
=================== 
A tuple $(\Theta_0, \Theta_1)$ of matrices.

=#


function _compute_Thetas(algo::Algorithm, k::Int)
    n, m_bar, m = algo.n, algo.m_bar, algo.m

    # Construct Theta_0
    Theta0 = [I(n + m_bar) zeros(n + m_bar, m_bar + m);
        zeros(m, n + 2 * m_bar) I(m)]

    # Retrieve X matrix
    Xs = get_Xs(algo, k, k + 1)
    X_block = Xs[k+1]

    # Construct Theta_1
    lower_block = [zeros(m_bar + m, n + m_bar) I(m_bar + m)]
    Theta1 = vcat(X_block, lower_block)

    return Theta0, Theta1
end

#=  
    _compute_thetas(algo::Algorithm)

Compute the lowercase theta matrices for the iteration-dependent Lyapunov context.

The matrices are defined as follows:

$$\theta_{0} =
\begin{bmatrix}
I_{\bar{m}_{\text{func}}} & 0_{\bar{m}_{\text{func}} \times \bar{m}_{\text{func}}} & 0_{\bar{m}_{\text{func}} \times m_{\text{func}}} \\
0_{m_{\text{func}} \times \bar{m}_{\text{func}}} & 0_{m_{\text{func}} \times \bar{m}_{\text{func}}} & I_{m_{\text{func}}}
\end{bmatrix}$$

and

$$\theta_{1} =
\begin{bmatrix}
0_{(\bar{m}_{\text{func}}+m_{\text{func}}) \times \bar{m}_{\text{func}}} & I_{(\bar{m}_{\text{func}}+m_{\text{func}})}
\end{bmatrix}$$

where:

- $\bar{m}_{\text{func}}$ is given by `algo.m_bar_func`,
- $m_{\text{func}}$ is given by `algo.m_func`.

**Note:** The theta matrices are only defined when there is at least one functional component.

Input to the function:
==================      
algo: An instance of Algorithm, Algorithm

Output to the function
=================== 
A tuple $(\theta_{0}, \theta_{1})$ of matrices.

=#

function _compute_thetas(algo::Algorithm)
    m_bar_func, m_func = algo.m_bar_func, algo.m_func
    m_func > 0 || throw(ArgumentError("theta matrices require at least one functional component."))

    theta0 = [I(m_bar_func) zeros(m_bar_func, m_bar_func + m_func);
        zeros(m_func, 2 * m_bar_func) I(m_func)]

    theta1 = [zeros(m_bar_func + m_func, m_bar_func) I(m_bar_func + m_func)]

    return theta0, theta1
end

#=  
    verify_iteration_dependent_Lyapunov(prob::InclusionProblem, algo::Algorithm, K::Int, Q_0::Matrix, Q_K::Matrix, q_0::Union{Vector, Nothing}=nothing, q_K::Union{Vector, Nothing}=nothing; solver=:clarabel, show_output=:on)

Verify a chain of iteration-dependent Lyapunov inequalities via an SDP.

This method verifies a chain of iteration-dependent Lyapunov inequalities for a given inclusion problem and algorithm by setting up and solving an appropriate semidefinite program (SDP) using JuMP.

Input to the function:
==================      
prob: An InclusionProblem instance containing interpolation conditions, InclusionProblem
algo: An Algorithm instance providing dimensions and methods to compute matrices, Algorithm
K: A positive integer defining the iteration budget, Int
Q_0: A symmetric matrix of dimension $[n + \bar{m} + m] \times [n + \bar{m} + m]$, Matrix
Q_K: A symmetric matrix of dimension $[n + \bar{m} + m] \times [n + \bar{m} + m]$, Matrix
q_0: A vector for functional components (if any), with appropriate dimensions, Union{Vector, Nothing} (optional, default=nothing)
q_K: A vector for functional components (if any), with appropriate dimensions, Union{Vector, Nothing} (optional, default=nothing)
solver: Solver to use (:mosek,  :clarabel, :copt, :scs, :cosmo), Symbol (keyword, default=:clarabel)
show_output: Whether to show solver output (:on or :off), Symbol (keyword, default=:on)

Output to the function
=================== 
A tuple $(true, c)$ if the SDP is solved successfully, or $(false, nothing)$ otherwise.

=#

function verify_iteration_dependent_Lyapunov(
    prob::InclusionProblem,
    algo::Algorithm,
    K::Int,
    Q_0::Matrix, Q_K::Matrix,
    q_0::Union{Vector,Nothing}=nothing, q_K::Union{Vector,Nothing}=nothing;
    solver=:mosek, # options are :mosek, :clarabel, :cosmo, :scs, :copt may add more later
    show_output=true # options are true, false
)
    # --- 1. Validation ---
    (prob.m == algo.m && Set(prob.I_func) == Set(algo.I_func) && Set(prob.I_op) == Set(algo.I_op)) || throw(ArgumentError("Problem and Algorithm definitions are inconsistent."))
    K > 0 || throw(ArgumentError("K must be positive."))

    # --- 2. Dimensions ---
    n, m, m_bar, m_func, m_bar_func = algo.n, algo.m, algo.m_bar, algo.m_func, algo.m_bar_func
    dim_Q = n + m_bar + m
    size(Q_0) == (dim_Q, dim_Q) || throw(DimensionMismatch("Q_0 has incorrect dimensions."))
    size(Q_K) == (dim_Q, dim_Q) || throw(DimensionMismatch("Q_K has incorrect dimensions."))

    isapprox(Q_0, Q_0', atol=1e-10, rtol=1e-10) || throw(ArgumentError("Q_0 must be symmetric."))

    isapprox(Q_K, Q_K', atol=1e-10, rtol=1e-10) || throw(ArgumentError("Q_K must be symmetric."))


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
    @variable(model, c >= 0)

    Qs = Dict{Int,Any}(0 => Q_0, K => Q_K)
    for k in 1:K-1
        Qs[k] = @variable(model, [1:dim_Q, 1:dim_Q], Symmetric, base_name = "Q_$k")
    end

    # qs = Dict{Int, Any}()
    # if m_func > 0
    #     dim_q = m_bar_func + m_func
    #     isnothing(q_0) || size(q_0) == (dim_q,) || throw(DimensionMismatch("q_0 has incorrect dimensions."))
    #     isnothing(q_K) || size(q_K) == (dim_q,) || throw(DimensionMismatch("q_K has incorrect dimensions."))
    #     qs[0], qs[K] = q_0, q_K
    #     for k in 1:K-1
    #         qs[k] = @variable(model, [1:dim_q], base_name="q_$k")
    #     end
    # end

    qs = Dict{Int,Any}()
    if m_func > 0
        dim_q = m_bar_func + m_func
        (q_0 !== nothing && size(q_0) == (dim_q,)) || throw(DimensionMismatch("q_0 has incorrect dimensions or is missing."))
        (q_K !== nothing && size(q_K) == (dim_q,)) || throw(DimensionMismatch("q_K has incorrect dimensions or is missing."))
        qs[0], qs[K] = q_0, q_K
        for k in 1:K-1
            qs[k] = @variable(model, [1:dim_q], base_name = "q_$k")
        end
    end

    # --- 5. Build Main Constraints ---
    Ws, ws = Dict(), Dict()
    for k in 0:K-1
        Theta0, Theta1 = _compute_Thetas(algo, k)
        W_expr = Theta1' * Qs[k+1] * Theta1 - (k == 0 ? c : 1.0) * Theta0' * Qs[k] * Theta0
        Ws[k] = W_expr

        if m_func > 0
            theta0, theta1 = _compute_thetas(algo)
            w_expr = theta1' * qs[k+1] - (k == 0 ? c : 1.0) * theta0' * qs[k]
            ws[k] = w_expr
        end
    end

    # --- 6. Add Interpolation Constraints ---
    for k in 0:K-1
        PSD_sum = -Ws[k]
        EQ_sum = m_func > 0 ? -ws[k] : nothing

        for i in 1:m
            interp_data_list = get_component_data(prob, i)
            for interp_data in interp_data_list
                all_pairs = Vector{Any}()
                for j in 1:algo.m_bar_is[i], k_ in k:k+1
                    push!(all_pairs, (j, k_))
                end
                push!(all_pairs, ("star", "star"))

                idx, M, a, eq = if i in algo.I_op
                    interp_data[2], interp_data[1], nothing, false
                else
                    interp_data[4], interp_data[1], interp_data[2], interp_data[3]
                end

                # pair_combos = if idx == InterpolationIndex.idx_i_lt_j
                #     combinations(all_pairs, 2)
                # elseif idx == InterpolationIndex.idx_i_ne_j
                #     filter(p -> p[1] != p[2], collect(Base.product(all_pairs, all_pairs)))
                # elseif idx == InterpolationIndex.idx_i_only
                #     [[p] for p in all_pairs]
                # elseif idx == InterpolationIndex.idx_i_ne_star
                #     j_k_pairs = filter(p -> p != ("star", "star"), all_pairs)
                #     [[p, ("star", "star")] for p in j_k_pairs]
                # end

                pair_combos =
                    if idx == idx_i_lt_j
                        combinations(all_pairs, 2)
                    elseif idx == idx_i_ne_j
                        filter(p -> p[1] != p[2], collect(Base.product(all_pairs, all_pairs)))
                    elseif idx == idx_i_only
                        [[p] for p in all_pairs]
                    elseif idx == idx_i_ne_star
                        j_k_pairs = filter(p -> p != ("star", "star"), all_pairs)
                        [[p, ("star", "star")] for p in j_k_pairs]
                    else
                        throw(ArgumentError("Invalid interpolation index: $idx"))
                    end

                for pairs in pair_combos
                    pairs_vec = [p for p in pairs]
                    W_mat = compute_W(algo, i, pairs_vec, k, k + 1, M)
                    mult = eq ? @variable(model) : @variable(model, lower_bound = 0)

                    PSD_sum += mult * W_mat
                    if i in algo.I_func
                        F_vec = compute_F_aggregated(algo, i, pairs_vec, k, k + 1, a)
                        EQ_sum += mult * F_vec
                    end
                end
            end
        end
        # Add final LMI and equality constraints for this iteration `k`
        @constraint(model, PSD_sum in PSDCone())
        if m_func > 0
            @constraint(model, EQ_sum .== 0)
        end
    end

    # --- 7. Finalize and Solve ---
    @objective(model, Min, c)

    try
        optimize!(model)
        status = termination_status(model)
        if show_output == true
           @info "[🎯 ] termination_status is $(status)"
        end
        if status in (OPTIMAL, INFEASIBLE_OR_UNBOUNDED)
            return (true, value(c))
        else
            return (false, nothing)
        end
    catch e
        if e isa Mosek.LicenseError
            rethrow(e)
        end
        return (false, nothing)
    end
end


# ===========================================================================
# PERFORMANCE METRIC GENERATORS
# ===========================================================================

#=  
    get_parameters_distance_to_solution(algo::Algorithm, k::Int; i=1, j=1)

Compute the matrices for the distance to solution at iteration k.

This method computes the following matrix:

$$Q_{k} = \left( P_{(i,j)}\, Y_{k}^{k,k} - P_{(i,\star)}\, Y_{\star}^{k,k} \right)^T
        \left( P_{(i,j)}\, Y_{k}^{k,k} - P_{(i,\star)}\, Y_{\star}^{k,k} \right),$$

and

$$q_{k} = 0 \quad \text{(if functional components exist)}.$$

**Definitions**

- $Y_{k}^{k,k}$ is the Y matrix at iteration $k$ over the horizon $[k, k]$.
- $Y_{\star}^{k,k}$ is the "star" Y matrix over $[k, k]$.
- $P_{(i,j)}$ and $P_{(i,\star)}$ are the projection matrices for component $i$.

**Dimensions**

- $\dim(Q_{k}) = n + \bar{m} + m$.
- If $\text{algo.m\_func} > 0$, then $\mathrm{len}(q_{k}) = \bar{m}_{\text{func}} + m_{\text{func}}$.

Input to the function:
==================      
algo: An instance of Algorithm. It must provide: algo.m (total number of components), algo.m_bar_is (a list where the i-th entry gives the number of evaluations for component i), Methods get_Ys(k_min, k_max) and get_Ps(), Algorithm
k: Iteration index. Must satisfy k ≥ 0, Int
i: Component index (1-indexed). Must satisfy 1 ≤ i ≤ algo.m, Int (keyword, default=1)
j: Evaluation index for component i. Must satisfy 1 ≤ j ≤ algo.m_bar_is[i], Int (keyword, default=1)

Output to the function
=================== 
If algo.m_func == 0: $Q_{k}$
Otherwise: A tuple $(Q_{k}, q_{k})$

=#

function get_parameters_distance_to_solution(algo::Algorithm, k::Int; i=1, j=1)
    k >= 0 || throw(ArgumentError("k must be non-negative."))
    Ys, Ps = get_Ys(algo, k, k), get_Ps(algo)
    diff = Ps[(i, j)] * Ys[k] - Ps[(i, "star")] * Ys["star"]
    Q_k = diff' * diff

    if algo.m_func > 0
        dim_q = algo.m_bar_func + algo.m_func
        return Q_k, zeros(dim_q)
    else
        return Q_k
    end
end

#=  
    get_parameters_function_value_suboptimality(algo::Algorithm, k::Int; j=1)

Compute the matrices for function-value suboptimality at iteration k.

This method computes the following matrices:

$$Q_{k} = 0, \quad
q_{k} = \left( F_{(1,j,k)}^{k,k} - F_{(1,\star,\star)}^{k,k} \right)^T.$$

**Definitions**

- $F_{(1,j,k)}^{k,k}$ is the F matrix for functional component 1 corresponding to evaluation $j$ at iteration $k$ over the horizon $[k, k]$.
- $F_{(1,\star,\star)}^{k,k}$ is the star F matrix for functional component 1 over the horizon $[k, k]$.

**Dimensions**

- $\dim(Q_{k}) = n + \bar{m} + m,$
- $\mathrm{len}(q_{k}) = \bar{m}_{\text{func}} + m_{\text{func}}.$

**Note:** This function is only defined for problems with a single functional component, i.e., $m = m_{\text{func}} = 1$.

Input to the function:
==================      
algo: An instance of Algorithm. It must provide: algo.m (total number of components), algo.m_func (number of functional components), algo.m_bar_is (a list where the first entry gives the number of evaluations for component 1), the method get_Fs(k_min, k_max) to obtain the F matrices, Algorithm
k: Iteration index. Must satisfy k ≥ 0, Int
j: Evaluation index for component 1. Must satisfy 1 ≤ j ≤ algo.m_bar_is[1], Int (keyword, default=1)

Output to the function
=================== 
A tuple $(Q_{k}, q_{k})$ where $Q_{k}$ is a zero matrix of dimensions $(n + \bar{m} + m) \times (n + \bar{m} + m)$, and $q_{k}$ is a column vector given by $\left( F_{(1,j,k)}^{k,k} - F_{(1,\star,\star)}^{k,k} \right)^T$.

=#


function get_parameters_function_value_suboptimality(algo::Algorithm, k::Int; j=1)
    (algo.m == 1 && algo.m_func == 1) || throw(ArgumentError("Only for m=m_func=1"))
    k >= 0 || throw(ArgumentError("k must be non-negative."))

    dim_Q = algo.n + algo.m_bar + algo.m
    Q_k = zeros(dim_Q, dim_Q)

    Fs = get_Fs(algo, k, k)

    q_k = Vector(vec((Fs[(1, j, k)] - Fs[(1, "star", "star")])'))

    return Q_k, q_k
end

#=  
    get_parameters_fixed_point_residual(algo::Algorithm, k::Int)

Compute the matrices for the fixed-point residual at iteration k.

This method computes the following matrix:

$$Q_k = ( X_{k+1}^{k,k} - X_k^{k,k} )^T ( X_{k+1}^{k,k} - X_k^{k,k} )$$

and

$$q_{k} = 0 \quad \text{(if functional components exist)}.$$

**Definitions**

- $X_{k+1}^{k,k}$ is the X matrix at iteration $k+1$ over the horizon $[k, k+1]$.
- $X_{k}^{k,k}$ is the X matrix at iteration $k$ over the horizon $[k, k+1]$.

**Dimensions**

- $\dim(Q_{k}) = n + \bar{m} + m$.
- If $\text{algo.m\_func} > 0$, then $\mathrm{len}(q_{k}) = \bar{m}_{\text{func}} + m_{\text{func}}$.

Input to the function:
==================      
algo: An instance of Algorithm. It must provide: algo.m (total number of components), the method get_Xs(k_min, k_max) to obtain the X matrices, Algorithm
k: Iteration index. Must satisfy k ≥ 0, Int

Output to the function
=================== 
If algo.m_func == 0: $Q_{k}$
Otherwise: A tuple $(Q_{k}, q_{k})$

=#

function get_parameters_fixed_point_residual(algo::Algorithm, k::Int)
    k >= 0 || throw(ArgumentError("k must be non-negative."))
    Xs = get_Xs(algo, k, k)
    diff = Xs[k+1] - Xs[k]
    Q_k = diff' * diff

    if algo.m_func > 0
        dim_q = algo.m_bar_func + algo.m_func
        return Q_k, zeros(dim_q)
    else
        return Q_k
    end
end

#=  
    get_parameters_optimality_measure(algo::Algorithm, k::Int)

Compute the matrices for the optimality measure at iteration k.

This method computes the following matrix:

$$Q_{k} = \begin{cases}
\left( P_{(1,1)}\, U_{k}^{k,k} \right)^T \, P_{(1,1)}\, U_{k}^{k,k}
& \text{if } m = 1, \\[1em]
\left( \sum_{i=1}^{m} P_{(i,1)}\, U_{k}^{k,k} \right)^T 
\left( \sum_{i=1}^{m} P_{(i,1)}\, U_{k}^{k,k} \right)
+ \sum_{i=2}^{m} \left( \left( P_{(1,1)} - P_{(i,1)} \right) Y_{k}^{k,k} \right)^T 
\left( \left( P_{(1,1)} - P_{(i,1)} \right) Y_{k}^{k,k} \right)
& \text{if } m > 1.
\end{cases}$$

and

$$q_{k} = 0 \quad \text{(if functional components exist)}.$$

**Definitions**

- $U_{k}^{k,k}$ is the U matrix at iteration $k$ over the horizon $[k,k]$.
- $Y_{k}^{k,k}$ is the Y matrix at iteration $k$ over the horizon $[k,k]$.
- $P_{(i,1)}$ are the projection matrices for component $i$.

**Dimensions**

- $\dim(Q_{k}) = n + \bar{m} + m$.
- If $\text{algo.m\_func} > 0$, then $\mathrm{len}(q_{k}) = \bar{m}_{\text{func}} + m_{\text{func}}$.

Input to the function:
==================      
algo: An instance of Algorithm. It must provide: algo.m (total number of components), algo.m_bar (total number of evaluations per iteration), algo.m_bar_is (a list where the i-th entry gives the number of evaluations for component i), the methods get_Us(k_min, k_max) and get_Ys(k_min, k_max) to obtain the U and Y matrices, the method get_Ps() to obtain the projection matrices, Algorithm
k: Iteration index. Must satisfy k ≥ 0, Int

Output to the function
=================== 
If algo.m_func == 0: $Q_{k}$
Otherwise: A tuple $(Q_{k}, q_{k})$ where $Q_{k}$ is a matrix of dimensions $(n + \bar{m} + m) \times (n + \bar{m} + m)$, and $q_{k}$ is a zero vector of length $(\bar{m}_{\text{func}} + m_{\text{func}})$.

=#

function get_parameters_optimality_measure(algo::Algorithm, k::Int)
    k >= 0 || throw(ArgumentError("k must be non-negative."))
    Us, Ys, Ps = get_Us(algo, k, k), get_Ys(algo, k, k), get_Ps(algo)

    if algo.m == 1
        term = Ps[(1, 1)] * Us[k]
        Q_k = term' * term
    else
        sum_U = sum(Ps[(i, 1)] * Us[k] for i in 1:algo.m)
        term1 = sum_U' * sum_U
        sum_Y = sum(((Ps[(1, 1)] - Ps[(i, 1)]) * Ys[k])' * ((Ps[(1, 1)] - Ps[(i, 1)]) * Ys[k]) for i in 2:algo.m)
        Q_k = term1 + sum_Y
    end

    if algo.m_func > 0
        dim_q = algo.m_bar_func + algo.m_func
        return Q_k, zeros(dim_q)
    else
        return Q_k
    end
end


end # module IterationDependent