# AutoLyap.jl/src/problemclass.jl

module ProblemClass

using LinearAlgebra

export InterpolationCondition, OperatorInterpolationCondition, FunctionInterpolationCondition,
    get_data, InclusionProblem, update_component_instances!, get_component_data,
    # Enums
    InterpolationIndex, idx_i_lt_j, idx_i_ne_j, idx_i_only, idx_i_ne_star,
    # Operator Conditions
    MaximallyMonotone, StronglyMonotone, LipschitzOperator, Cocoercive, WeakMintyVariationalInequality,
    # Function Conditions
    Convex, StronglyConvex, WeaklyConvex, Smooth, SmoothConvex, SmoothStronglyConvex,
    SmoothWeaklyConvex, IndicatorFunctionOfClosedConvexSet, SupportFunctionOfClosedConvexSet, GradientDominated

# ---------------------------------------------------------------------------
# InterpolationIndices Enum
# ---------------------------------------------------------------------------
"""
An enum representing the allowed types of interpolation indices.
Replaces the Python `InterpolationIndices` class.
"""
@enum InterpolationIndex begin
    idx_i_lt_j
    idx_i_ne_j
    idx_i_only
    idx_i_ne_star
end

# ---------------------------------------------------------------------------
# Abstract Types for Interpolation Conditions
# ---------------------------------------------------------------------------
"""
Abstract supertype for an interpolation condition.
Derived types must implement the `get_data` method.
"""
abstract type InterpolationCondition end

"""
Abstract type for operator interpolation conditions. `get_data` for subtypes
should return a `Vector` of `Tuple{Matrix, InterpolationIndex}`.
"""
abstract type OperatorInterpolationCondition <: InterpolationCondition end

"""
Abstract type for function interpolation conditions. `get_data` for subtypes
should return a `Vector` of `Tuple{Matrix, Vector, Bool, InterpolationIndex}`.
"""
abstract type FunctionInterpolationCondition <: InterpolationCondition end

# A generic `get_data` function signature to be implemented by concrete types.
function get_data end

# ---------------------------------------------------------------------------
# Concrete Operator Interpolation Condition Structs
# ---------------------------------------------------------------------------

struct MaximallyMonotone <: OperatorInterpolationCondition end
function get_data(::MaximallyMonotone)
    matrix = 0.5 * [0 0 -1 1;
        0 0 1 -1;
        -1 1 0 0;
        1 -1 0 0]
    return [(matrix, idx_i_lt_j)]
end

struct StronglyMonotone <: OperatorInterpolationCondition
    mu::Float64
    function StronglyMonotone(mu::Real)
        mu_f = Float64(mu)
        if mu_f <= 0 || !isfinite(mu_f)
            throw(ArgumentError("Strong monotonicity parameter (mu) must be positive and finite."))
        end
        new(mu_f)
    end
end
function get_data(cond::StronglyMonotone)
    μ = cond.mu
    matrix = 0.5 * [2μ -2μ -1 1;
        -2μ 2μ 1 -1;
        -1 1 0 0;
        1 -1 0 0]
    return [(matrix, idx_i_lt_j)]
end

struct LipschitzOperator <: OperatorInterpolationCondition
    L::Float64
    function LipschitzOperator(L::Real)
        L_f = Float64(L)
        if L_f <= 0 || !isfinite(L_f)
            throw(ArgumentError("Lipschitz parameter (L) must be positive and finite."))
        end
        new(L_f)
    end
end
function get_data(cond::LipschitzOperator)
    L² = cond.L^2
    matrix = [-L² L² 0 0;
        L² -L² 0 0;
        0 0 1 -1;
        0 0 -1 1]
    return [(matrix, idx_i_lt_j)]
end

struct Cocoercive <: OperatorInterpolationCondition
    beta::Float64
    function Cocoercive(beta::Real)
        beta_f = Float64(beta)
        if beta_f <= 0 || !isfinite(beta_f)
            throw(ArgumentError("Cocoercivity parameter (beta) must be positive and finite."))
        end
        new(beta_f)
    end
end
function get_data(cond::Cocoercive)
    β = cond.beta
    matrix = 0.5 * [0 0 -1 1;
        0 0 1 -1;
        -1 1 2β -2β;
        1 -1 -2β 2β]
    return [(matrix, idx_i_lt_j)]
end

struct WeakMintyVariationalInequality <: OperatorInterpolationCondition
    rho_minty::Float64
    function WeakMintyVariationalInequality(rho_minty::Real)
        rho_f = Float64(rho_minty)
        if rho_f < 0 || !isfinite(rho_f)
            throw(ArgumentError("Weak MVI parameter (rho_minty) must be nonnegative and finite."))
        end
        new(rho_f)
    end
end
function get_data(cond::WeakMintyVariationalInequality)
    ρ = cond.rho_minty
    matrix = 0.5 * [0 0 -1 0;
        0 0 1 0;
        -1 1 -ρ 0;
        0 0 0 0]
    return [(matrix, idx_i_ne_star)]
end

# ---------------------------------------------------------------------------
# Concrete Function Interpolation Condition Structs
# ---------------------------------------------------------------------------

# Helper function, equivalent to the Python static method
function _compute_interpolation_data(mu, L)
    vector = [-1.0, 1.0]
    eq = false
    interp_idx = idx_i_ne_j

    if !isfinite(L) # Nonsmooth case
        matrix = 0.5 * [mu -mu 0 1;
            -mu mu 0 -1;
            0 0 0 0;
            1 -1 0 0]
    else # Smooth case
        matrix = (1 / (2 * (L - mu))) * [L*mu -L*mu -mu L;
            -L*mu L*mu mu -L;
            -mu mu 1 -1;
            L -L -1 1]
    end
    return (matrix, vector, eq, interp_idx)
end

abstract type ParametrizedFunctionInterpolationCondition <: FunctionInterpolationCondition end

struct SmoothStronglyConvex <: ParametrizedFunctionInterpolationCondition
    mu::Float64
    L::Float64
    function SmoothStronglyConvex(mu::Real, L::Real)
        mu_f, L_f = Float64(mu), Float64(L)
        if !(0 < mu_f < L_f && isfinite(mu_f) && isfinite(L_f))
            throw(ArgumentError("Requires 0 < mu < L, with mu and L finite."))
        end
        new(mu_f, L_f)
    end
end
get_data(cond::SmoothStronglyConvex) = [_compute_interpolation_data(cond.mu, cond.L)]

struct Convex <: ParametrizedFunctionInterpolationCondition end
get_data(::Convex) = [_compute_interpolation_data(0.0, Inf)]

struct StronglyConvex <: ParametrizedFunctionInterpolationCondition
    mu::Float64
    function StronglyConvex(mu::Real)
        mu_f = Float64(mu)
        if !(mu_f > 0 && isfinite(mu_f))
            throw(ArgumentError("For StronglyConvex, mu must be > 0 and finite."))
        end
        new(mu_f)
    end
end
get_data(cond::StronglyConvex) = [_compute_interpolation_data(cond.mu, Inf)]

struct WeaklyConvex <: ParametrizedFunctionInterpolationCondition
    mu::Float64 # mu will be negative
    function WeaklyConvex(mu_tilde::Real)
        mu_f = -Float64(mu_tilde)
        if !(mu_f < 0)
            throw(ArgumentError("For WeaklyConvex, mu_tilde must be > 0."))
        end
        new(mu_f)
    end
end
get_data(cond::WeaklyConvex) = [_compute_interpolation_data(cond.mu, Inf)]

struct Smooth <: ParametrizedFunctionInterpolationCondition
    L::Float64
    function Smooth(L::Real)
        L_f = Float64(L)
        if !(L_f > 0 && isfinite(L_f))
            throw(ArgumentError("For Smooth, L must be > 0 and finite."))
        end
        new(L_f)
    end
end
function get_data(cond::Smooth)
    # A general L-smooth function has a convexity parameter mu = -L.
    return [_compute_interpolation_data(-cond.L, cond.L)]
end

struct SmoothConvex <: ParametrizedFunctionInterpolationCondition
    L::Float64
    function SmoothConvex(L::Real)
        L_f = Float64(L)
        if !(L_f > 0 && isfinite(L_f))
            throw(ArgumentError("For SmoothConvex, L must be > 0 and finite."))
        end
        new(L_f)
    end
end
get_data(cond::SmoothConvex) = [_compute_interpolation_data(0.0, cond.L)]

struct SmoothWeaklyConvex <: ParametrizedFunctionInterpolationCondition
    mu::Float64
    L::Float64
    function SmoothWeaklyConvex(mu_tilde::Real, L::Real)
        mu_f = -Float64(mu_tilde)
        L_f = Float64(L)
        if !(mu_f < 0 && L_f > 0 && isfinite(L_f))
            throw(ArgumentError("Requires mu_tilde > 0 and L to be > 0 and finite."))
        end
        new(mu_f, L_f)
    end
end
get_data(cond::SmoothWeaklyConvex) = [_compute_interpolation_data(cond.mu, cond.L)]


struct IndicatorFunctionOfClosedConvexSet <: FunctionInterpolationCondition end
function get_data(::IndicatorFunctionOfClosedConvexSet)
    matrix_ineq = 0.5 * [0 0 0 1;
        0 0 0 -1;
        0 0 0 0;
        1 -1 0 0]
    vector_ineq = [0.0, 0.0]

    matrix_eq = [0 0; 0 0]
    vector_eq = [1.0]

    return [(matrix_ineq, vector_ineq, false, idx_i_ne_j),
        (matrix_eq, vector_eq, true, idx_i_only)]
end

struct SupportFunctionOfClosedConvexSet <: FunctionInterpolationCondition end
function get_data(::SupportFunctionOfClosedConvexSet)
    matrix_ineq = 0.5 * [0 0 0 0;
        0 0 1 -1;
        0 1 0 0;
        0 -1 0 0]
    vector_ineq = [0.0, 0.0]

    matrix_eq = 0.5 * [0 1; 1 0]
    vector_eq = [-1.0]

    return [(matrix_ineq, vector_ineq, false, idx_i_ne_j),
        (matrix_eq, vector_eq, true, idx_i_only)]
end

struct GradientDominated <: FunctionInterpolationCondition
    mu_gd::Float64
    function GradientDominated(mu_gd::Real)
        mu_gd_f = Float64(mu_gd)
        if !(mu_gd_f > 0 && isfinite(mu_gd_f))
            throw(ArgumentError("Gradient-dominated parameter (mu_gd) must be positive and finite."))
        end
        new(mu_gd_f)
    end
end
function get_data(cond::GradientDominated)
    M1 = zeros(4, 4)
    a1 = [-1.0, 1.0]

    M2 = zeros(4, 4)
    M2[3, 3] = -1 / (2 * cond.mu_gd)
    a2 = [1.0, -1.0]

    return [(M1, a1, false, idx_i_ne_star),
        (M2, a2, false, idx_i_ne_star)]
end

# ---------------------------------------------------------------------------
# InclusionProblem Struct
# ---------------------------------------------------------------------------

"""
    InclusionProblem(components)

Encapsulates an inclusion problem.
"""
mutable struct InclusionProblem
    m::Int
    components::Dict{Int,Vector{InterpolationCondition}}
    I_op::Vector{Int}
    I_func::Vector{Int}

    function InclusionProblem(components_in::Vector)
        if isempty(components_in)
            throw(ArgumentError("At least one component is required."))
        end
        m = length(components_in)
        components = Dict{Int,Vector{InterpolationCondition}}()

        # Build and validate components dictionary
        for (i, comp) in enumerate(components_in)
            cond_list = comp isa Vector ? comp : [comp]
            _validate_component_uniformity(i, cond_list)
            for cond in cond_list
                _validate_condition_data(cond)
            end
            components[i] = cond_list
        end

        # Post-validation checks
        for conds in values(components)
            if any(c isa GradientDominated || c isa WeakMintyVariationalInequality for c in conds) && m != 1
                throw(ArgumentError("If a component contains GradientDominated or WeakMintyVariationalInequality, m must be 1."))
            end
        end

        I_op = [k for (k, conds) in components if conds[1] isa OperatorInterpolationCondition]
        I_func = [k for (k, conds) in components if conds[1] isa FunctionInterpolationCondition]

        new(m, components, I_op, I_func)
    end
end

function _validate_component_uniformity(index::Int, conditions::Vector)
    if isempty(conditions)
        return
    end
    is_op = conditions[1] isa OperatorInterpolationCondition
    for i in 2:length(conditions)
        if (conditions[i] isa OperatorInterpolationCondition) != is_op
            throw(ArgumentError("Component $index contains a mix of operator and function conditions."))
        end
    end
end

function _validate_condition_data(cond::InterpolationCondition)
    data = get_data(cond)
    if cond isa OperatorInterpolationCondition
        for tup in data
            if !(tup isa Tuple && length(tup) == 2)
                throw(ArgumentError("Operator data must be a tuple of 2 elements."))
            end
            matrix, interp_idx = tup
            if !(matrix isa Matrix)
                throw(ArgumentError("Operator matrix must be a Matrix."))
            end
            if size(matrix, 1) != size(matrix, 2)
                throw(ArgumentError("Operator matrix must be square."))
            end
            if !(matrix ≈ matrix')
                throw(ArgumentError("Operator matrix must be symmetric."))
            end
            if !(interp_idx isa InterpolationIndex)
                throw(ArgumentError("Operator indices must be of type InterpolationIndex."))
            end
        end
    elseif cond isa FunctionInterpolationCondition
        for tup in data
            if !(tup isa Tuple && length(tup) == 4)
                throw(ArgumentError("Function data must be a tuple of 4 elements."))
            end
            matrix, vector, eq, interp_idx = tup
            if !(matrix isa Matrix)
                throw(ArgumentError("Function matrix must be a Matrix."))
            end
            if size(matrix, 1) != size(matrix, 2)
                throw(ArgumentError("Function matrix must be square."))
            end
            if !(matrix ≈ matrix')
                throw(ArgumentError("Function matrix must be symmetric."))
            end
            if !(vector isa Vector)
                throw(ArgumentError("Function vector must be a Vector."))
            end
            if size(matrix, 1) != 2 * length(vector)
                throw(ArgumentError("Function matrix rows must equal 2 * length(vector)."))
            end
            if !(interp_idx isa InterpolationIndex)
                throw(ArgumentError("Function indices must be of type InterpolationIndex."))
            end
        end
    end
end

# Public API for InclusionProblem
get_component(prob::InclusionProblem, i::Int) = prob.components[i]

function get_component_data(prob::InclusionProblem, i::Int)
    if !(1 <= i <= prob.m)
        throw(ArgumentError("Index $i out of bounds for problem with $(prob.m) components."))
    end
    vcat([get_data(c) for c in prob.components[i]]...)
end

function update_component_instances!(prob::InclusionProblem, index::Int, new_instances)
    if !(1 <= index <= prob.m)
        throw(ArgumentError("Index $index out of bounds for problem with $(prob.m) components."))
    end
    cond_list = new_instances isa Vector ? new_instances : [new_instances]
    _validate_component_uniformity(index, cond_list)
    for cond in cond_list
        _validate_condition_data(cond)
    end
    prob.components[index] = cond_list

    # Re-calculate I_op and I_func
    prob.I_op = [k for (k, conds) in prob.components if conds[1] isa OperatorInterpolationCondition]
    prob.I_func = [k for (k, conds) in prob.components if conds[1] isa FunctionInterpolationCondition]

    return prob
end

# ---------------------------------------------------------------------------
# Keyword outer constructors (forward to existing positional constructors)
# ---------------------------------------------------------------------------

# Operator conditions


function StronglyMonotone(; mu::Real)
    return StronglyMonotone(mu)
end

function LipschitzOperator(; L::Real)
    return LipschitzOperator(L)
end

function Cocoercive(; beta::Real)
    return Cocoercive(beta)
end

function WeakMintyVariationalInequality(; rho_minty::Real)
    return WeakMintyVariationalInequality(rho_minty)
end

# Function conditions

function SmoothStronglyConvex(; mu::Real, L::Real)
    return SmoothStronglyConvex(mu, L)
end

function StronglyConvex(; mu::Real)
    return StronglyConvex(mu)
end

function WeaklyConvex(; mu_tilde::Real)
    return WeaklyConvex(mu_tilde)
end

function Smooth(; L::Real)
    return Smooth(L)
end

function SmoothConvex(; L::Real)
    return SmoothConvex(L)
end

function SmoothWeaklyConvex(; mu_tilde::Real, L::Real)
    return SmoothWeaklyConvex(mu_tilde, L)
end


function GradientDominated(; mu_gd::Real)
    return GradientDominated(mu_gd)
end

# Problem container
function InclusionProblem(; components)
    return InclusionProblem(components)
end

end # module ProblemClass