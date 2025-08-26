export GradientNesterovMomentum, set_gamma!, set_delta! 

"""
    GradientNesterovMomentum(gamma, delta)

A mutable struct representing the Gradient method with Nesterov-like momentum.
It subtypes the abstract `Algorithm` type.

# Fields
- `gamma::Float64`: The step-size parameter.
- `delta::Float64`: The momentum parameter.
"""
mutable struct GradientNesterovMomentum <: Algorithm
    # --- Fields required by the Algorithm abstract type ---
    n::Int
    m::Int
    m_bar_is::Vector{Int}
    m_bar::Int
    I_func::Vector{Int}
    I_op::Vector{Int}
    m_func::Int
    m_op::Int
    m_bar_func::Int
    m_bar_op::Int
    kappa::Dict{Int, Int}
    # --- Fields specific to this algorithm ---
    gamma::Float64
    delta::Float64

    function GradientNesterovMomentum(gamma::Real, delta::Real)
        # --- Set fixed parameters based on the Python super() call ---
        n, m, m_bar_is, I_func, I_op = 2, 1, [1], [1], Int[]
        _validate_algorithm_params(n, m, m_bar_is, I_func, I_op)

        # Calculate derived parameters
        m_bar = sum(m_bar_is)
        m_func = length(I_func)
        m_op = length(I_op)
        m_bar_func = sum(m_bar_is[i] for i in I_func; init=0)
        m_bar_op   = sum(m_bar_is[i] for i in I_op; init=0)
        kappa = m_func > 0 ? Dict(I_func[i] => i for i in 1:m_func) : Dict{Int, Int}()

        new(n, m, m_bar_is, m_bar, I_func, I_op, m_func, m_op, m_bar_func, m_bar_op, kappa,
            Float64(gamma), Float64(delta))
    end
end

# Keyword outer constructor
GradientNesterovMomentum(; gamma::Real, delta::Real) = GradientNesterovMomentum(gamma, delta)

# --- Setter functions for algorithm parameters ---
function set_gamma!(alg::GradientNesterovMomentum, gamma::Real)
    alg.gamma = Float64(gamma)
    return alg
end

function set_delta!(alg::GradientNesterovMomentum, delta::Real)
    alg.delta = Float64(delta)
    return alg
end


"""
    get_ABCD(alg::GradientNesterovMomentum, k::Int)

Return the state-space matrices for the Gradient with Nesterov Momentum method.
The iteration `k` is unused as this is a time-invariant algorithm.
"""
function get_ABCD(alg::GradientNesterovMomentum, k::Int)
    γ, δ = alg.gamma, alg.delta

    A = [1+δ  -δ;
         1.0  0.0]

    B = [-γ;
         0.0]

    C = [1+δ -δ]

    D = [0.0;;]
    
    return (A, B, C, D)
end
