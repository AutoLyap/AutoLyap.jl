export ForwardMethod, set_gamma!

"""
    ForwardMethod(gamma)

A mutable struct representing the Forward (or Gradient) method.
It subtypes the abstract `Algorithm` type.

# Fields
- `gamma::Float64`: The step-size parameter.
"""
mutable struct ForwardMethod <: Algorithm
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

    function ForwardMethod(gamma::Real)
        # --- Set fixed parameters based on the Python super() call ---
        n, m, m_bar_is, I_func, I_op = 1, 1, [1], Int[], [1]
        _validate_algorithm_params(n, m, m_bar_is, I_func, I_op)

        # Calculate derived parameters
        m_bar = sum(m_bar_is)
        m_func = length(I_func)
        m_op = length(I_op)
        m_bar_func = sum(m_bar_is[i] for i in I_func; init=0)
        m_bar_op   = sum(m_bar_is[i] for i in I_op; init=0)
        kappa = m_func > 0 ? Dict(I_func[i] => i for i in 1:m_func) : Dict{Int, Int}()

        new(n, m, m_bar_is, m_bar, I_func, I_op, m_func, m_op, m_bar_func, m_bar_op, kappa, Float64(gamma))
    end
end

# Keyword outer constructor
ForwardMethod(; gamma::Real) = ForwardMethod(gamma)

# We can rely on a generic `set_gamma!` or add a specific one.
function set_gamma!(alg::ForwardMethod, gamma::Real)
    alg.gamma = Float64(gamma)
    return alg
end

"""
    get_ABCD(alg::ForwardMethod, k::Int)

Return the state-space matrices for the Forward method.
The iteration `k` is unused as this is a time-invariant algorithm.
"""
function get_ABCD(alg::ForwardMethod, k::Int)
    γ = alg.gamma

    A = [1.0;;]
    B = [-γ;;]
    C = [1.0;;]
    D = [0.0;;]
    
    return (A, B, C, D)
end


