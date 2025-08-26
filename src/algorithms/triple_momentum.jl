export TripleMomentum # set_sigma! and set_beta! are already exported

"""
    TripleMomentum(sigma, beta)

A mutable struct representing the Triple Momentum method.
It subtypes the abstract `Algorithm` type.

# Fields
- `sigma::Float64`: The strong convexity parameter (often denoted μ).
- `beta::Float64`: The smoothness parameter (often denoted L).
"""
mutable struct TripleMomentum <: Algorithm
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
    sigma::Float64
    beta::Float64

    function TripleMomentum(sigma::Real, beta::Real)
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
            Float64(sigma), Float64(beta))
    end
end

# Keyword outer constructor
TripleMomentum(; sigma::Real, beta::Real) = TripleMomentum(sigma, beta)

# --- Setter functions for algorithm parameters ---
function set_sigma!(alg::TripleMomentum, sigma::Real)
    alg.sigma = Float64(sigma)
    return alg
end

function set_beta!(alg::TripleMomentum, beta::Real)
    alg.beta = Float64(beta)
    return alg
end


"""
    get_ABCD(alg::TripleMomentum, k::Int)

Return the state-space matrices for the Triple Momentum method.
The iteration `k` is unused as this is a time-invariant algorithm.
"""
function get_ABCD(alg::TripleMomentum, k::Int)
    σ, β = alg.sigma, alg.beta
    q = σ / β
    sqrt_q = sqrt(q)
    
    α = (2 - sqrt_q) / β
    β_var = (1 - sqrt_q)^2 / (1 + sqrt_q) # Using `β_var` to avoid shadowing the input `β`
    γ_var = (1 - sqrt_q)^2 / ((2 - sqrt_q) * (1 + sqrt_q))
    
    A = [1 + β_var  -β_var;
         1.0        0.0]

    B = [-α;
         0.0]

    C = [1 + γ_var  -γ_var]

    D = [0.0;;]
    
    return (A, B, C, D)
end



