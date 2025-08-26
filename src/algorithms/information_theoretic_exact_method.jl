export ITEM, set_mu!

"""
    ITEM(mu, L)

A mutable struct representing the Information Theoretic Exact Method (ITEM).
This is a time-varying algorithm for a single functional component.

# Fields
- `mu::Float64`: The strong convexity parameter.
- `L::Float64`: The smoothness parameter.
"""
mutable struct ITEM <: Algorithm
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
    mu::Float64
    L::Float64

    function ITEM(mu::Real, L::Real)
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
            Float64(mu), Float64(L))
    end
end

# Keyword outer constructor
ITEM(; mu::Real, L::Real) = ITEM(mu, L)

# --- Setter functions for algorithm parameters ---
function set_L!(alg::ITEM, L::Real)
    alg.L = Float64(L)
    return alg
end

function set_mu!(alg::ITEM, mu::Real)
    alg.mu = Float64(mu)
    return alg
end

# --- Private helper functions for ITEM ---
function _get_A_item(alg::ITEM, k::Int)
    q = alg.mu / alg.L
    A_k = 0.0
    for _ in 1:k
        A_k = ((1 + q) * A_k + 2 * (1 + sqrt((1 + A_k) * (1 + q * A_k)))) / ((1 - q)^2)
    end
    return A_k
end

function _compute_beta_item(alg::ITEM, k::Int)
    q = alg.mu / alg.L
    A_k = _get_A_item(alg, k)
    A_k1 = _get_A_item(alg, k + 1)
    return A_k / ((1 - q) * A_k1)
end

function _compute_delta_item(alg::ITEM, k::Int)
    q = alg.mu / alg.L
    A_k = _get_A_item(alg, k)
    A_k1 = _get_A_item(alg, k + 1)
    numerator = ((1 - q)^2) * A_k1 - (1 + q) * A_k
    denominator = 2 * (1 + q + q * A_k)
    return numerator / denominator
end


"""
    get_ABCD(alg::ITEM, k::Int)

Return the time-varying state-space matrices for the ITEM algorithm.
"""
function get_ABCD(alg::ITEM, k::Int)
    k >= 0 || throw(ArgumentError("k must be non-negative."))
    
    β = _compute_beta_item(alg, k)
    δ = _compute_delta_item(alg, k)
    q = alg.mu / alg.L

    A = [β      1-β;
         q*β*δ  1-q*β*δ]

    B = [-1/alg.L;
         -δ/alg.L]

    C = [β 1-β]

    D = [0.0;;]
    
    return (A, B, C, D)
end

