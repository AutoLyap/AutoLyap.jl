export OptimizedGradientMethod, set_K!

"""
    OptimizedGradientMethod(L, K)

A mutable struct representing the Optimized Gradient Method.
This is a time-varying algorithm with a finite horizon `K`.

# Fields
- `L::Float64`: The smoothness parameter.
- `K::Int`: The total number of iterations (horizon).
"""
mutable struct OptimizedGradientMethod <: Algorithm
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
    L::Float64
    K::Int

    function OptimizedGradientMethod(L::Real, K::Int)
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
            Float64(L), K)
    end
end

# Keyword outer constructor
OptimizedGradientMethod(; L::Real, K::Int) = OptimizedGradientMethod(L, K)

# --- Setter functions for algorithm parameters ---
function set_L!(alg::OptimizedGradientMethod, L::Real)
    alg.L = Float64(L)
    return alg
end

function set_K!(alg::OptimizedGradientMethod, K::Int)
    alg.K = K
    return alg
end

# --- Private helper function ---
function _compute_theta_ogm(k::Int, K::Int)
    0 <= k <= K || throw(ArgumentError("k must be a non-negative integer <= K."))
    
    θ = 1.0
    for i in 1:k
        if i == K
            θ = (1 + sqrt(1 + 8 * θ^2)) / 2
        else
            θ = (1 + sqrt(1 + 4 * θ^2)) / 2
        end
    end
    return θ
end


"""
    get_ABCD(alg::OptimizedGradientMethod, k::Int)

Return the time-varying state-space matrices for the Optimized Gradient Method.
"""
function get_ABCD(alg::OptimizedGradientMethod, k::Int)
    K = alg.K
    L = alg.L
    
    A, B, C, D = if k < K
        θ_k = _compute_theta_ogm(k, K)
        θ_kp1 = _compute_theta_ogm(k + 1, K)
        
        A_mat = [1 + (θ_k - 1)/θ_kp1   (1 - θ_k)/θ_kp1;
                 1.0                   0.0]
        
        B_mat = [-(1 + (2*θ_k - 1)/θ_kp1)/L;
                 -1/L]
        
        C_mat = [1.0 0.0]
        
        D_mat = [0.0;;]

        (A_mat, B_mat, C_mat, D_mat)

    elseif k == K
        (zeros(2,2), zeros(2,1), [1.0 0.0], [0.0;;])
    
    else
        throw(ArgumentError("k must be less than or equal to K."))
    end

    return (A, B, C, D)
end