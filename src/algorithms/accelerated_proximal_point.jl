export AcceleratedProximalPoint # set_gamma! is already exported

"""
    AcceleratedProximalPoint(gamma; type=:operator)

A mutable struct representing the Accelerated Proximal Point algorithm.

# Arguments
- `gamma::Real`: The step-size parameter.

# Keyword Arguments
- `type::Symbol`: Can be `:operator` or `:function`. This determines the
  problem structure the algorithm is configured for. Defaults to `:operator`.
"""
mutable struct AcceleratedProximalPoint <: Algorithm
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

    function AcceleratedProximalPoint(gamma::Real; type::Symbol=:operator)
        if type == :operator
            n, m, m_bar_is, I_func, I_op = 3, 1, [1], Int[], [1]
        elseif type == :function
            n, m, m_bar_is, I_func, I_op = 3, 1, [1], [1], Int[]
        else
            throw(ArgumentError("type must be either :operator or :function"))
        end
        
        _validate_algorithm_params(n, m, m_bar_is, I_func, I_op)

        # Calculate derived parameters
        m_bar = sum(m_bar_is)
        m_func = length(I_func)
        m_op = length(I_op)
		m_bar_func = sum(m_bar_is[i] for i in I_func; init=0)
        m_bar_op = sum(m_bar_is[i] for i in I_op; init=0)
        kappa = m_func > 0 ? Dict(I_func[i] => i for i in 1:m_func) : Dict{Int, Int}()

        new(n, m, m_bar_is, m_bar, I_func, I_op, m_func, m_op, m_bar_func, m_bar_op, kappa, Float64(gamma))
    end
end

# --- Keyword outer constructor(s) forwarding to positional constructor ---
function AcceleratedProximalPoint(; gamma::Real, type::Symbol=:operator)
    return AcceleratedProximalPoint(gamma; type=type)
end

"""
    set_gamma!(alg::AcceleratedProximalPoint, gamma::Real)

Update the step-size `gamma` for an `AcceleratedProximalPoint` instance.
"""
function set_gamma!(alg::AcceleratedProximalPoint, gamma::Real)
    alg.gamma = Float64(gamma)
    return alg
end

"""
    get_ABCD(alg::AcceleratedProximalPoint, k::Int)

Return the time-varying state-space matrices for the Accelerated Proximal Point algorithm.
Note: Assumes `k` is a 0-indexed iteration count, as in the original Python formula.
"""
function get_ABCD(alg::AcceleratedProximalPoint, k::Int)
    k >= 0 || throw(ArgumentError("k must be non-negative."))
    
    λ = k / (k + 2)
    γ = alg.gamma

    A = [0.0    1.0        0.0; 
         -2*λ   1+λ        λ;
        0.0    1.0        0.0]

    B = [-γ;
         -γ*(1+λ);
         0.0]

    C = [0.0 1.0 0.0]

    D = [-γ;;] # 1x1 matrix

    # Ensure B is a 3x1 matrix, not a vector
    return (A, reshape(B, 3, 1), C, D)
end