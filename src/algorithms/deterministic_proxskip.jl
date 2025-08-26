export ProxSkip, set_L! # set_gamma! is already exported

"""
    ProxSkip(gamma, L)

A mutable struct representing the ProxSkip algorithm.
The parameter `L` determines the dimensions of the system matrices.

# Fields
- `gamma::Float64`: The step-size parameter.
- `L::Int`: The number of inner-loop gradient steps.
"""
mutable struct ProxSkip <: Algorithm
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
    L::Int

    function ProxSkip(gamma::Real, L::Int)
        alg = new() # Create an uninitialized instance
        set_gamma!(alg, gamma)
        set_L!(alg, L) # This will set L and all derived fields
        return alg
    end
end

# Keyword outer constructor
ProxSkip(; gamma::Real, L::Int) = ProxSkip(gamma, L)

function set_gamma!(alg::ProxSkip, gamma::Real)
    gamma > 0 && isfinite(gamma) || throw(ArgumentError("gamma must be positive and finite."))
    alg.gamma = Float64(gamma)
    return alg
end

function set_L!(alg::ProxSkip, L::Int)
    L >= 1 || throw(ArgumentError("L must be an integer >= 1."))
    alg.L = L

    # --- Re-calculate all dimension-dependent fields ---
    n, m, I_func, I_op = 2, 2, [1, 2], Int[]
    m_bar_is = [L, 1]
    _validate_algorithm_params(n, m, m_bar_is, I_func, I_op)

    alg.n, alg.m, alg.m_bar_is = n, m, m_bar_is
    alg.I_func, alg.I_op = I_func, I_op
    alg.m_bar = sum(m_bar_is)
    alg.m_func = length(I_func)
    alg.m_op = length(I_op)
	alg.m_bar_func = sum(m_bar_is[i] for i in I_func; init=0)
    alg.m_bar_op = sum(m_bar_is[i] for i in I_op; init=0)
    alg.kappa = Dict(I_func[i] => i for i in 1:alg.m_func)
    
    return alg
end

# Private helper for D matrix creation
function _create_D_proxskip(alg::ProxSkip)
    L = alg.L
    γ = alg.gamma
    D = zeros(L + 1, L + 1)
    
    for i_py in 1:(L-1)
        i_jl = i_py + 1
        
        D[i_jl, 1:i_py] .= -γ
        
        D[i_jl, end] = -γ * (i_py - 1 + L)
    end
    
    
    D[end, 1] = -γ
    
    
    D[end, end] = -γ * L
    
    return D
end

"""
    get_ABCD(alg::ProxSkip, k::Int)

Return the state-space matrices for the ProxSkip algorithm.
The iteration `k` is unused as this is a time-invariant algorithm.
"""
function get_ABCD(alg::ProxSkip, k::Int)
    L = alg.L
    γ = alg.gamma

    A = [1.0  γ*(1-L);
         0.0  0.0]

    B_row1 = hcat(-γ * ones(1, L), -γ * (2*L - 1))
    B_row2 = hcat(zeros(1, L), -1.0)
    B = vcat(B_row1, B_row2)

    C_col1 = ones(L, 1)
    C_col2 = γ * (1 - L) * ones(L, 1)
    C_block = hcat(C_col1, C_col2)
    C = vcat([1.0 0.0], C_block)

    D = _create_D_proxskip(alg)
    
    return (A, B, C, D)
end
