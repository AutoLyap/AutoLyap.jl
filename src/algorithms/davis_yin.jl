export DavisYin, set_lambda!, set_gamma! #is already exported

"""
    DavisYin(gamma, lambda_value)

A mutable struct representing the Davis-Yin splitting algorithm.
It subtypes the abstract `Algorithm` type.

# Fields
- `gamma::Float64`: The step-size parameter.
- `lambda_value::Float64`: The relaxation parameter (denoted as λ in the paper).
"""
mutable struct DavisYin <: Algorithm
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
    lambda_value::Float64

    function DavisYin(gamma::Real, lambda_value::Real)
        # --- Set fixed parameters based on the Python super() call ---
        n, m, m_bar_is, I_func, I_op = 1, 3, [1, 1, 1], [1, 2, 3], Int[]
        _validate_algorithm_params(n, m, m_bar_is, I_func, I_op)

        # Calculate derived parameters
        m_bar = sum(m_bar_is)
        m_func = length(I_func)
        m_op = length(I_op)
		m_bar_func = sum(m_bar_is[i] for i in I_func; init=0)
        m_bar_op = sum(m_bar_is[i] for i in I_op; init=0)
        kappa = Dict(I_func[i] => i for i in 1:m_func)

        new(n, m, m_bar_is, m_bar, I_func, I_op, m_func, m_op, m_bar_func, m_bar_op, kappa,
            Float64(gamma), Float64(lambda_value))
    end
end

# Keyword outer constructor
DavisYin(; gamma::Real, lambda_value::Real) = DavisYin(gamma, lambda_value)

# --- Setter functions for algorithm parameters ---
function set_lambda!(alg::DavisYin, lambda_value::Real)
    alg.lambda_value = Float64(lambda_value)
    return alg
end

# Note: `set_gamma!` is already defined generically for ProximalPoint,
# but we can add a specific method for DavisYin if needed for type stability

function set_gamma!(alg::DavisYin, gamma::Real)
    alg.gamma = Float64(gamma)
    return alg
end


"""
    get_ABCD(alg::DavisYin, k::Int)

Return the state-space matrices for the Davis-Yin splitting algorithm.
The iteration `k` is unused as this is a time-invariant algorithm.
"""
function get_ABCD(alg::DavisYin, k::Int)
    γ, λ = alg.gamma, alg.lambda_value

    A = [1.0;;] # 1x1 Matrix

    B = -γ * λ * [1.0 1.0 1.0] # 1x3 Matrix

    C = [1.0;
         1.0;
         1.0] # 3x1 Matrix

    D = -γ * [1.0 0.0 0.0;
              1.0 0.0 0.0;
              2.0 1.0 1.0]

    return (A, B, C, D)
end
