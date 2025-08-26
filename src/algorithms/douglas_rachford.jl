export DouglasRachford, set_gamma!, set_lambda! 

"""
    DouglasRachford(gamma, lambda_value; operator_version=true)

A mutable struct representing the Douglas-Rachford splitting algorithm.
It subtypes the abstract `Algorithm` type.

# Fields
- `gamma::Float64`: The step-size parameter.
- `lambda_value::Float64`: The relaxation parameter (denoted as λ).
"""
mutable struct DouglasRachford <: Algorithm
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

    function DouglasRachford(gamma::Real, lambda_value::Real; operator_version::Bool=true)
        # In the Python code, both branches of the if/else call super() with the same arguments.
        # We reflect this by having one path.
        if operator_version
            n, m, m_bar_is, I_func, I_op = 1, 2, [1, 1], Int[], [1, 2]
        else
            n, m, m_bar_is, I_func, I_op = 1, 2, [1, 1], [1, 2], Int[]
        end
        _validate_algorithm_params(n, m, m_bar_is, I_func, I_op)

        # Calculate derived parameters
        m_bar = sum(m_bar_is)
        m_func = length(I_func)
        m_op = length(I_op)
        m_bar_func = sum(m_bar_is[i] for i in I_func; init=0)
        m_bar_op = sum(m_bar_is[i] for i in I_op; init=0)
        kappa = m_func > 0 ? Dict(I_func[i] => i for i in 1:m_func) : Dict{Int, Int}()

        new(n, m, m_bar_is, m_bar, I_func, I_op, m_func, m_op, m_bar_func, m_bar_op, kappa,
            Float64(gamma), Float64(lambda_value))
    end
end

# Keyword outer constructor
DouglasRachford(; gamma::Real, lambda_value::Real, operator_version::Bool=true) =
    DouglasRachford(gamma, lambda_value; operator_version=operator_version)

# --- Setter functions for algorithm parameters ---
# We can rely on the existing set_gamma! and set_lambda! for DavisYin

function set_gamma!(alg::DouglasRachford, gamma::Real)
    alg.gamma = Float64(gamma)
    return alg
end

function set_lambda!(alg::DouglasRachford, lambda_value::Real)
    alg.lambda_value = Float64(lambda_value)
    return alg
end


"""
    get_ABCD(alg::DouglasRachford, k::Int)

Return the state-space matrices for the Douglas-Rachford algorithm.
The iteration `k` is unused as this is a time-invariant algorithm.
"""
function get_ABCD(alg::DouglasRachford, k::Int)
    γ, λ = alg.gamma, alg.lambda_value

    A = [1.0;;]

    B = [-γ*λ -γ*λ]

    C = [1.0;
         1.0]

    D = [-γ   0.0;
         -2*γ -γ]
    
    return (A, B, C, D)
end