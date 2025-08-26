export NesterovFastGradientMethod, set_gamma!

"""
    NesterovFastGradientMethod(gamma)

A mutable struct representing Nesterov's Fast Gradient Method.
This is a time-varying algorithm for a single functional component.

# Fields
- `gamma::Float64`: The step-size parameter.
"""
mutable struct NesterovFastGradientMethod <: Algorithm
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

    function NesterovFastGradientMethod(gamma::Real)
        # --- Set fixed parameters based on the Python super() call ---
        n, m, m_bar_is, I_func, I_op = 2, 1, [2], [1], Int[]
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
NesterovFastGradientMethod(; gamma::Real) = NesterovFastGradientMethod(gamma)

# We can rely on a generic `set_gamma!` or add a specific one.
function set_gamma!(alg::NesterovFastGradientMethod, gamma::Real)
    alg.gamma = Float64(gamma)
    return alg
end


"""
    get_ABCD(alg::NesterovFastGradientMethod, k::Int)

Return the time-varying state-space matrices for Nesterov's Fast Gradient Method.
"""
function get_ABCD(alg::NesterovFastGradientMethod, k::Int)
    k >= 0 || throw(ArgumentError("k must be non-negative."))
    
    # Calculate lambda_k and lambda_{k+1}
    λ_curr = 1.0
    λ_prev = 1.0 # Initialize lambda_prev
    for _ in 0:k
        λ_prev = λ_curr
        λ_curr = (1 + sqrt(1 + 4 * λ_curr^2)) / 2
    end
    
    α = (λ_prev - 1) / λ_curr

    A = [1+α  -α;
         1.0  0.0]

    B = [-alg.gamma 0.0;
         0.0        0.0]

    C = [1+α  -α;
         1.0  0.0]

    D = [0.0 0.0;
         0.0 0.0]
    
    return (A, B, C, D)
end
