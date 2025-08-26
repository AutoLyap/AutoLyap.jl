export ChambollePock, set_tau!, set_sigma!, set_theta!

"""
    ChambollePock(tau, sigma, theta)

A mutable struct representing the Chambolle-Pock algorithm.
It subtypes the abstract `Algorithm` type.

# Fields
- `tau::Float64`: The primal step-size.
- `sigma::Float64`: The dual step-size.
- `theta::Float64`: The over-relaxation parameter.
"""
mutable struct ChambollePock <: Algorithm
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
    tau::Float64
    sigma::Float64
    theta::Float64

    function ChambollePock(tau::Real, sigma::Real, theta::Real)
        # --- Set fixed parameters based on the Python super() call ---
        n, m, m_bar_is, I_func, I_op = 2, 2, [1, 1], [1, 2], Int[]
        _validate_algorithm_params(n, m, m_bar_is, I_func, I_op)

        # Calculate derived parameters
        m_bar = sum(m_bar_is)
        m_func = length(I_func)
        m_op = length(I_op)
		m_bar_func = sum(m_bar_is[i] for i in I_func; init=0)
        m_bar_op = sum(m_bar_is[i] for i in I_op; init=0)
        kappa = Dict(I_func[i] => i for i in 1:m_func)

        new(n, m, m_bar_is, m_bar, I_func, I_op, m_func, m_op, m_bar_func, m_bar_op, kappa, 
            Float64(tau), Float64(sigma), Float64(theta))
    end
end

# Keyword outer constructor
ChambollePock(; tau::Real, sigma::Real, theta::Real) = ChambollePock(tau, sigma, theta)

# --- Setter functions for algorithm parameters ---
function set_tau!(alg::ChambollePock, tau::Real)
    alg.tau = Float64(tau)
    return alg
end

function set_sigma!(alg::ChambollePock, sigma::Real)
    alg.sigma = Float64(sigma)
    return alg
end

function set_theta!(alg::ChambollePock, theta::Real)
    alg.theta = Float64(theta)
    return alg
end

"""
    get_ABCD(alg::ChambollePock, k::Int)

Return the state-space matrices for the Chambolle-Pock algorithm.
The iteration `k` is unused as this is a time-invariant algorithm.
"""
function get_ABCD(alg::ChambollePock, k::Int)
    τ, σ, θ = alg.tau, alg.sigma, alg.theta

    A = [1.0  -τ;
         0.0  0.0]

    B = [-τ   0.0;
         0.0  1.0]

    C = [1.0  -τ;
         1.0  1/σ - τ*(1+θ)]

    D = [-τ        0.0;
                     -τ*(1+θ)  -1/σ]


    return (A, B, C, D)
end
