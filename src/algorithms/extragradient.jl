export Extragradient, ExtragradientUnconstrained, ExtragradientConstrained, set_delta!

# --- Define an abstract subtype for all Extragradient methods ---
abstract type Extragradient <: Algorithm end

# --- Concrete struct for the Unconstrained version ---
mutable struct ExtragradientUnconstrained <: Extragradient
    # Fields required by Algorithm
    n::Int; m::Int; m_bar_is::Vector{Int}; m_bar::Int; I_func::Vector{Int}; I_op::Vector{Int}
    m_func::Int; m_op::Int; m_bar_func::Int; m_bar_op::Int; kappa::Dict{Int, Int}
    # Algorithm-specific fields
    gamma::Float64
    delta::Float64
end

# --- Concrete struct for the Constrained version ---
mutable struct ExtragradientConstrained <: Extragradient
    # Fields required by Algorithm
    n::Int; m::Int; m_bar_is::Vector{Int}; m_bar::Int; I_func::Vector{Int}; I_op::Vector{Int}
    m_func::Int; m_op::Int; m_bar_func::Int; m_bar_op::Int; kappa::Dict{Int, Int}
    # Algorithm-specific fields
    gamma::Float64
    delta::Float64
end

"""
    Extragradient(gamma, delta; type=:unconstrained)

Factory function to create an instance of the Extragradient algorithm.

# Arguments
- `gamma::Real`: The step-size parameter.
- `delta::Real`: The second step-size or weight parameter.

# Keyword Arguments
- `type::Symbol`: Can be `:unconstrained` or `:constrained`. Defaults to `:unconstrained`.
"""
function Extragradient(gamma::Real, delta::Real; type::Symbol=:unconstrained)
    if type == :unconstrained
        n, m, m_bar_is, I_func, I_op = 1, 1, [2], Int[], [1]
        alg = ExtragradientUnconstrained(n, m, m_bar_is, 0, I_func, I_op, 0, 0, 0, 0, Dict(), 0.0, 0.0)
    elseif type == :constrained
        n, m, m_bar_is, I_func, I_op = 1, 2, [2, 2], [2], [1]
        alg = ExtragradientConstrained(n, m, m_bar_is, 0, I_func, I_op, 0, 0, 0, 0, Dict(), 0.0, 0.0)
    else
        throw(ArgumentError("type must be either :unconstrained or :constrained"))
    end
    
    # Set parameters and derived fields
    alg.gamma = Float64(gamma)
    alg.delta = Float64(delta)
    _validate_algorithm_params(n, m, m_bar_is, I_func, I_op)
    alg.m_bar = sum(m_bar_is)
    alg.m_func = length(I_func)
    alg.m_op = length(I_op)
	alg.m_bar_func = sum(m_bar_is[i] for i in I_func; init=0)
    alg.m_bar_op = sum(m_bar_is[i] for i in I_op; init=0)
    alg.kappa = alg.m_func > 0 ? Dict(I_func[i] => i for i in 1:alg.m_func) : Dict{Int, Int}()

    return alg
end


# --- Setter functions dispatching on the abstract type ---
function set_gamma!(alg::Extragradient, gamma::Real)
    alg.gamma = Float64(gamma)
    return alg
end

function set_delta!(alg::Extragradient, delta::Real)
    alg.delta = Float64(delta)
    return alg
end

# --- get_ABCD methods dispatching on the concrete types ---
function get_ABCD(alg::ExtragradientUnconstrained, k::Int)
    A = [1.0;;]
    B = [0.0 -alg.delta]
    C = [1.0; 1.0]
    D = [0.0 0.0; -alg.gamma 0.0]
    return (A, B, C, D)
end

function get_ABCD(alg::ExtragradientConstrained, k::Int)
    A = [1.0;;]
    B = [0.0 -alg.delta 0.0 -alg.delta]
    C = [1.0; 1.0; 1.0; 1.0]
    D = [0.0  -alg.delta 0.0  -alg.delta;
         -alg.gamma 0.0 -alg.gamma 0.0;
         -alg.gamma 0.0 -alg.gamma 0.0;
         0.0  -alg.delta 0.0  -alg.delta]
    # In the python version, rows 2 and 3 of D are identical, let's fix this for clarity
    D = [0.0 0.0 0.0 0.0;
         -alg.gamma 0.0 -alg.gamma 0.0;
         -alg.gamma 0.0 -alg.gamma 0.0;
         0.0 -alg.delta 0.0 -alg.delta]
    # The python D matrix was:
    # D_py = [[0,0,0,0], [-g,0,-g,0], [-g,0,-g,0], [0,-d,0,-d]]
    # This seems to have row 1 and 4 swapped. Let's assume the python code is correct.
    D_from_python = [0.0 0.0 0.0 0.0;
                     -alg.gamma 0.0 -alg.gamma 0.0;
                     -alg.gamma 0.0 -alg.gamma 0.0;
                     0.0 -alg.delta 0.0 -alg.delta]

    return (A, B, C, D_from_python)
end

