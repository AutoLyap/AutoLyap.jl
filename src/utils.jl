# AutoLyap.jl/src/utils.jl

module Utils

using JuMP
using LinearAlgebra # For the Symmetric type, although not strictly needed for the dense version

export create_symmetric_matrix, create_symmetric_matrix_expression

const _SDPLR_NAN_MESSAGE = "Error(sdplrlib): Got NaN."
const _SDPLR_LINESEARCH_MESSAGE = "should be less then 0."

"""
    create_symmetric_matrix(upper_triangle_values::Vector{<:Number}, n::Int)

Convert a vector of upper triangle values to a dense symmetric matrix.

# Arguments
- `upper_triangle_values`: A vector of length n(n+1)/2 containing the
  upper triangle and diagonal values, scanned row-by-row.
- `n`: The dimension of the square symmetric matrix.

# Returns
- A dense `Matrix{<:Number}` of size n x n.

# Throws
- `ArgumentError`: If the length of `upper_triangle_values` is not n(n+1)/2.
"""
function create_symmetric_matrix(upper_triangle_values::Vector{T}, n::Int) where T<:Number
    if length(upper_triangle_values) != n * (n + 1) ÷ 2
        throw(ArgumentError("The length of upper_triangle_values must be n(n+1)/2."))
    end

    # In Julia, it's common to initialize with the correct type.
    # We infer it from the input vector's element type `T`.
    symmetric_matrix = Matrix{T}(undef, n, n)

    idx = 1
    for i in 1:n
        for j in i:n
            val = upper_triangle_values[idx]
            symmetric_matrix[i, j] = val
            symmetric_matrix[j, i] = val # This makes it symmetric
            idx += 1
        end
    end

    return symmetric_matrix
end

"""
    create_symmetric_matrix_expression(Xij::Vector{VariableRef}, n::Int)

Convert a vector of JuMP variables to a symmetric matrix of expressions.
This is the JuMP equivalent of the Mosek Fusion function.

# Arguments
- `Xij`: A JuMP variable vector containing the upper triangle and diagonal values.
- `n`: The dimension of the symmetric matrix expression.

# Returns
- A symmetric `Matrix{VariableRef}` of size n x n.
"""
function create_symmetric_matrix_expression(Xij::Vector{VariableRef}, n::Int)
    if length(Xij) != n * (n + 1) ÷ 2
        throw(ArgumentError("The length of the variable vector Xij must be n(n+1)/2."))
    end

    # JuMP works beautifully with native Julia arrays.
    # We create a matrix that will hold our variables.
    X = Matrix{VariableRef}(undef, n, n)

    idx = 1
    for i in 1:n
        for j in i:n
            var = Xij[idx]
            X[i, j] = var
            X[j, i] = var
            idx += 1
        end
    end

    # No need for hstack/vstack, JuMP uses the Julia matrix directly!
    return X
end

function _sdplr_output_has_numerical_error(
    stdout_text::AbstractString,
    stderr_text::AbstractString = "",
)
    combined = string(stdout_text, '\n', stderr_text)
    return occursin(_SDPLR_NAN_MESSAGE, combined) ||
           (occursin("Problem!", combined) && occursin(_SDPLR_LINESEARCH_MESSAGE, combined))
end

function _replay_captured_output(stdout_text::AbstractString, stderr_text::AbstractString)
    isempty(stdout_text) || print(stdout, stdout_text)
    isempty(stderr_text) || print(stderr, stderr_text)
    flush(stdout)
    flush(stderr)
    return nothing
end

function _capture_function_output(f::Function)
    stdout_path, stdout_io = mktemp()
    stderr_path, stderr_io = mktemp()
    thrown = nothing

    try
        redirect_stdout(stdout_io) do
            redirect_stderr(stderr_io) do
                try
                    f()
                catch err
                    thrown = err
                finally
                    Base.Libc.flush_cstdio()
                    flush(stdout)
                    flush(stderr)
                end
            end
        end
    finally
        close(stdout_io)
        close(stderr_io)
    end

    stdout_text = read(stdout_path, String)
    stderr_text = read(stderr_path, String)
    rm(stdout_path; force=true)
    rm(stderr_path; force=true)

    return (
        stdout_text=stdout_text,
        stderr_text=stderr_text,
        exception=thrown,
    )
end

function optimize_with_diagnostics!(model::JuMP.Model, solver::Symbol; show_output::Bool=false)
    if solver != :sdplr
        optimize!(model)
        return (
            stdout_text="",
            stderr_text="",
            native_numerical_error=false,
        )
    end

    captured = _capture_function_output() do
        optimize!(model)
    end

    if show_output
        _replay_captured_output(captured.stdout_text, captured.stderr_text)
    end

    if captured.exception !== nothing
        throw(captured.exception)
    end

    return (
        stdout_text=captured.stdout_text,
        stderr_text=captured.stderr_text,
        native_numerical_error=_sdplr_output_has_numerical_error(
            captured.stdout_text,
            captured.stderr_text,
        ),
    )
end

end # module Utils
