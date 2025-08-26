# AutoLyap.jl/src/algorithm.jl

module Algorithms

using LinearAlgebra

# Note: We are not including ProblemClass here, as the base algorithm
# doesn't depend on it, only on its parameters (m, I_func, etc.)

#=
    Solves the problem:

$\text{find } y \in H \text{ such that } 0 \in \sum_{i \in I_{\text{func}}} \partial f_i(y) + \sum_{i \in I_{\text{op}}} G_i(y)$

    with the state-space representation:

    $\mathbf{x}^{k+1} = (A_k \otimes I)\mathbf{x}^k + (B_k \otimes I)\mathbf{u}^k $
       $\mathbf {y}^k = (C_k \otimes I)\mathbf{x}^k + (D_k \otimes I)\mathbf{u}^k$

    where:

    -  $n$ is the dimension of $\mathbf{x}$
    - $m$ is the total number of components (indices $\{1,\dots, m\}$, split into
     $I_{\text{func}}$ and $I_{\text{op}}$.
    - For each $i$,  $\bar{m}_i$ is the number of evaluation and $\bar{m} = \sum_{i=1}^{m} \bar{m}_i$. 
=#

export Algorithm, get_ABCD, get_AsBsCsDs, get_Us, get_Ys, get_Xs, get_Ps, get_Fs,
       compute_E, compute_W, compute_F_aggregated

#=  
    abstract type Algorithm end

Abstract supertype for all algorithm representations.

A concrete subtype `MyAlg <: Algorithm` must have the following fields
to be compatible with the functions defined in this module:
- `n::Int`$\equiv n$: Dimension of x. 
- `m::Int`$\equiv m$: Total components.
- `m_bar_is::Vector{Int}` $\equiv (\bar{m}_i)_{i = 1}^m$: List of evaluation per component, i.e., m_bar_is[i]$\equiv \bar{m}_i$ corresponds to how many times the function $f_i$ or operator $G_i$ is evaluated per iteration.
- `m_bar::Int`$\equiv \bar{m} = \sum_{i=1}^m \bar{m}_i$: Total evaluations.
- `I_func::Vector{Int}`: Functional indices.
- `I_op::Vector{Int}`: Operator indices.
- `m_func::Int`: Count of functional components.
- `m_op::Int`: Count of operator components.
- `m_bar_func::Int`: Total evaluations for functional components.
- `m_bar_op::Int`: Total evaluations for operator components.
- `kappa::Dict{Int, Int}`: Mapping for functional indices.

It must also provide a method for `get_ABCD(alg::MyAlg, k::Int)`.
=#


abstract type Algorithm end

#=  

    _validate_algorithm_params(n, m, m_bar_is, I_func, I_op)

A helper function to be called by constructors of concrete algorithm types
to perform validation of shared parameters.

=#

function _validate_algorithm_params(n, m, m_bar_is, I_func, I_op)
    if n < 1
        throw(ArgumentError("n must be at least 1"))
    end
    if m < 1
        throw(ArgumentError("m must be at least 1"))
    end
    if m != length(m_bar_is)
        throw(ArgumentError("m must equal the length of m_bar_is"))
    end
    if !isdisjoint(I_func, I_op)
        throw(ArgumentError("I_func and I_op must be disjoint"))
    end
    if Set(I_func) ∪ Set(I_op) != Set(1:m)
        throw(ArgumentError("I_func and I_op must cover {1,…, m}"))
    end
    if !(issorted(I_func) && allunique(I_func))
        throw(ArgumentError("I_func must be in increasing order"))
    end
    if !(issorted(I_op) && allunique(I_op))
        throw(ArgumentError("I_op must be in increasing order"))
    end
end

#=  

    get_ABCD(alg::Algorithm, k::Int)

    ** Dimensions **
    - **A**: $(n \times n)$
    - **B**: $(n \times \bar{m})$
    - **C**: $(\bar{m} \times n)$
    - **D**: $(\bar{m} \times \bar{m})$

Return the system matrices (A, B, C, D) for iteration k.
This function must be implemented by each concrete subtype of `Algorithm`.

=#

function get_ABCD(alg::Algorithm, k::Int)
    throw(NotImplementedError("get_ABCD is not implemented for type $(typeof(alg))."))
end

#=  
    get_AsBsCsDs(alg::Algorithm, k_min::Int, k_max::Int)

Return a dictionary mapping each iteration index k to the tuple (A, B, C, D).

=#

function get_AsBsCsDs(alg::Algorithm, k_min::Int, k_max::Int)
    if !(0 <= k_min <= k_max) throw(ArgumentError("Require 0 <= k_min <= k_max")) end
    return Dict(k => get_ABCD(alg, k) for k in k_min:k_max)
end


# --- U, Y, X, P, F MATRIX GENERATION ---

#= 
 Generate a U matrix.

        The total number of columns is given by:

        
           $n + ((k_\text{max} - k_\text{min} + 1) \cdot \bar{m} + m)$.

        - If ``star=True``, returns $U_{\text{star}} $defined as:


            $$
            U_{\text{star}} = \begin{bmatrix}
             \mathbf{0}_{m \times \left(n + ((k_\text{max} - k_\text{min} + 1) \cdot \bar{m})\right)} &
             N &
             \mathbf{0}_{m \times 1}
             \end{bmatrix},
             $$

          where

          .. math::
             $$N = \begin{bmatrix} I_{m-1} \\ -\mathbf{1}_{1 \times (m-1)} \end{bmatrix}.$$

        - If ``star=False`` and k is provided (with k_min ≤ k ≤ k_max), returns:


             $$U_k = \begin{bmatrix}
             \mathbf{0}_{\bar{m} \times \left(n + (k - k_\text{min})\bar{m}\right)} &
             I_{\bar{m}} &
             \mathbf{0}_{\bar{m} \times \left((k_\text{max} - k)\bar{m} + m\right)}
             \end{bmatrix}.$$
        # Input to the function 
        k_min: The minimum iteration index, integer.
        k_max: The maximum iteration index, integer.
        k: (Optional) The current iteration index , required if star is false, integer
        star: If true, generate the star matrix, boolean

        # Output to the function
        # The generated $U$ matrix.
=#

function _generate_U(alg::Algorithm, k_min, k_max; k=nothing, star=false)
    total_cols = alg.n + (k_max - k_min + 1) * alg.m_bar + alg.m
    if star
        if alg.m > 1
            N = vcat(I(alg.m - 1), -ones(1, alg.m - 1))
            return hcat(zeros(alg.m, alg.n + (k_max - k_min + 1) * alg.m_bar), N, zeros(alg.m, 1))
        else
            return zeros(alg.m, total_cols)
        end
    else
        isnothing(k) && throw(ArgumentError("When star is false, k must be provided"))
        left = zeros(alg.m_bar, alg.n + (k - k_min) * alg.m_bar)
        ident = I(alg.m_bar)
        right = zeros(alg.m_bar, (k_max - k) * alg.m_bar + alg.m)
        return hcat(left, ident, right)
    end
end

#= 

Return a dictionary of U matrices for iterations k_min ≤ k ≤ k_max, including the star matrix.

 Input to the function:
 ==================      
k_min: The minimum iteration index, integer
k_max: The maximum iteration index, integer

Output to the function 
=================== 
A dictionary where keys are iteration indices (and 'star') and values are $U$ matrices.


=#

function get_Us(alg::Algorithm, k_min::Int, k_max::Int)
    if !(0 <= k_min <= k_max) throw(ArgumentError("Require 0 <= k_min <= k_max")) end
    Us = Dict{Any, Matrix{Float64}}()
    for k in k_min:k_max
        Us[k] = _generate_U(alg, k_min, k_max; k=k)
    end
    Us["star"] = _generate_U(alg, k_min, k_max; star=true)
    return Us
end

#= 
    _generate_Y(alg::Algorithm, sys_mats, k_min, k_max; k=nothing, star=false)

Generate the output matrix Y using system matrices from sys_mats.

The total number of columns is:

$n + (k_\text{max} - k_\text{min} + 1) \cdot \bar{m} + m.$

- If ``star=True``, returns $Y_{\text{star}}$ defined as:

  $$Y_{\text{star}} = \begin{bmatrix} \mathbf{0}_{m \times (\text{total\_cols} - 1)} & \mathbf{1}_{m \times 1} \end{bmatrix}.$$

- Otherwise, if star is False then k must be provided:
  
  - If $k = k_\text{min}$, then

    $$Y_{k_\text{min}} = \begin{bmatrix} C_{k_\text{min}} & D_{k_\text{min}} & \mathbf{0} \end{bmatrix},$$

    where the zeros block has shape $(\bar{m}, ((k_\text{max} - k_\text{min}) \cdot \bar{m} + m))$.

  - If $k = k_\text{min} + 1$, then

    $$Y_{k_\text{min}+1} = \begin{bmatrix} C_{k_\text{min}+1} A_{k_\text{min}} & C_{k_\text{min}+1} B_{k_\text{min}} & D_{k_\text{min}+1} & \mathbf{0} \end{bmatrix},$$

    with the zeros block of shape $(\bar{m}, ((k_\text{max} - k_\text{min} - 1) \cdot \bar{m} + m))$.

  - If $k \ge k_\text{min} + 2$, then construct Y_k by:
    - Block 1: $C_k (A_{k-1} \cdots A_{k_\text{min}})$,
    - For each $j$ from $k_\text{min}$ to $k-2$, the next block is
     $C_k (A_{k-1} \cdots A_{j+1}) B_j$,
    - Then block: $C_k B_{k-1}$,
    - Followed by $D_k$ and a zeros block of shape $(\bar{m}, ((k_\text{max} - k) \cdot \bar{m} + m))$.

Input to the function:
==================      
sys_mats: Dictionary mapping each iteration index to (A, B, C, D)
k_min: The minimum iteration index, integer
k_max: The maximum iteration index, integer
k: (Optional) The current iteration index (required if star is false), integer
star: If true, generate $Y_{\text{star}}$, boolean

Output to the function
=================== 
The generated $Y$ matrix.

=#

function _generate_Y(alg::Algorithm, sys_mats, k_min, k_max; k=nothing, star=false)
    total_cols = alg.n + (k_max - k_min + 1) * alg.m_bar + alg.m
    if star
        return hcat(zeros(alg.m, total_cols - 1), ones(alg.m, 1))
    end
    isnothing(k) && throw(ArgumentError("When star is false, k must be provided"))

    blocks = []
    if k == k_min
        C_k, D_k = sys_mats[k][3], sys_mats[k][4]
        push!(blocks, C_k)
        push!(blocks, D_k)
    else # k > k_min
        C_k, D_k = sys_mats[k][3], sys_mats[k][4]
        # First block: C_k * A_{k-1} * ... * A_{k_min}
        prod_A = C_k
        for i in k-1:-1:k_min
            prod_A = prod_A * sys_mats[i][1]
        end
        push!(blocks, prod_A)
        # Intermediate blocks: C_k * (A_{k-1}*...*A_{j+1}) * B_j
        for j in k_min:k-2
            prod_B = C_k
            for i in k-1:-1:j+1
                prod_B = prod_B * sys_mats[i][1]
            end
            push!(blocks, prod_B * sys_mats[j][2])
        end
        # Last two blocks before zero padding: C_k*B_{k-1} and D_k
        push!(blocks, C_k * sys_mats[k-1][2])
        push!(blocks, D_k)
    end
    
    Y_mat = hcat(blocks...)
    # Zero padding
    num_cols_padded = size(Y_mat, 2)
    padding = zeros(alg.m_bar, total_cols - num_cols_padded)
    return hcat(Y_mat, padding)
end

#=  

    get_Ys(alg::Algorithm, k_min::Int, k_max::Int)

Return a dictionary of $Y$ matrices for iterations k_min ≤ k ≤ k_max, including $Y_{\text{star}}$.

Input to the function:
==================      
k_min: The minimum iteration index, integer
k_max: The maximum iteration index, integer

Output to the function 
=================== 
A dictionary where keys are iteration indices (and 'star') and values are $Y$ matrices.

=#

function get_Ys(alg::Algorithm, k_min::Int, k_max::Int)
    if !(0 <= k_min <= k_max) throw(ArgumentError("Require 0 <= k_min <= k_max")) end
    sys_mats = get_AsBsCsDs(alg, k_min, k_max)
    Ys = Dict{Any, Matrix{Float64}}()
    for k in k_min:k_max
        Ys[k] = _generate_Y(alg, sys_mats, k_min, k_max; k=k)
    end
    Ys["star"] = _generate_Y(alg, sys_mats, k_min, k_max; star=true)
    return Ys
end

#=  
    _generate_X_k(alg::Algorithm, sys_mats, k, k_min, k_max)

Generate the state matrix X_k for iterations k_min ≤ k ≤ k_max+1.

The total number of columns is:

$n + (k_\text{max} - k_\text{min} + 1) \cdot \bar{m} + m.$

The $X_k$ matrix represents how the state $\mathbf{x}^k$ depends on initial conditions and inputs:

- For $k = k_\text{min}$, $X_{k_\text{min}}$ is given by

  $$X_{k_\text{min}} = \begin{bmatrix} I_n & \mathbf{0} \end{bmatrix},$$

  where the zeros block has the appropriate number of columns.

- For $k = k_\text{min} + 1$, $X_{k_\text{min}+1}$ is given by

  $$X_{k_\text{min}+1} = \begin{bmatrix} A_{k_\text{min}} & B_{k_\text{min}} & \mathbf{0} \end{bmatrix},$$

  with a zeros block of appropriate size.

- For $k \ge k_\text{min} + 2$, $X_k$ is constructed as:

  $$X_{k}=\begin{bmatrix}A_{k-1}\mid\cdots\mid A_{k_{\text{min}}}\mid & \left[(A_{k-1}\cdots A_{j+1})B_{j}\right]_{j=k_{\text{min}},\dots,k-2}\mid & B_{k-1} & \mid\mathbf{0}\end{bmatrix}.$$

Input to the function:
==================      
sys_mats: Dictionary mapping each iteration index to (A, B, C, D)
k: The current iteration index, integer
k_min: The minimum iteration index, integer
k_max: The maximum iteration index, integer

Output to the function
=================== 
The generated X_k matrix representing state evolution.

=#

function _generate_X_k(alg::Algorithm, sys_mats, k, k_min, k_max)
    total_cols = alg.n + (k_max - k_min + 1) * alg.m_bar + alg.m
    if !(k_min <= k <= k_max + 1) throw(ArgumentError("k must be in [k_min, k_max+1]")) end

    blocks = []
    if k == k_min
        push!(blocks, I(alg.n))
    else
        # First block: A_{k-1} * ... * A_{k_min}
        prod_A = I(alg.n)
        for i in k-1:-1:k_min
            prod_A = prod_A * sys_mats[i][1]
        end
        push!(blocks, prod_A)
        # Intermediate blocks
        for j in k_min:k-2
            prod_B = I(alg.n)
            for i in k-1:-1:j+1
                prod_B = prod_B * sys_mats[i][1]
            end
            push!(blocks, prod_B * sys_mats[j][2])
        end
        push!(blocks, sys_mats[k-1][2])
    end

    X_mat = hcat(blocks...)
    num_cols_padded = size(X_mat, 2)
    padding = zeros(alg.n, total_cols - num_cols_padded)
    return hcat(X_mat, padding)
end

#=  
    get_Xs(alg::Algorithm, k_min::Int, k_max::Int)

Return a dictionary mapping each iteration index k (for k_min ≤ k ≤ k_max+1) to the corresponding X_k matrix.

The X_k matrices are constructed using system matrices retrieved via get_AsBsCsDs and the helper method _generate_X_k. Each X_k matrix represents how the state $\mathbf{x}^k$ depends on initial conditions and algorithm inputs over the iteration sequence.

Input to the function:
==================      
k_min: The minimum iteration index, integer
k_max: The maximum iteration index, integer

Output to the function 
=================== 
A dictionary with keys as iteration indices and values as X_k matrices.

=#

function get_Xs(alg::Algorithm, k_min::Int, k_max::Int)
    if !(0 <= k_min <= k_max) throw(ArgumentError("Require 0 <= k_min <= k_max")) end
    sys_mats = get_AsBsCsDs(alg, k_min, k_max)
    Xs = Dict{Int, Matrix{Float64}}()
    for k in k_min:k_max+1
        Xs[k] = _generate_X_k(alg, sys_mats, k, k_min, k_max)
    end
    return Xs
end

#=  
    get_Ps(alg::Algorithm)

Return a dictionary of projection matrices P.

For each component index $i \in \{1,\dots, m\}$ and for each evaluation index
$j \in \{1,\dots, \bar{m}_i\}$, the projection matrix $P_{(i,j)}$ is a $1 \times \bar{m}$ row vector with a 1 at the (offset+j)-th position, where

$$\text{offset} = \sum_{l=1}^{i-1} \bar{m}_l.$$

Additionally, $P_{(i,\star)}$ is a 1 x m row vector with a 1 in the i-th position.

In summary, $P$ is defined as follows: 

$$
\forall_{i\in[1:m]}\forall_{j\in[1:\overline{m}_{i}]}\quad P_{i,j}=\begin{bmatrix}\mathbf{0}_{1\times\sum_{r=1}^{i-1}\overline{m}_{r}}\mid & \left(e_{j}^{\overline{m}_{i}}\right)^{\top}\mid & \mathbf{0}_{1\times\sum_{r=i+1}^{m}\overline{m}_{r}}\end{bmatrix}\in\mathbb{R}^{1\times\overline{m}}
$$
 
$$
\forall_{i\in[1:m]}\quad P_{i,\star}=\left(e_{i}^{m}\right)^{\top}\in\mathbb{R}^{1\times m}
$$

The projection matrices are used to extract specific evaluation points from algorithm trajectories, enabling the selection of particular components and evaluation indices for constraint generation.

Input to the function:
==================      
None (uses algorithm structure)

Output to the function 
=================== 
A dictionary where keys are tuples (i, j) or (i, 'star') and values are projection matrices.

=#

function get_Ps(alg::Algorithm)
    Ps = Dict{Tuple{Int, Any}, Matrix{Float64}}()
    offset = 0
    for i in 1:alg.m
        for j in 1:alg.m_bar_is[i]
            vec = zeros(1, alg.m_bar)
            vec[1, offset + j] = 1
            Ps[(i, j)] = vec
        end
        star_vec = zeros(1, alg.m)
        star_vec[1, i] = 1
        Ps[(i, "star")] = star_vec
        offset += alg.m_bar_is[i]
    end
    return Ps
end

#=  
    _generate_F(alg::Algorithm, i, k_min, k_max; j=nothing, k=nothing, star=false)

Generate a single row of the $F$ matrix for a functional component indexed by i.

The overall F row has dimension:

$$\left( 1, \, ((k_\text{max} - k_\text{min} + 1) \cdot \bar{m}_{\text{func}} + m_{\text{func}}) \right).$$

- For the non-star case (i.e. $F_{(i,j,k)}$), a 1 is placed at the location corresponding to
  the j-th evaluation of component i, shifted by the contributions of all preceding functional components
  and by $(k - k_\text{min})$ blocks of size $\bar{m}_{\text{func}}$.

- For the star case (i.e. $F_{(i,\star,\star)}$), a 1 is placed in the last $m_{\text{func}}$ entries, specifically at index

  $$(k_\text{max} - k_\text{min} + 1) \cdot \bar{m}_{\text{func}} + (\kappa[i]).$$

The $F$ matrices are used to select specific function evaluations from the global function evaluation vector, enabling the construction of linear constraints in semidefinite programming formulations.

Input to the function:
==================      
i: Functional component index (must be in I_func), integer
k_min: The minimum iteration index, integer
k_max: The maximum iteration index, integer
j: (Optional) Evaluation index for the non-star case, integer
k: (Optional) Iteration index for the non-star case (must be in [k_min, k_max]), integer
star: If true, generate the star row, boolean

Output to the function
=================== 
A 1 x total_dim row vector representing the generated F row.

=#

function _generate_F(alg::Algorithm, i, k_min, k_max; j=nothing, k=nothing, star=false)
    total_dim = (k_max - k_min + 1) * alg.m_bar_func + alg.m_func
    i in alg.I_func || throw(ArgumentError("i=$i must be in I_func."))
    
    F_row = zeros(1, total_dim)
    if star
        idx = alg.kappa[i]
        F_row[1, (k_max - k_min + 1) * alg.m_bar_func + idx] = 1
    else
        isnothing(j) || isnothing(k) && throw(ArgumentError("j and k must be provided for non-star F"))
        offset = sum(alg.m_bar_is[l] for l in alg.I_func if l < i; init=0)
        start_idx = alg.m_bar_func * (k - k_min) + offset
        F_row[1, start_idx + j] = 1
    end
    return F_row
end

#=  
    get_Fs(alg::Algorithm, k_min::Int, k_max::Int)

Return a dictionary of F matrices for all functional components.

The dictionary keys are defined as follows:

- For non-star F matrices: keys are of the form (i, j, k), where
  - i is in I_func,
  - j is in [1, m_bar_is[i]],
  - k is in [k_min, k_max].
  
- For star F matrices: keys are of the form (i, 'star', 'star') for each i in I_func.

The F matrices are constructed using the helper method _generate_F and are used to select specific function evaluations from the global function evaluation vector in semidefinite programming formulations.

Input to the function:
==================      
k_min: The minimum iteration index, integer
k_max: The maximum iteration index, integer

Output to the function 
=================== 
A dictionary mapping keys to the corresponding F row matrices.

=#

function get_Fs(alg::Algorithm, k_min::Int, k_max::Int)
    if !(0 <= k_min <= k_max) throw(ArgumentError("Require 0 <= k_min <= k_max")) end
    Fs = Dict{Tuple{Int, Any, Any}, Matrix{Float64}}()
    if alg.m_func > 0
        for k in k_min:k_max
            for i in alg.I_func
                for j in 1:alg.m_bar_is[i]
                    Fs[(i, j, k)] = _generate_F(alg, i, k_min, k_max; j=j, k=k)
                end
            end
        end
        for i in alg.I_func
            Fs[(i, "star", "star")] = _generate_F(alg, i, k_min, k_max; star=true)
        end
    end
    return Fs
end


# --- LIFTED CONSTRAINT MATRICES (E, W, F_aggregated) ---

#=  
    compute_E(alg::Algorithm, i::Int, pairs::Vector{<:Tuple}, k_min::Int, k_max::Int)

Compute the E matrix for component i using a list of (j, k) pairs.

The E matrix is defined as:

$$E^{k_{\text{min}}, k_{\text{max}}}_{(i, j_1, k_1, \dots, j_{n_{i,o}}, k_{n_{i,o}})} =
\begin{bmatrix}
P_{(i,j_1)}Y_{k_1} \\
\vdots \\
P_{(i,j_{n_{i,o}})}Y_{k_{n_{i,o}}} \\
P_{(i,j_1)}U_{k_1} \\
\vdots \\
P_{(i,j_{n_{i,o}})}U_{k_{n_{i,o}}}
\end{bmatrix}.$$

The resulting matrix has dimensions:

$$2 \cdot (\text{number of pairs}) \times \left[ n + (k_{\text{max}} - k_{\text{min}} + 1) \cdot \bar{m} + m \right].$$

Input to the function:
==================      
i: The component index, integer
pairs: A list of (j, k) pairs. For non-star pairs, j must be an integer in [1, m_bar_is[i]] and k must lie in [k_min, k_max]. For the star case, the pair should be ('star', 'star'), Vector{<:Tuple}
k_min: The minimum iteration index, integer
k_max: The maximum iteration index, integer

Output to the function
=================== 
The computed E matrix.

=#

function compute_E(alg::Algorithm, i::Int, pairs::Vector{<:Tuple}, k_min::Int, k_max::Int)
    Ps = get_Ps(alg)
    Ys = get_Ys(alg, k_min, k_max)
    Us = get_Us(alg, k_min, k_max)

    Y_blocks, U_blocks = [], []
    for (j, k) in pairs
        key = (j isa String && j == "star") ? "star" : k
        P = Ps[(i, j)]
        push!(Y_blocks, P * Ys[key])
        push!(U_blocks, P * Us[key])
    end

    return vcat(Y_blocks..., U_blocks...)
end

#=  
    compute_W(alg::Algorithm, i::Int, pairs::Vector{<:Tuple}, k_min::Int, k_max::Int, M::Matrix)

Compute the W matrix for component i.

The W matrix is given by:

$$W = E^T M E,$$

where E is computed via `compute_E`.

Input to the function:
==================      
i: The component index, integer
pairs: A list of (j, k) pairs. For non-star pairs, each pair must be (int, int) with j in the valid range for component i and k in [k_min, k_max]. For star rows, the pair should be ('star', 'star'), Vector{<:Tuple}
k_min: The minimum iteration index, integer
k_max: The maximum iteration index, integer
M: A symmetric matrix of dimension [2*(number of pairs) x 2*(number of pairs)], Matrix

Output to the function
=================== 
The computed W matrix.

=#

function compute_W(alg::Algorithm, i::Int, pairs::Vector{<:Tuple}, k_min::Int, k_max::Int, M::Matrix)
    exp_dim = 2 * length(pairs)
    if size(M) != (exp_dim, exp_dim) throw(ArgumentError("M must be of size ($exp_dim, $exp_dim)")) end
    if !(M ≈ M') throw(ArgumentError("M must be symmetric")) end
    E = compute_E(alg, i, pairs, k_min, k_max)
    return E' * M * E
end

#=  
    compute_F_aggregated(alg::Algorithm, i::Int, pairs::Vector{<:Tuple}, k_min::Int, k_max::Int, a::Vector)

Compute the aggregated F vector for component i.

The aggregated F vector is computed as:

$$F_{\text{aggregated}} = \begin{bmatrix} (F_{(i,j_1,k_1)})^T & \cdots & (F_{(i,j_n,k_n)})^T \end{bmatrix} a,$$

where each F row is obtained from the F matrices (via `get_Fs`), transposed, and horizontally stacked. The resulting matrix has shape (total_dim, number of pairs) and is then multiplied by the weight vector $a$ (a 1D array of length equal to the number of pairs) to yield a column vector of shape (total_dim, 1).

Here,

$$\text{total\_dim} = n + (k_{\text{max}} - k_{\text{min}} + 1) \cdot \bar{m}_{\text{func}} + m_{\text{func}}.$$

Input to the function:
==================      
i: The component index (should be in I_func), integer
pairs: A list of (j, k) pairs. For non-star cases, each pair must be (int, int) with j in [1, m_bar_is[i]] and k in [k_min, k_max]. For the star case, the pair should be ('star', 'star'), Vector{<:Tuple}
k_min: The minimum iteration index, integer
k_max: The maximum iteration index, integer
a: A 1D array of weights with length equal to the number of pairs, Vector

Output to the function
=================== 
The aggregated F vector as a column vector.

=#

function compute_F_aggregated(alg::Algorithm, i::Int, pairs::Vector{<:Tuple}, k_min::Int, k_max::Int, a::Vector)
    i in alg.I_func || throw(ArgumentError("i=$i must be in I_func."))
    if length(a) != length(pairs) throw(ArgumentError("Length of `a` must match length of `pairs`")) end

    Fs_dict = get_Fs(alg, k_min, k_max)
    F_cols = []
    for (j, k) in pairs
        key = (j == "star" && k == "star") ? (i, "star", "star") : (i, j, k)
        push!(F_cols, Fs_dict[key]')
    end
    
    F_stack = hcat(F_cols...)
    return F_stack * a
end

##  Proximal Point Algorithm 

include("algorithms/proximal_point.jl")

## Accelerated Proximal Point

include("algorithms/accelerated_proximal_point.jl")

##  Chambolle-Pock

include("algorithms/chambolle_pock.jl")

## Davis-Yin

include("algorithms/davis_yin.jl")

## Douglas-Rachford

include("algorithms/douglas_rachford.jl")

## Extragradient

include("algorithms/extragradient.jl")

## Forward Method

include("algorithms/forward.jl")

## Gradient Method

include("algorithms/gradient.jl")

## Gradient with Nesterov-like Momentum

include("algorithms/gradient_with_Nesterov_like_momentum.jl")

## Heavy Ball Method

include("algorithms/heavy_ball.jl")

## ITEM

include("algorithms/information_theoretic_exact_method.jl")

## Malitsky-Tam FRB 

include("algorithms/malitsky_tam_frb.jl")

## Nesterov Constant Method

include("algorithms/nesterov_constant.jl")

## Nesterov Fast Gradient Method

include("algorithms/nesterov_fast_gradient_method.jl")

## Optimized Gradient Method

include("algorithms/optimized_gradient_method.jl")

## Triple Momentum Method

include("algorithms/triple_momentum.jl")

## Tseng's FBF Method

include("algorithms/tseng_fbf.jl")

export Utils, ProblemClass, Algorithms, IterationIndependent, IterationDependent

end # module Algorithms