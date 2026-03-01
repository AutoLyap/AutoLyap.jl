### A Pluto.jl notebook ###
# v0.20.23

using Markdown
using InteractiveUtils

# ╔═╡ 2ec2d60a-1523-11f1-a90f-3723ba1c0c9f
begin
    import Pkg
    # careful: this is _not_ a reproducible environment
    # activate the global environment
    Pkg.activate()
end

# ╔═╡ 6db801d2-a48c-4a15-9ee0-4eb1c1305c79
begin
	using AutoLyap, Clarabel;
	using AutoLyap: IterationIndependent; 
end

# ╔═╡ d4638467-0f47-4aa5-8ff7-7e4146642504
md"""
# Finding linear convergence rates for the Douglas–Rachford method via `AutoLyap.jl`

We show here how `AutoLyap.jl` can be used to find linear convergence rates for the Douglas–Rachford method using a few lines of Julia code. In particular, consider the inclusion problem

$$
\text{find } y \in \mathcal{H} \text{ such that } 0 \in G_1(y) + G_2(y),
$$

where $G_1: \mathcal{H} \to \mathcal{H}$ is a maximally monotone operator and $G_2: \mathcal{H} \to \mathcal{H}$ is a $\mu$-strongly monotone and $L$-Lipschitz continuous operator. The Douglas–Rachford method is

$$y_1^k = J_{\gamma G_1}(x^k),$$
$$y_2^k = J_{\gamma G_2}(2y_1^k - x^k),$$
$$x^{k+1} = x^k + \lambda(y_2^k - y_1^k),$$

where $k=0,1,2,\ldots$, $J_{\gamma G_i}$ is the resolvent for $G_i$ with step-size $\gamma \in \mathbb{R}_{++}$, $\lambda \in \mathbb{R}$ is a relaxation parameter, and $x^0 \in \mathcal{H}$ is our initial point.
"""

# ╔═╡ bebf905b-e1b1-4767-96b5-f821456f35b0
md"""
Let us start with loading our packages. 
"""

# ╔═╡ 67c3b4b7-1f99-48b4-8ac8-0a58c17fe904
md"""
The code below, using $(\mu, L, \gamma, \lambda) = (1, 2, 1, 2)$, performs a bisection search to find the smallest $\rho \in [0,1]$ such that $||y_1^k - y^\star||^2 \in O(\rho^k)$ as $k$ goes to $\infty$, where $y^\star \in \text{zero}(G_1 + G_2)$; this is  provable via the Lyapunov analysis in [1].
"""

# ╔═╡ 9e93a876-db65-403d-8cf8-dbfb904d0139
md"""
Let us start with defining our operators $G_1$ and $G_2$. 
"""

# ╔═╡ 86cf8378-c90c-4194-81c1-dba25317c8f3
g1_conditions = MaximallyMonotone()

# ╔═╡ 63ad8746-1e48-45fb-add4-392874ce51d0
g2_conditions = [
    StronglyMonotone(mu = 1.0),
    LipschitzOperator(L = 2.0)
]

# ╔═╡ 4c0ed8a5-ad56-4227-b9b9-b0a40375516f
components_list = [g1_conditions, g2_conditions]

# ╔═╡ d1c5b609-0c33-4838-99a3-cba1a78d2060
md"""
Let us define the problem type now, which is essentially an monotone inclusion problem. 
"""

# ╔═╡ 3921a1bf-1d0f-4dbb-a1a9-e39c57b4a2d0
problem = InclusionProblem(components = components_list)

# ╔═╡ 480ff309-5b22-4827-a9ab-4cd6f564e24e
md"""
The next step is defining the optimization algorithm in consideration itserlf, which is Douglas-Rachford Splitting. 
"""

# ╔═╡ a62dfd63-5875-453c-9d73-57ccc54aee6a
algorithm = DouglasRachford(gamma = 1.0, lambda_value = 2.0, operator_version=true)

# ╔═╡ 18920310-9a5f-46fb-8c4b-1f5992479d1f
md"""
We now define the performance metric, which correspoinds to distance from the optimal solution.
"""

# ╔═╡ 57d2b602-ddd3-495a-908d-ab6efbc8f33d
(P, T) = IterationIndependent.get_parameters_distance_to_solution(algorithm)

# ╔═╡ d2f0025d-ce27-4595-9306-60fb69a7c3af
md"""
Now we are ready to run a bisection search to find the best convergence rate for Douglas-Rachford Splitting for our setup.
"""

# ╔═╡ 5b423621-2563-415e-a674-48e1a4dc450a
result = IterationIndependent.bisection_search_rho(
    problem,
    algorithm,
    P,
    T;
    lower_bound=0.0,
    upper_bound=1.0,
    tol=1e-8,
    solver = :clarabel,
    show_output = false
)

# ╔═╡ 7702ec1e-9024-47c4-8246-d335275b3773
md"""
Time to get the best convergence rate!
"""

# ╔═╡ 74ad4737-1937-402b-99e5-7679a8e74ad8
rho = result["rho"]

# ╔═╡ 2abff818-c0a4-478a-be1f-d14772c7349d
println("[ 🎎 ] Computed DRS convergence rate (rho) for (StronglyMonotone + Lipschitz): $rho") #$

# ╔═╡ 5f276a99-ce19-40ce-9a3c-90b49989d638
println("Bisection status: ", result["status"])

# ╔═╡ Cell order:
# ╟─d4638467-0f47-4aa5-8ff7-7e4146642504
# ╠═2ec2d60a-1523-11f1-a90f-3723ba1c0c9f
# ╟─bebf905b-e1b1-4767-96b5-f821456f35b0
# ╠═6db801d2-a48c-4a15-9ee0-4eb1c1305c79
# ╟─67c3b4b7-1f99-48b4-8ac8-0a58c17fe904
# ╟─9e93a876-db65-403d-8cf8-dbfb904d0139
# ╠═86cf8378-c90c-4194-81c1-dba25317c8f3
# ╠═63ad8746-1e48-45fb-add4-392874ce51d0
# ╠═4c0ed8a5-ad56-4227-b9b9-b0a40375516f
# ╟─d1c5b609-0c33-4838-99a3-cba1a78d2060
# ╠═3921a1bf-1d0f-4dbb-a1a9-e39c57b4a2d0
# ╟─480ff309-5b22-4827-a9ab-4cd6f564e24e
# ╠═a62dfd63-5875-453c-9d73-57ccc54aee6a
# ╟─18920310-9a5f-46fb-8c4b-1f5992479d1f
# ╠═57d2b602-ddd3-495a-908d-ab6efbc8f33d
# ╟─d2f0025d-ce27-4595-9306-60fb69a7c3af
# ╠═5b423621-2563-415e-a674-48e1a4dc450a
# ╟─7702ec1e-9024-47c4-8246-d335275b3773
# ╠═74ad4737-1937-402b-99e5-7679a8e74ad8
# ╠═2abff818-c0a4-478a-be1f-d14772c7349d
# ╠═5f276a99-ce19-40ce-9a3c-90b49989d638
