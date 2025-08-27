# `AutoLyap.jl`

`AutoLyap.jl` is a native Julia implementation of the *AutoLyap* methodology and the associated Python package  [AutoLyap](https://github.com/AutoLyap/AutoLyap), which were developed in the papers:

>1. AutoLyap: A Python package for computer-assisted Lyapunov analyses for first-order methods by [Manu Upadhyaya](https://manuupadhyaya.github.io/), [Adrien B. Taylor](https://adrientaylor.github.io/), [Sebastian Banert](https://github.com/sbanert), and [Pontus Giselsson](https://www.control.lth.se/personnel/personnel/pontus-giselsson/), 2025. [[arXiv Link](https://arxiv.org/pdf/2506.24076)]
>2. Manu Upadhyaya, Sebastian Banert, Adrien B. Taylor, and Pontus  Giselsson. "Automated tight Lyapunov analysis for first-order methods." *Mathematical Programming* 209, no. 1 (2025): 133-170. [[arXiv Link](https://arxiv.org/pdf/2302.06713)]
>

The package is functionally equivalent to the Python package [AutoLyap](https://github.com/AutoLyap/AutoLyap), however the certain design patterns are different in the Julia package, e.g., the Julia package uses the notion of `struct+method` along with `multiple dispatch` in Julia over the notion of `class` in Python. 

The `runtestsl.jl` contains Julia test code for all analogous Python test code mentioned in the paper. 

### Installation

In the `Julia REPL` , type

```julia
] add AutoLyap
```

That's it! Now you are ready to use `AutoLyap.jl` by just typing 

```julia
using AutoLyap
```

in the `Julia REPL`. For an example, please proceed to the next section.

### Usage 

Below is a short example on using `AutoLyap.jl` for Douglas-Rachford splitting. 

#### Example: Douglas-Rachford Method

We show here how `AutoLyap.jl` can be used to find linear convergence rates for the Douglas–Rachford method using a few lines of Julia code. In particular, consider the inclusion problem

$$
\text{find } y \in \mathcal{H} \text{ such that } 0 \in G_1(y) + G_2(y),
$$

where $G_1: \mathcal{H} \to \mathcal{H}$ is a maximally monotone operator and $G_2: \mathcal{H} \to \mathcal{H}$ is a $\mu$-strongly monotone and $L$-Lipschitz continuous operator. The Douglas–Rachford method is

$$y_1^k = J_{\gamma G_1}(x^k),$$
$$y_2^k = J_{\gamma G_2}(2y_1^k - x^k),$$
$$x^{k+1} = x^k + \lambda(y_2^k - y_1^k),$$

where $k=0,1,2,\ldots$, $J_{\gamma G_i}$ is the resolvent for $G_i$ with step-size $\gamma \in \mathbb{R}_{++}$, $\lambda \in \mathbb{R}$ is a relaxation parameter, and $x^0 \in \mathcal{H}$ is our initial point. The Douglas–Rachford method is implemented in `AutoLyap.jl` via `DouglasRachford` in `src/algorithms/douglas_rachford.jl`.

The code below, using $(\mu, L, \gamma, \lambda) = (1, 2, 1, 2)$, performs a bisection search to find the smallest $\rho \in [0,1]$ such that $||y_1^k - y^\star||^2 \in O(\rho^k)$ as $k$ goes to $\infty$, where $y^\star \in \text{zero}(G_1 + G_2)$; this is  provable via the Lyapunov analysis in [1], 


```julia
using AutoLyap 

solver_val = :clarabel # options are :mosek (commercial), :clarabel (open-source), :cosmo (open-source), :scs (open-source), :copt (commercial); for the commercial packages you will need valid licenses

show_output_val = false # options are true or false

# -----------------------------------------
# Step 1: Defining the Mathematical Problem
# -----------------------------------------

# We can use the exported types directly
g1_conditions = MaximallyMonotone()
g2_conditions = [
    StronglyMonotone(mu = 1.0),
    LipschitzOperator(L = 2.0)
]
components_list = [g1_conditions, g2_conditions]
problem = InclusionProblem(components = components_list)

# ----------------------------------------------------------------------
# Step 2: Defining the Optimization Algorithm
# ----------------------------------------------------------------------

algorithm = DouglasRachford(gamma = 1.0, lambda_value = 2.0, operator_version=true)

# ----------------------------------------------------------------------
# Step 3: Defining the Performance Metric
# ----------------------------------------------------------------------

(P, T) = IterationIndependent.get_parameters_distance_to_solution(algorithm)

# ----------------------------------------------------------------------
# Step 4: Finding the Best Convergence Rate via an SDP
# ----------------------------------------------------------------------

rho = IterationIndependent.bisection_search_rho(
    problem,
    algorithm,
    P,
    T;
    lower_bound=0.0,
    upper_bound=1.0,
    tol=1e-8,
    solver = solver_val,
    show_output = show_output_val
)

# ----------------------------------------------------------------------
# Step 5: Output
# ----------------------------------------------------------------------

println("[ 🎎 ] Computed DRS convergence rate (rho) for (StronglyMonotone + Lipschitz): $rho") #$
```

For other examples, please see the examples in the `test/runtests.jl` folder.

## Reporting issues
Please report any issues via the [Github issue tracker](https://github.com/AutoLyap/AutoLyap.jl/issues). All types of issues are welcome including bug reports, feature requests, implementation for a specific research problem and so on.

## Contact
Please feel free to contact us regarding any subject including but not limited to comments about this repo, performance estimation problems in general, implementation for a specific research problem, or just to say hi 😃!



