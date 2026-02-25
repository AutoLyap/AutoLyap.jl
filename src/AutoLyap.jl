# AutoLyap.jl/src/AutoLyap.jl

module AutoLyap

# Include the code from our other files
include("utils.jl")
include("problemclass.jl")
include("algorithm.jl")
include("iteration_independent.jl")
include("iteration_dependent.jl")

# Then, bring the functions/types from the modules into the AutoLyap scope
using .Utils
using .ProblemClass
using .Algorithms
import .IterationIndependent
import .IterationDependent
using .IterationIndependent: bisection_search_rho

# Re-export everything for the user
# From utils.jl
export create_symmetric_matrix, create_symmetric_matrix_expression
# From problemclass.jl
export InterpolationCondition, OperatorInterpolationCondition, FunctionInterpolationCondition,
       get_data, InclusionProblem, update_component_instances!,
       InterpolationIndex, idx_i_lt_j, idx_i_ne_j, idx_i_only, idx_i_ne_star,
       MaximallyMonotone, StronglyMonotone, LipschitzOperator, Cocoercive, WeakMintyVariationalInequality,
       Convex, StronglyConvex, WeaklyConvex, Smooth, SmoothConvex, SmoothStronglyConvex,
       SmoothWeaklyConvex, IndicatorFunctionOfClosedConvexSet, SupportFunctionOfClosedConvexSet, GradientDominated
# From algorithm.jl
export Algorithm, get_ABCD, get_AsBsCsDs, get_Us, get_Ys, get_Xs, get_Ps, get_Fs,
       compute_E, compute_W, compute_F_aggregated,
       ProximalPoint, set_gamma!,
       AcceleratedProximalPoint,
       ChambollePock, set_tau!, set_sigma!, set_theta!,
       DavisYin, set_lambda!,
       ProxSkip, set_L!,
       DouglasRachford,
       Extragradient, set_delta!,
       ForwardMethod,
       GradientMethod,
       GradientNesterovMomentum,
       HeavyBallMethod,
       ITEM, set_mu!,
       MalitskyTamFRB,
       NesterovConstant, set_beta!,
       NesterovFastGradientMethod,
       OptimizedGradientMethod, set_K!,
       TripleMomentum,
       TsengFBF
# From iteration_independent.jl
export bisection_search_rho

# get_parameters_distance_to_solution,
# get_parameters_linear_function_value_suboptimality,
# get_parameters_sublinear_function_value_suboptimality,
# get_parameters_fixed_point_residual,
# get_parameters_duality_gap,
# get_parameters_optimality_measure

# From iteration_dependent.jl
# get_parameters_distance_to_solution,
# get_parameters_function_value_suboptimality,
# get_parameters_fixed_point_residual,
# get_parameters_optimality_measure

end # module AutoLyap
