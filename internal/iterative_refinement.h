#pragma once

#include "cupdlpx_types.h"
#include "cupdlpx.h"
#include "internal_types.h"
#ifdef __cplusplus
extern "C"
{
#endif

    // create an sub-lp problem from mask
    cupdlpx_result_t *optimize_iterative_refinement(
    const pdhg_parameters_t *params,
    const lp_problem_t *original_problem,
    const int max_refine_steps,
    const double final_tol,
    const bool verbose,
    const bool solver_verbose);
#ifdef __cplusplus
} // extern "C"
#endif