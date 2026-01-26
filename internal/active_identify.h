#pragma once

#include "cupdlpx_types.h"
#include "cupdlpx.h"
#include "internal_types.h"
#ifdef __cplusplus
extern "C"
{
#endif

    // create an sub-lp problem from mask
    lp_problem_t *construct_sub_lp(
        const lp_problem_t *complete_lp,
        const bool *mask_row,
        const bool *mask_col,
        const pdhg_solver_state_t *state);
    matrix_desc_t slice_matrix(const matrix_desc_t *A, const bool *mask_row, const bool *mask_col);
    cupdlpx_result_t *optimize_with_adaptive_active_identify(
        const pdhg_parameters_t *params,
        const lp_problem_t *original_problem,
        const int max_adaptive_iteration,
        const double init_mask_threshold,
        const double tol,
        const double time_limit,
        bool verbose);
    cupdlpx_result_t *optimize_two_stage(
        const pdhg_parameters_t *params,
        const lp_problem_t *original_problem,
        const double coarse_tol, // e.g., 1e-4
        const double fine_tol,  
        const double time_limit,
        const bool oscillation_based_scaling,
        bool verbose,
        bool inner_verbose);
    typedef struct {
        int  orig_n;
        int  orig_m;
        int  init_reduced_n;
        int  init_reduced_m;
        int  final_reduced_n;
        int  final_reduced_m;
        int  phase0_iter;
        double   phase0_time;
        int  adaptive_iter_sum;
        double   adaptive_time_sum;
        int  adaptive_loops;
        int  final_iter;
        double   final_time;
        double   total_wall_time;
        double   final_res;
    } adaptive_active_identify_stats_t;
#ifdef __cplusplus
} // extern "C"
#endif