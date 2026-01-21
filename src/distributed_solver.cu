/*
Copyright 2025 Haihao Lu

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "cupdlpx.h"
#include "distribution_utils.h"
#include "distributed_solver.h"
#include "distributed_op.h"
#include "internal_types.h"
#include "preconditioner.h"
#include "presolve.h"
#include "solver.h"
#include "core_operation.h"
#include "utils.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>
#include <mpi.h>

static void allreduce_obj_bound_norm(pdhg_solver_state_t *state, const pdhg_parameters_t *params);

static cupdlpx_result_t *distributed_optimize_core(const pdhg_parameters_t *params,
                            const lp_problem_t *original_problem,
                            grid_context_t *grid_context);

cupdlpx_result_t *distributed_optimize(
    const pdhg_parameters_t *params,
    const lp_problem_t *original_problem
)
{
    grid_context_t grid_context = initialize_parallel_context(params->grid_shape.row_dims, params->grid_shape.col_dims);
    pdhg_parameters_t sub_params = *params;
    sub_params.verbose = (grid_context.rank_global == 0) ? params->verbose : false;

    return distributed_optimize_core(&sub_params, original_problem, &grid_context);
}

static cupdlpx_result_t *distributed_optimize_core(const pdhg_parameters_t *params,
                            const lp_problem_t *original_problem,
                            grid_context_t *grid_context)
{
    print_initial_info(params, original_problem);
    const lp_problem_t *working_problem = original_problem;

    rescale_info_t *rescale_info = NULL;
    cupdlpx_presolve_info_t *presolve_info = NULL;

    if (grid_context->rank_global == 0) 
    {
        if (params->presolve)
        {
            presolve_info = pslp_presolve(original_problem, params);
            if (presolve_info->problem_solved_during_presolve)
            {
                //TODO: Deal with cases that solved during presolve
                cupdlpx_result_t *result = create_result_from_presolve(presolve_info, original_problem);
                cupdlpx_presolve_info_free(presolve_info);
                pdhg_final_log(result, params);
                return result;
            }
            working_problem = presolve_info->reduced_problem;
        }
        rescale_info = rescale_problem(params, working_problem);
    }
    if (params->verbose) printf("Rank 0: Preprocess Complete!\n");
    
    {
        char *buf = NULL;
        size_t sz = 0;

        if (grid_context->rank_global == 0) {
            sz = get_lp_problem_size(working_problem);
            buf = (char*)malloc(sz);
            char *ptr_tmp = buf; 
            serialize_lp_problem_to_ptr(working_problem, &ptr_tmp);
        }

        big_bcast_bytes((void**)&buf, &sz, 0, grid_context->comm_global);

        if (grid_context->rank_global != 0) {
            const char *ptr_tmp = buf;
            working_problem = deserialize_lp_problem_from_ptr(&ptr_tmp);
        }

        if (buf) free(buf);
    }
    if (params->verbose) printf("Rank 0: Synchronize Problem!\n");
    {
        char *buf = NULL;
        size_t sz = 0;

        if (grid_context->rank_global == 0) {
            sz = get_rescale_info_size(rescale_info);
            buf = (char*)malloc(sz);
            serialize_rescale_info(rescale_info, buf);
        }

        big_bcast_bytes((void**)&buf, &sz, 0, grid_context->comm_global);

        if (grid_context->rank_global != 0) {
            rescale_info = deserialize_rescale_info(buf);
        }

        if (buf) free(buf);
    }
    if (params->verbose) printf("Rank 0: Synchronize Rescaling Info!\n");
    int n_start = 0;
    int m_start = 0;
    rescale_info_t *local_rescale_info = partition_rescale_info(
        rescale_info, 
        grid_context,
        params->partition_method,
        &n_start,  
        &m_start  
    );
    rescale_info_free(rescale_info);
    if (params->verbose) printf("Rank 0: Rescaling Info Partitioned!\n");
    lp_problem_t *local_working_problem = partition_lp_problem(
        working_problem,
        grid_context,
        params->partition_method,
        &n_start,  
        &m_start  
    );
    if (params->verbose) printf("Rank 0: Problem Partitioned!\n");
    pdhg_solver_state_t *state = initialize_solver_state(params, local_working_problem, local_rescale_info);
    state->grid_context = grid_context;
    allreduce_obj_bound_norm(state, params);


    rescale_info_free(local_rescale_info);

    //TODO: Support distributed Power Method
    // initialize_step_size_and_primal_weight(state, params);
    state->step_size = 1.0;
    state->primal_weight = 1.0;
    
    clock_t start_time = clock();
    bool do_restart = false;
    while (state->total_count < params->termination_criteria.iteration_limit)
    {
        if ((state->is_this_major_iteration || state->total_count == 0) ||
            (state->total_count % get_print_frequency(state->total_count) == 0))
        {
            compute_residual_distributed(state, params->optimality_norm);
            if (state->is_this_major_iteration &&
                state->total_count < 3 * params->termination_evaluation_frequency)
            {
                compute_infeasibility_information(state);
            }

            state->cumulative_time_sec =
                (double)(clock() - start_time) / CLOCKS_PER_SEC;

            check_termination_criteria(state, &params->termination_criteria);
            display_iteration_stats(state, params->verbose);
            if (state->termination_reason != TERMINATION_REASON_UNSPECIFIED) {
                break;
            }
        }

        if ((state->is_this_major_iteration || state->total_count == 0))
        {
            do_restart =
                should_do_adaptive_restart(state, &params->restart_params,
                                           params->termination_evaluation_frequency);
            if (do_restart)
                perform_restart_distributed(state, params);
        }

        state->is_this_major_iteration =
            ((state->total_count + 1) % params->termination_evaluation_frequency) ==
            0;

        compute_next_pdhg_primal_solution_distributed(state);
        compute_next_pdhg_dual_solution_distributed(state);

        if (state->is_this_major_iteration || do_restart)
        {
            compute_fixed_point_error_distributed(state);
            if (do_restart)
            {
                state->initial_fixed_point_error = state->fixed_point_error;
                do_restart = false;
            }
        }
        halpern_update(state, params->reflection_coefficient);

        state->inner_count++;
        state->total_count++;
    }

    if (state->termination_reason == TERMINATION_REASON_UNSPECIFIED)
    {
        state->termination_reason = TERMINATION_REASON_ITERATION_LIMIT;
        compute_residual(state, params->optimality_norm);
        display_iteration_stats(state, params->verbose);
    }

    cupdlpx_result_t *result = create_result_from_state(state, original_problem);

    if (params->presolve && presolve_info)
    {
        pslp_postsolve(presolve_info, result, original_problem);
        cupdlpx_presolve_info_free(presolve_info);
    }

    pdhg_final_log(result, params);
    pdhg_solver_state_free(state);
    return result;
}

static void allreduce_obj_bound_norm(pdhg_solver_state_t *state, const pdhg_parameters_t *params){
    if (params->optimality_norm == NORM_TYPE_L_INF) {
        double local_val = state->objective_vector_norm;
        MPI_Allreduce(&local_val, &state->objective_vector_norm, 1, 
                      MPI_DOUBLE, MPI_MAX, state->grid_context->comm_row);
    } else {
        double local_sq = state->objective_vector_norm * state->objective_vector_norm;
        double global_sq = 0.0;
        MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, 
                      state->grid_context->comm_row);
        state->objective_vector_norm = sqrt(global_sq);
    }

    if (params->optimality_norm == NORM_TYPE_L_INF) {
        double local_val = state->constraint_bound_norm;
        MPI_Allreduce(&local_val, &state->constraint_bound_norm, 1, 
                      MPI_DOUBLE, MPI_MAX, state->grid_context->comm_col);
    } else {
        double local_sq = state->constraint_bound_norm * state->constraint_bound_norm;
        double global_sq = 0.0;
        MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, 
                      state->grid_context->comm_col);
        state->constraint_bound_norm = sqrt(global_sq);
    }
}