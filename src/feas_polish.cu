#include "cupdlpx.h"
#include "internal_types.h"
#include "preconditioner.h"
#include "solver.h"
#include "utils.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>

//Feasibility Polishing
void feasibility_polish(const pdhg_parameters_t *params, pdhg_solver_state_t *state)
{
    clock_t start_time = clock();
    if (state->relative_primal_residual < params->termination_criteria.eps_feas_polish_relative &&
        state->relative_dual_residual < params->termination_criteria.eps_feas_polish_relative)
    {
        
        printf("Skipping feasibility polishing as the solution is already sufficiently feasible.\n");
        return;
    }
    double original_primal_weight = 0.0;
    if (params->bound_objective_rescaling)
    {
        original_primal_weight = 1.0;
    }
    else
    {
        original_primal_weight = (state->objective_vector_norm + 1.0) / (state->constraint_bound_norm + 1.0);
    }

    //PRIMAL FEASIBILITY POLISHING
    pdhg_solver_state_t *primal_state = initialize_primal_feas_polish_state(state);
    primal_state->primal_weight = original_primal_weight;
    primal_state->best_primal_weight = original_primal_weight;
    primal_feasibility_polish(params, primal_state, state);

    if (primal_state->termination_reason == TERMINATION_REASON_FEAS_POLISH_SUCCESS)
    {
        CUDA_CHECK(cudaMemcpy(
            state->pdhg_primal_solution, primal_state->pdhg_primal_solution,
            state->num_variables * sizeof(double), cudaMemcpyDeviceToDevice));
        state->absolute_primal_residual = primal_state->absolute_primal_residual;
        state->relative_primal_residual = primal_state->relative_primal_residual;
        state->primal_objective_value = primal_state->primal_objective_value;
    }
    state->feasibility_iteration += primal_state->total_count - 1;
    
    //DUAL FEASIBILITY POLISHING
    pdhg_solver_state_t *dual_state = initialize_dual_feas_polish_state(state);
    dual_state->primal_weight = original_primal_weight;
    dual_state->best_primal_weight = original_primal_weight;
    dual_feasibility_polish(params, dual_state, state);

    if (dual_state->termination_reason == TERMINATION_REASON_FEAS_POLISH_SUCCESS)
    {
        CUDA_CHECK(cudaMemcpy(
            state->pdhg_dual_solution, dual_state->pdhg_dual_solution,
            state->num_constraints * sizeof(double), cudaMemcpyDeviceToDevice));
        state->absolute_dual_residual = dual_state->absolute_dual_residual;
        state->relative_dual_residual = dual_state->relative_dual_residual;
        state->dual_objective_value = dual_state->dual_objective_value;
    }
    state->feasibility_iteration += dual_state->total_count - 1;

    state->objective_gap = fabs(state->primal_objective_value - state->dual_objective_value);
    state->relative_objective_gap = state->objective_gap / (1.0 + fabs(state->primal_objective_value) + fabs(state->dual_objective_value));
    
    // FINAL LOGGING
    pdhg_feas_polish_final_log(primal_state, dual_state, params->verbose);
    primal_feas_polish_state_free(primal_state);
    dual_feas_polish_state_free(dual_state);

    state->feasibility_polishing_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;
    return;
}

void primal_feasibility_polish(const pdhg_parameters_t *params, pdhg_solver_state_t *state, const pdhg_solver_state_t *ori_state)
{
    print_initial_feas_polish_info(true, params);
    clock_t start_time = clock();
    bool do_restart = false;
    while (state->termination_reason == TERMINATION_REASON_UNSPECIFIED)
    {
        if ((state->is_this_major_iteration || state->total_count == 0) || (state->total_count % get_print_frequency(state->total_count) == 0))
        {
            compute_primal_feas_polish_residual(state, ori_state);

            state->cumulative_time_sec = (double)(clock() - start_time) / CLOCKS_PER_SEC;

            check_feas_polishing_termination_criteria(state, &params->termination_criteria, true);
            display_feas_polish_iteration_stats(state, params->verbose, true);
        }

        if ((state->is_this_major_iteration || state->total_count == 0))
        {
            do_restart = should_do_adaptive_restart(state, &params->restart_params, params->termination_evaluation_frequency);
            if (do_restart)
                perform_primal_restart(state);
        }

        state->is_this_major_iteration = ((state->total_count + 1) % params->termination_evaluation_frequency) == 0;

        compute_next_pdhg_primal_solution(state);
        compute_next_pdhg_dual_solution(state);

        if (state->is_this_major_iteration || do_restart)
        {
            compute_primal_fixed_point_error(state);
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
    return;
}

void dual_feasibility_polish(const pdhg_parameters_t *params, pdhg_solver_state_t *state, const pdhg_solver_state_t *ori_state)
{
    print_initial_feas_polish_info(false, params);
    clock_t start_time = clock();
    bool do_restart = false;
    while (state->termination_reason == TERMINATION_REASON_UNSPECIFIED)
    {
        if ((state->is_this_major_iteration || state->total_count == 0) || (state->total_count % get_print_frequency(state->total_count) == 0))
        {
            compute_dual_feas_polish_residual(state, ori_state);

            state->cumulative_time_sec = (double)(clock() - start_time) / CLOCKS_PER_SEC;

            check_feas_polishing_termination_criteria(state, &params->termination_criteria, false);
            display_feas_polish_iteration_stats(state, params->verbose, false);
        }

        if ((state->is_this_major_iteration || state->total_count == 0))
        {
            do_restart = should_do_adaptive_restart(state, &params->restart_params, params->termination_evaluation_frequency);
            if (do_restart)
                perform_dual_restart(state);
        }

        state->is_this_major_iteration = ((state->total_count + 1) % params->termination_evaluation_frequency) == 0;

        compute_next_pdhg_primal_solution(state);
        compute_next_pdhg_dual_solution(state);

        if (state->is_this_major_iteration || do_restart)
        {
            compute_dual_fixed_point_error(state);
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
    return;
}

static pdhg_solver_state_t *initialize_primal_feas_polish_state(
    const pdhg_solver_state_t *original_state)
{
    pdhg_solver_state_t *primal_state = (pdhg_solver_state_t *)malloc(sizeof(pdhg_solver_state_t));
    *primal_state = *original_state;
    int num_var = original_state->num_variables;
    int num_cons = original_state->num_constraints;

#define ALLOC_ZERO(dest, bytes)           \
    CUDA_CHECK(cudaMalloc(&dest, bytes)); \
    CUDA_CHECK(cudaMemset(dest, 0, bytes));

    //RESET PROBLEM TO FEASIBILITY PROBLEM
    ALLOC_ZERO(primal_state->objective_vector, num_var * sizeof(double));
    primal_state->objective_constant = 0.0;

#define ALLOC_AND_COPY_DEV(dest, src, bytes)  \
    CUDA_CHECK(cudaMalloc(&dest, bytes)); \
    CUDA_CHECK(cudaMemcpy(dest, src, bytes, cudaMemcpyDeviceToDevice));

    //ALLOCATE AND COPY SOLUTION VECTORS
    ALLOC_AND_COPY_DEV(primal_state->initial_primal_solution, original_state->initial_primal_solution, num_var * sizeof(double));
    ALLOC_AND_COPY_DEV(primal_state->current_primal_solution, original_state->current_primal_solution, num_var * sizeof(double));
    ALLOC_AND_COPY_DEV(primal_state->pdhg_primal_solution, original_state->pdhg_primal_solution, num_var * sizeof(double));
    ALLOC_AND_COPY_DEV(primal_state->reflected_primal_solution, original_state->reflected_primal_solution, num_var * sizeof(double));
    ALLOC_AND_COPY_DEV(primal_state->primal_product, original_state->primal_product, num_cons * sizeof(double));

    //ALLOC ZERO FOR OTHERS
    ALLOC_ZERO(primal_state->initial_dual_solution, num_cons * sizeof(double));
    ALLOC_ZERO(primal_state->current_dual_solution, num_cons * sizeof(double));
    ALLOC_ZERO(primal_state->pdhg_dual_solution, num_cons * sizeof(double));
    ALLOC_ZERO(primal_state->reflected_dual_solution, num_cons * sizeof(double));
    ALLOC_ZERO(primal_state->dual_product, num_var * sizeof(double));

    ALLOC_ZERO(primal_state->dual_slack, num_var * sizeof(double));
    ALLOC_ZERO(primal_state->primal_slack, num_cons * sizeof(double));
    ALLOC_ZERO(primal_state->dual_residual, num_var * sizeof(double));
    ALLOC_ZERO(primal_state->primal_residual, num_cons * sizeof(double));
    ALLOC_ZERO(primal_state->delta_primal_solution, num_var * sizeof(double));
    ALLOC_ZERO(primal_state->delta_dual_solution, num_cons * sizeof(double));

    //RESET SCALAR
    primal_state->primal_weight_error_sum = 0.0;
    primal_state->primal_weight_last_error = 0.0;
    primal_state->best_primal_weight = 0.0;
    primal_state->fixed_point_error = INFINITY;
    primal_state->initial_fixed_point_error = INFINITY;
    primal_state->last_trial_fixed_point_error = INFINITY;
    primal_state->step_size = original_state->step_size;
    primal_state->primal_weight = original_state->primal_weight;
    primal_state->is_this_major_iteration = false;
    primal_state->total_count = 0;
    primal_state->inner_count = 0;
    primal_state->termination_reason = TERMINATION_REASON_UNSPECIFIED;
    primal_state->cumulative_time_sec = 0.0;
    primal_state->best_primal_dual_residual_gap = INFINITY;

    //IGNORE DUAL RESIDUAL AND OBJECTIVE GAP
    primal_state->relative_dual_residual = 0.0;
    primal_state->absolute_dual_residual = 0.0;
    primal_state->relative_objective_gap = 0.0;
    primal_state->objective_gap = 0.0;

    return primal_state;
}

void primal_feas_polish_state_free(pdhg_solver_state_t *state)
{
    #define SAFE_CUDA_FREE(p)   \
    if ((p) != NULL) {          \
        CUDA_CHECK(cudaFree(p));\
        (p) = NULL;             \
    }                           \
    
    if (!state) return;
    SAFE_CUDA_FREE(state->objective_vector);
    SAFE_CUDA_FREE(state->initial_primal_solution);
    SAFE_CUDA_FREE(state->current_primal_solution);
    SAFE_CUDA_FREE(state->pdhg_primal_solution);
    SAFE_CUDA_FREE(state->reflected_primal_solution);

    SAFE_CUDA_FREE(state->dual_product);
    SAFE_CUDA_FREE(state->initial_dual_solution);
    SAFE_CUDA_FREE(state->current_dual_solution);
    SAFE_CUDA_FREE(state->pdhg_dual_solution);
    SAFE_CUDA_FREE(state->reflected_dual_solution);
    SAFE_CUDA_FREE(state->primal_product);
    
    SAFE_CUDA_FREE(state->primal_slack);
    SAFE_CUDA_FREE(state->dual_slack);
    SAFE_CUDA_FREE(state->primal_residual);
    SAFE_CUDA_FREE(state->dual_residual);
    SAFE_CUDA_FREE(state->delta_primal_solution);
    SAFE_CUDA_FREE(state->delta_dual_solution);
    free(state);
}

__global__ void zero_finite_value_vectors_kernel(
    double *__restrict__ vec, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        if (isfinite(vec[idx]))
            vec[idx] = 0.0;
    }
}
    

static pdhg_solver_state_t *initialize_dual_feas_polish_state(
    const pdhg_solver_state_t *original_state)
{
    pdhg_solver_state_t *dual_state = (pdhg_solver_state_t *)malloc(sizeof(pdhg_solver_state_t));
    *dual_state = *original_state;
    int num_var = original_state->num_variables;
    int num_cons = original_state->num_constraints;

    #define ALLOC_AND_COPY_DEV(dest, src, bytes)  \
        CUDA_CHECK(cudaMalloc(&dest, bytes)); \
        CUDA_CHECK(cudaMemcpy(dest, src, bytes, cudaMemcpyDeviceToDevice));

    //RESET PROBLEM TO DUAL FEASIBILITY PROBLEM
    #define SET_FINITE_TO_ZERO(vec, n) \
        { \
            int threads = 256; \
            int blocks = (n + threads - 1) / threads; \
            zero_finite_value_vectors_kernel<<<blocks, threads>>>(vec, n); \
            CUDA_CHECK(cudaDeviceSynchronize()); \
        }

    ALLOC_AND_COPY_DEV(dual_state->constraint_lower_bound, original_state->constraint_lower_bound, num_cons * sizeof(double));
    ALLOC_AND_COPY_DEV(dual_state->constraint_upper_bound, original_state->constraint_upper_bound, num_cons * sizeof(double));
    ALLOC_AND_COPY_DEV(dual_state->variable_lower_bound, original_state->variable_lower_bound, num_var * sizeof(double));
    ALLOC_AND_COPY_DEV(dual_state->variable_upper_bound, original_state->variable_upper_bound, num_var * sizeof(double));

    SET_FINITE_TO_ZERO(dual_state->constraint_lower_bound, num_cons);
    SET_FINITE_TO_ZERO(dual_state->constraint_upper_bound, num_cons);
    SET_FINITE_TO_ZERO(dual_state->variable_lower_bound, num_var);
    SET_FINITE_TO_ZERO(dual_state->variable_upper_bound, num_var);

    #define ALLOC_ZERO(dest, bytes)           \
        CUDA_CHECK(cudaMalloc(&dest, bytes)); \
        CUDA_CHECK(cudaMemset(dest, 0, bytes));

    ALLOC_ZERO(dual_state->constraint_lower_bound_finite_val, num_cons * sizeof(double));
    ALLOC_ZERO(dual_state->constraint_upper_bound_finite_val, num_cons * sizeof(double));
    ALLOC_ZERO(dual_state->variable_lower_bound_finite_val, num_var * sizeof(double));
    ALLOC_ZERO(dual_state->variable_upper_bound_finite_val, num_var * sizeof(double));

    //ALLOCATE AND COPY SOLUTION VECTORS
    ALLOC_AND_COPY_DEV(dual_state->initial_dual_solution, original_state->initial_dual_solution, num_cons * sizeof(double));
    ALLOC_AND_COPY_DEV(dual_state->current_dual_solution, original_state->current_dual_solution, num_cons * sizeof(double));
    ALLOC_AND_COPY_DEV(dual_state->pdhg_dual_solution, original_state->pdhg_dual_solution, num_cons * sizeof(double));
    ALLOC_AND_COPY_DEV(dual_state->reflected_dual_solution, original_state->reflected_dual_solution, num_cons * sizeof(double));
    ALLOC_AND_COPY_DEV(dual_state->dual_product, original_state->dual_product, num_var * sizeof(double));
    ALLOC_AND_COPY_DEV(dual_state->dual_slack, original_state->dual_slack, num_var * sizeof(double));

    //ALLOC ZERO FOR OTHERS
    ALLOC_ZERO(dual_state->initial_primal_solution, num_var * sizeof(double));
    ALLOC_ZERO(dual_state->current_primal_solution, num_var * sizeof(double));
    ALLOC_ZERO(dual_state->pdhg_primal_solution, num_var * sizeof(double));
    ALLOC_ZERO(dual_state->reflected_primal_solution, num_var * sizeof(double));
    ALLOC_ZERO(dual_state->primal_product, num_cons * sizeof(double));
    ALLOC_ZERO(dual_state->primal_slack, num_cons * sizeof(double));
    ALLOC_ZERO(dual_state->dual_residual, num_var * sizeof(double));
    ALLOC_ZERO(dual_state->primal_residual, num_cons * sizeof(double));
    ALLOC_ZERO(dual_state->delta_primal_solution, num_var * sizeof(double));
    ALLOC_ZERO(dual_state->delta_dual_solution, num_cons * sizeof(double));

    //RESET SCALAR
    dual_state->primal_weight_error_sum = 0.0;
    dual_state->primal_weight_last_error = 0.0;
    dual_state->best_primal_weight = 0.0;
    dual_state->fixed_point_error = INFINITY;
    dual_state->initial_fixed_point_error = INFINITY;
    dual_state->last_trial_fixed_point_error = INFINITY;
    dual_state->step_size = original_state->step_size;
    dual_state->primal_weight = original_state->primal_weight;
    dual_state->is_this_major_iteration = false;
    dual_state->total_count = 0;
    dual_state->inner_count = 0;
    dual_state->termination_reason = TERMINATION_REASON_UNSPECIFIED;
    dual_state->cumulative_time_sec = 0.0;
    dual_state->best_primal_dual_residual_gap = INFINITY;

    //IGNORE PRIMAL RESIDUAL AND OBJECTIVE GAP
    dual_state->relative_primal_residual = 0.0;
    dual_state->absolute_primal_residual = 0.0;
    dual_state->relative_objective_gap = 0.0;
    dual_state->objective_gap = 0.0;
    return dual_state;
}

void dual_feas_polish_state_free(pdhg_solver_state_t *state)
{
    #define SAFE_CUDA_FREE(p)   \
    if ((p) != NULL) {          \
        CUDA_CHECK(cudaFree(p));\
        (p) = NULL;             \
    }                           \
    
    if (!state) return;
    SAFE_CUDA_FREE(state->constraint_lower_bound);
    SAFE_CUDA_FREE(state->constraint_upper_bound);
    SAFE_CUDA_FREE(state->variable_lower_bound);
    SAFE_CUDA_FREE(state->variable_upper_bound);
    SAFE_CUDA_FREE(state->constraint_lower_bound_finite_val);
    SAFE_CUDA_FREE(state->constraint_upper_bound_finite_val);
    SAFE_CUDA_FREE(state->variable_lower_bound_finite_val);
    SAFE_CUDA_FREE(state->variable_upper_bound_finite_val);

    SAFE_CUDA_FREE(state->initial_primal_solution);
    SAFE_CUDA_FREE(state->current_primal_solution);
    SAFE_CUDA_FREE(state->pdhg_primal_solution);
    SAFE_CUDA_FREE(state->reflected_primal_solution);

    SAFE_CUDA_FREE(state->dual_product);
    SAFE_CUDA_FREE(state->initial_dual_solution);
    SAFE_CUDA_FREE(state->current_dual_solution);
    SAFE_CUDA_FREE(state->pdhg_dual_solution);
    SAFE_CUDA_FREE(state->reflected_dual_solution);
    SAFE_CUDA_FREE(state->primal_product);
    
    SAFE_CUDA_FREE(state->primal_slack);
    SAFE_CUDA_FREE(state->dual_slack);
    SAFE_CUDA_FREE(state->primal_residual);
    SAFE_CUDA_FREE(state->dual_residual);
    SAFE_CUDA_FREE(state->delta_primal_solution);
    SAFE_CUDA_FREE(state->delta_dual_solution);
    free(state);
}

static void perform_primal_restart(pdhg_solver_state_t *state)
{
    CUDA_CHECK(cudaMemcpy(state->initial_primal_solution, state->pdhg_primal_solution, state->num_variables * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state->current_primal_solution, state->pdhg_primal_solution, state->num_variables * sizeof(double), cudaMemcpyDeviceToDevice));
    state->inner_count = 0;
    state->last_trial_fixed_point_error = INFINITY;
}

static void perform_dual_restart(pdhg_solver_state_t *state)
{
    CUDA_CHECK(cudaMemcpy(state->initial_dual_solution, state->pdhg_dual_solution, state->num_constraints * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state->current_dual_solution, state->pdhg_dual_solution, state->num_constraints * sizeof(double), cudaMemcpyDeviceToDevice));
    state->inner_count = 0;
    state->last_trial_fixed_point_error = INFINITY;
}

__global__ void compute_delta_primal_solution_kernel(
    const double *initial_primal, const double *pdhg_primal, double *delta_primal,
    int n_vars)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_vars)
    {
        delta_primal[i] = pdhg_primal[i] - initial_primal[i];
    }
}

__global__ void compute_delta_dual_solution_kernel(
    const double *initial_dual, const double *pdhg_dual, double *delta_dual,
    int n_cons)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_cons)
    {
        delta_dual[i] = pdhg_dual[i] - initial_dual[i];
    }
}

static void compute_primal_fixed_point_error(pdhg_solver_state_t *state)
{
    compute_delta_primal_solution_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
        state->current_primal_solution,
        state->reflected_primal_solution,
        state->delta_primal_solution,
        state->num_variables);
    double primal_norm = 0.0;
    CUBLAS_CHECK(cublasDnrm2_v2_64(state->blas_handle,
                                   state->num_variables,
                                   state->delta_primal_solution,
                                   1,
                                   &primal_norm));
    state->fixed_point_error = primal_norm * primal_norm * state->primal_weight;
}

static void compute_dual_fixed_point_error(pdhg_solver_state_t *state)
{
    compute_delta_dual_solution_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(
        state->current_dual_solution,
        state->reflected_dual_solution,
        state->delta_dual_solution,
        state->num_constraints);
    double dual_norm = 0.0;
    CUBLAS_CHECK(cublasDnrm2_v2_64(state->blas_handle,
                                   state->num_constraints,
                                   state->delta_dual_solution,
                                   1,
                                   &dual_norm));
    state->fixed_point_error = dual_norm * dual_norm / state->primal_weight;
}


