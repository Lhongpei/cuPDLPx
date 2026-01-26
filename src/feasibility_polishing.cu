#include "feasibility_polishing.h"
#include "cupdlpx.h"
#include "internal_types.h"
#include "preconditioner.h"
#include "utils.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>

#define ALLOC_ZERO(dest, bytes)           \
    CUDA_CHECK(cudaMalloc(&dest, bytes)); \
    CUDA_CHECK(cudaMemset(dest, 0, bytes));

void primal_feasibility_iterate_once(
    pdhg_solver_state_t *state,
    double *d1,
    double *d2,
    double *d1_prev,
    double *d2_prev,
    double *d1_bar,
    double *d2_bar,
    double *d1_moment,
    double *d2_moment,
    double *grad_d1,
    double *grad_d2,
    double *delta_d,
    double *direction,
    double *t_curr,
    double *t_prev);

// To use previous api, we update solution in pdhg_primal_solution
// and use initial_primal_solution to store solution before
// primal feasibility polishing
void proj_scheme_primal_feasibility_polishing(
    pdhg_solver_state_t *state,
    const pdhg_parameters_t *params)
{
    int num_con = state->num_constraints;
    int num_var = state->num_variables;

    double *d1, *d2, *d1_prev, *d2_prev, *d1_bar, *d2_bar, *grad_d1, *grad_d2, *d1_moment, *d2_moment, *delta_d, *direction;
    ALLOC_ZERO(d1, num_con * sizeof(double));
    ALLOC_ZERO(d2, num_con * sizeof(double));
    ALLOC_ZERO(d1_prev, num_con * sizeof(double));
    ALLOC_ZERO(d2_prev, num_con * sizeof(double));
    ALLOC_ZERO(d1_bar, num_con * sizeof(double));
    ALLOC_ZERO(d2_bar, num_con * sizeof(double));
    ALLOC_ZERO(grad_d1, num_con * sizeof(double));
    ALLOC_ZERO(grad_d2, num_var * sizeof(double));
    ALLOC_ZERO(d1_moment, num_con * sizeof(double));
    ALLOC_ZERO(d2_moment, num_con * sizeof(double));
    ALLOC_ZERO(delta_d, num_con * sizeof(double));
    ALLOC_ZERO(direction, num_var * sizeof(double));
    state->total_count = 0;
    state->cumulative_time_sec = 0.0;
    double t_prev = 1.0;
    double t_curr = 1.0;
    state->termination_reason = TERMINATION_REASON_UNSPECIFIED;
    CUDA_CHECK(cudaMemcpy(
        state->initial_primal_solution, state->pdhg_primal_solution,
        state->num_variables * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(
        state->current_primal_solution, state->pdhg_primal_solution,
        state->num_variables * sizeof(double), cudaMemcpyDeviceToDevice));
    print_initial_feas_polish_info(true, params);
    clock_t start_time = clock();
    while (state->termination_reason == TERMINATION_REASON_UNSPECIFIED)
    {
        if ((state->is_this_major_iteration || state->total_count == 0) || (state->total_count % get_print_frequency(state->total_count) == 0))
        {
            compute_primal_feas_polish_residual(state, state->objective_vector);

            state->cumulative_time_sec = (double)(clock() - start_time) / CLOCKS_PER_SEC;

            check_feas_polishing_termination_criteria(state, &params->termination_criteria, true);
            display_feas_polish_iteration_stats(state, params->verbose, true);
        }
        primal_feasibility_iterate_once(
            state, d1, d2, d1_prev, d2_prev, d1_bar,
            d2_bar, d1_moment, d2_moment, grad_d1, grad_d2, delta_d, direction,
            &t_curr, &t_prev);
        state->total_count++;
    }
    if (state->termination_reason == TERMINATION_REASON_FEAS_POLISH_SUCCESS)
    {
        CUDA_CHECK(cudaMemcpy(
            state->current_primal_solution, state->pdhg_primal_solution,
            state->num_variables * sizeof(double), cudaMemcpyDeviceToDevice));
    }
}

void dual_feasibility_iterate_once(
    pdhg_solver_state_t *state,
    double *lambda,
    double *lambda_prev,
    double *lambda_bar,
    double *lambda_moment,
    double *grad_lambda,
    double *direction_y,
    double *original_dual_slack,
    double *t_curr,
    double *t_prev);



__global__ void primal_feasibility_iterate_kernel(
    double *d1,
    double *d2,
    double *d1_prev,
    double *d2_prev,
    double *grad_d1,
    double *grad_d2,
    double step_size,
    const double *__restrict__ d1_bar,
    const double *__restrict__ d2_bar,
    const double *__restrict__ Ax,
    const double *__restrict__ AL,
    const double *__restrict__ AU,
    int num_con)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_con)
        return;
    d1_prev[idx] = d1[idx];
    d2_prev[idx] = d2[idx];
    double AL_i = AL[idx];
    double AU_i = AU[idx];
    double Ax_i = Ax[idx];
    bool is_AL_i_finite = isfinite(AL_i);
    bool is_AU_i_finite = isfinite(AU_i);
    double grad_d1_i = (AL_i - Ax_i);
    double grad_d2_i = (Ax_i - AU_i);

    double d1_new = d1_bar[idx];
    if (is_AL_i_finite)
    {
        d1_new += step_size * grad_d1_i;
    }
    d1[idx] = fmax(d1_new, 0.0);

    double d2_new = d2_bar[idx];
    if (is_AU_i_finite)
    {
        d2_new += step_size * grad_d2_i;
    }
    d2[idx] = fmax(d2_new, 0.0);

    grad_d1[idx] = grad_d1_i * is_AL_i_finite;
    grad_d2[idx] = grad_d2_i * is_AU_i_finite;
}

__global__ void primal_feasibility_update_moment(
    double *d1_bar,
    double *d2_bar,
    double *d1_moment,
    double *d2_moment,
    double *delta_d,
    const double *__restrict__ d1,
    const double *__restrict__ d2,
    const double *__restrict__ d1_prev,
    const double *__restrict__ d2_prev,
    double beta,
    int num_con)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_con)
        return;
    double d1_delta = d1[idx] - d1_prev[idx];
    double d2_delta = d2[idx] - d2_prev[idx];
    double d1_bar_i = d1[idx] + beta * d1_delta;
    double d2_bar_i = d2[idx] + beta * d2_delta;
    delta_d[idx] = d1_bar_i - d2_bar_i;
    d1_moment[idx] = d1_delta;
    d2_moment[idx] = d2_delta;
    d1_bar[idx] = d1_bar_i;
    d2_bar[idx] = d2_bar_i;
}

__global__ void proj_grad_update_primal_solution(
    double *current_primal_solution,
    const double *__restrict__ target_primal_solution,
    const double *__restrict__ direction,
    const double *__restrict__ var_lower_bound,
    const double *__restrict__ var_upper_bound,
    int num_var)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_var)
        return;
    double updated_value =
        target_primal_solution[idx] + direction[idx];
    current_primal_solution[idx] =
        fmin(fmax(updated_value, var_lower_bound[idx]), var_upper_bound[idx]);
}

void primal_feasibility_iterate_once(
    pdhg_solver_state_t *state,
    double *d1,
    double *d2,
    double *d1_prev,
    double *d2_prev,
    double *d1_bar,
    double *d2_bar,
    double *d1_moment,
    double *d2_moment,
    double *grad_d1,
    double *grad_d2,
    double *delta_d,
    double *direction,
    double *t_curr,
    double *t_prev)
{
    *t_curr = 0.5 * (1.0 + sqrt(1.0 + 4.0 * (*t_prev) * (*t_prev)));
    double beta = (*t_prev - 1.0) / (*t_curr);
    primal_feasibility_update_moment<<<state->num_blocks_dual,
                                       THREADS_PER_BLOCK>>>(
        d1_bar, d2_bar, d1_moment, d2_moment, delta_d, d1, d2, d1_prev, d2_prev,
        beta, state->num_constraints);
    // Update current solution
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_dual_sol,
                                          delta_d));
    CUSPARSE_CHECK(
        cusparseDnVecSetValues(state->vec_dual_prod, direction));
    CUSPARSE_CHECK(cusparseSpMV(
        state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE,
        state->matAt, state->vec_dual_sol, &HOST_ZERO, state->vec_dual_prod,
        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->dual_spmv_buffer));
    proj_grad_update_primal_solution<<<state->num_blocks_primal,
                                       THREADS_PER_BLOCK>>>(
        state->pdhg_primal_solution, state->initial_primal_solution, direction,
        state->variable_lower_bound, state->variable_upper_bound, state->num_variables);
        
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_primal_sol,
                                          state->pdhg_primal_solution));
    CUSPARSE_CHECK(
        cusparseDnVecSetValues(state->vec_primal_prod, state->primal_product));

    CUSPARSE_CHECK(cusparseSpMV(
        state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE,
        state->matA, state->vec_primal_sol, &HOST_ZERO, state->vec_primal_prod,
        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->primal_spmv_buffer));

    primal_feasibility_iterate_kernel<<<state->num_blocks_dual,
                                        THREADS_PER_BLOCK>>>(
        d1, d2, d1_prev, d2_prev, grad_d1, grad_d2, state->step_size / 2, d1_bar,
        d2_bar, state->primal_product, state->constraint_lower_bound,
        state->constraint_upper_bound, state->num_constraints);
    double d1_dot, d2_dot;
    CUBLAS_CHECK(cublasDdot(
        state->blas_handle, state->num_constraints, grad_d1, 1, d1_moment, 1, &d1_dot));
    CUBLAS_CHECK(cublasDdot(
        state->blas_handle, state->num_constraints, grad_d2, 1, d2_moment, 1, &d2_dot));
    *t_prev = d1_dot + d2_dot < 0 ? 1.0 : *t_curr;

}

// To use previous api, we update solution in pdhg_dual_solution
// and use initial_dual_solution to store solution before
// dual feasibility polishing
void proj_scheme_dual_feasibility_polishing(
    pdhg_solver_state_t *state,
    const pdhg_parameters_t *params)
{
    int num_con = state->num_constraints;
    int num_var = state->num_variables;

    double *lambda, *lambda_prev, *lambda_bar, *grad_lambda, *lambda_moment, *direction_y, *original_dual_slack;
    ALLOC_ZERO(lambda, num_var * sizeof(double));
    ALLOC_ZERO(lambda_prev, num_var * sizeof(double));
    ALLOC_ZERO(lambda_bar, num_var * sizeof(double));
    ALLOC_ZERO(grad_lambda, num_var * sizeof(double));
    ALLOC_ZERO(lambda_moment, num_var * sizeof(double));
    ALLOC_ZERO(original_dual_slack, num_var * sizeof(double));
    ALLOC_ZERO(direction_y, num_con * sizeof(double));
    state->total_count = 0;
    state->cumulative_time_sec = 0.0;
    double t_prev = 1.0;
    double t_curr = 1.0;
    state->termination_reason = TERMINATION_REASON_UNSPECIFIED;
    CUDA_CHECK(cudaMemcpy(
        state->initial_dual_solution, state->pdhg_dual_solution,
        state->num_constraints * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(
        state->current_dual_solution, state->pdhg_dual_solution,
        state->num_constraints * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK((cudaMemcpy(
        original_dual_slack, state->dual_slack,
        state->num_variables * sizeof(double), cudaMemcpyDeviceToDevice)));
    print_initial_feas_polish_info(false, params);
    clock_t start_time = clock();
    while (state->termination_reason == TERMINATION_REASON_UNSPECIFIED)
    {
        if ((state->is_this_major_iteration || state->total_count == 0) || (state->total_count % get_print_frequency(state->total_count) == 0))
        {
            compute_dual_feas_polish_residual(state, state->constraint_lower_bound_finite_val,
                                              state->constraint_upper_bound_finite_val,
                                              state->initial_primal_solution);

            state->cumulative_time_sec = (double)(clock() - start_time) / CLOCKS_PER_SEC;

            check_feas_polishing_termination_criteria(state, &params->termination_criteria, false);
            display_feas_polish_iteration_stats(state, params->verbose, false);
        }
        dual_feasibility_iterate_once(
            state, lambda, lambda_prev, lambda_bar,
            lambda_moment, grad_lambda, direction_y, original_dual_slack,
            &t_curr, &t_prev);
        state->total_count++;
    }
    return;
}

__global__ void dual_feasibility_iterate_kernel(
    double *lambda,
    double *lambda_prev,
    const double *__restrict__ lambda_bar,
    double *grad_lambda,
    double step_size,
    const double *__restrict__ dual_product,
    const double *__restrict__ objective_vector,
    const double *__restrict__ dual_slack,
    int num_var)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_var)
        return;
    lambda_prev[idx] = lambda[idx];
    double grad_lambda_i = objective_vector[idx] - dual_product[idx] - dual_slack[idx];
    lambda[idx] = lambda_bar[idx] + step_size * grad_lambda_i;
    grad_lambda[idx] = grad_lambda_i;
}

__global__ void dual_feasibility_update_moment(
    double *lambda_bar,
    double *lambda_moment,
    double *lambda,
    double *lambda_prev,
    double beta,
    int num_var)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_var)
        return;
    double lambda_delta = lambda[idx] - lambda_prev[idx];
    double lambda_bar_i = lambda[idx] + beta * lambda_delta;
    lambda_moment[idx] = lambda_delta;
    lambda_bar[idx] = lambda_bar_i;
}

__global__ void proj_grad_update_dual_solution(
    double *current_dual_solution,
    double *current_dual_slack,
    const double *__restrict__ target_dual_solution,
    const double *__restrict__ original_dual_slack,
    const double *__restrict__ direction_y,
    const double *__restrict__ direction_s,
    const double *__restrict__ constraint_lower_bound,
    const double *__restrict__ constraint_upper_bound,
    const double *__restrict__ variable_lower_bound,
    const double *__restrict__ variable_upper_bound,
    int num_con,
    int num_var)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_con + num_var)
        return;
    if (idx < num_var)
    {
        double s_min = isfinite(variable_upper_bound[idx]) ? -INFINITY : 0.0;
        double s_max = isfinite(variable_lower_bound[idx]) ? INFINITY : 0.0;
        
        double updated_value =
            original_dual_slack[idx] + direction_s[idx];
        current_dual_slack[idx] =
            fmin(fmax(updated_value, s_min), s_max);
    }
    else
    {
        int con_idx = idx - num_var;
        
        double y_min = isfinite(constraint_upper_bound[con_idx]) ? -INFINITY: 0.0;
        double y_max = isfinite(constraint_lower_bound[con_idx]) ? INFINITY : 0.0;
        
        double updated_value =
            target_dual_solution[con_idx] + direction_y[con_idx];
        current_dual_solution[con_idx] =
            fmin(fmax(updated_value, y_min), y_max);
    }
}

void dual_feasibility_iterate_once(
    pdhg_solver_state_t *state,
    double *lambda,
    double *lambda_prev,
    double *lambda_bar,
    double *lambda_moment,
    double *grad_lambda,
    double *direction_y,
    double *original_dual_slack,
    double *t_curr,
    double *t_prev)
{
        // Update current solution
    *t_curr = 0.5 * (1.0 + sqrt(1.0 + 4.0 * (*t_prev) * (*t_prev)));
    double beta = (*t_prev - 1.0) / (*t_curr);
    dual_feasibility_update_moment<<<state->num_blocks_primal,
                                     THREADS_PER_BLOCK>>>(
        lambda_bar, lambda_moment, lambda, lambda_prev, beta, state->num_variables);

    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_primal_sol,
                                          lambda_bar));
    CUSPARSE_CHECK(
        cusparseDnVecSetValues(state->vec_primal_prod, direction_y));
    CUSPARSE_CHECK(cusparseSpMV(
        state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE,
        state->matA, state->vec_primal_sol, &HOST_ZERO, state->vec_primal_prod,
        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->primal_spmv_buffer));
    proj_grad_update_dual_solution<<<state->num_blocks_primal_dual,
                                     THREADS_PER_BLOCK>>>(
        state->pdhg_dual_solution, state->dual_slack, state->initial_dual_solution, original_dual_slack, direction_y, lambda_bar,
        state->constraint_lower_bound, state->constraint_upper_bound, state->variable_lower_bound, state->variable_upper_bound,
        state->num_constraints, state->num_variables);

    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_dual_sol,
                                          state->pdhg_dual_solution));
    CUSPARSE_CHECK(
        cusparseDnVecSetValues(state->vec_dual_prod, state->dual_product));

    CUSPARSE_CHECK(cusparseSpMV(
        state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE,
        state->matAt, state->vec_dual_sol, &HOST_ZERO, state->vec_dual_prod,
        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->dual_spmv_buffer));

    dual_feasibility_iterate_kernel<<<state->num_blocks_primal,
                                      THREADS_PER_BLOCK>>>(
        lambda, lambda_prev, lambda_bar, grad_lambda, state->step_size / 2,
        state->dual_product, state->objective_vector, state->dual_slack,
        state->num_variables);
    
    double lambda_dot;
    CUBLAS_CHECK(cublasDdot(
        state->blas_handle, state->num_variables, grad_lambda, 1, lambda_moment, 1, &lambda_dot));
    *t_prev = lambda_dot < 0 ? 1.0 : *t_curr;
}