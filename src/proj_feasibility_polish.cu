#include "feasibility_polish.h"
#include "utils.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ALLOC_ZERO(dest, bytes)                                                                                        \
    CUDA_CHECK(cudaMalloc(&dest, bytes));                                                                              \
    CUDA_CHECK(cudaMemset(dest, 0, bytes));

__global__ void update_t_and_beta_kernel(double *t_curr, double *t_prev, double *beta)
{
    *t_curr = 0.5 * (1.0 + sqrt(1.0 + 4.0 * (*t_prev) * (*t_prev)));
    *beta = (*t_prev - 1.0) / (*t_curr);
}

/* Strongly-convex AGD: fixed beta = (1-sqrt(mu/L)) / (1+sqrt(mu/L)) — bypasses the
 * smooth-convex t-update entirely. Called AFTER update_t_and_beta_kernel to
 * overwrite beta. t_curr/t_prev still advance so the existing gradient-restart
 * logic continues to work. */
__global__ void overwrite_beta_sc_kernel(double *beta, double beta_sc)
{
    *beta = beta_sc;
}

/* Process-global toggle / cached beta_sc, set by primal_polishing once per polish
 * call based on SCHEME1_SC_BETA env var. */
static bool g_use_sc = false;
static double g_beta_sc = 0.0;

/* Split warm-start y into (d1, d2): d1 = max(y, 0), d2 = max(-y, 0).
 * d1 is the multiplier for AL ≤ Ax (y>0 means lower bound active);
 * d2 is the multiplier for Ax ≤ AU (y<0 means upper bound active). */
__global__ void warm_start_d_from_y_kernel(const double *__restrict__ y,
                                            double *d1, double *d2,
                                            const double *__restrict__ con_lb,
                                            const double *__restrict__ con_ub,
                                            int num_con)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_con) return;
    double y_i = y[idx];
    /* Only initialize multipliers for sides with finite bounds. */
    d1[idx] = isfinite(con_lb[idx]) ? fmax(y_i, 0.0) : 0.0;
    d2[idx] = isfinite(con_ub[idx]) ? fmax(-y_i, 0.0) : 0.0;
}

/* Init d from x_warm directly via constraint violation:
 *   d1_init = max(0, AL - Ax_warm)  (positive iff lower bound violated)
 *   d2_init = max(0, Ax_warm - AU)  (positive iff upper bound violated)
 * No dependence on warm dual y — pure function of x_warm. Assumes Pock-Chambolle
 * rescaling makes row norms ||a_i|| ~ O(1); Jacobi preconditioning would divide
 * by ||a_i||^2 but that requires precomputing row norms (skipped for v1).
 */
__global__ void warm_start_d_from_x_kernel(const double *__restrict__ Ax_warm,
                                            double *d1, double *d2,
                                            const double *__restrict__ con_lb,
                                            const double *__restrict__ con_ub,
                                            int num_con)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_con) return;
    double Ax_i = Ax_warm[idx], lb = con_lb[idx], ub = con_ub[idx];
    d1[idx] = isfinite(lb) ? fmax(0.0, lb - Ax_i) : 0.0;
    d2[idx] = isfinite(ub) ? fmax(0.0, Ax_i - ub) : 0.0;
}

__global__ void update_t_prev_primal_kernel(double *t_curr, double *t_prev, double *d1_dot, double *d2_dot)
{
    *t_prev = (*d1_dot + *d2_dot < 0.0) ? 1.0 : *t_curr;
}

__global__ void update_t_prev_dual_kernel(double *t_curr, double *t_prev, double *lambda_dot)
{
    *t_prev = (*lambda_dot < 0.0) ? 1.0 : *t_curr;
}

__global__ void primal_feasibility_update_moment(
    double *d1_bar, double *d2_bar, double *d1_moment, double *d2_moment,
    double *delta_d, const double *__restrict__ d1, const double *__restrict__ d2,
    const double *__restrict__ d1_prev, const double *__restrict__ d2_prev,
    const double *beta_ptr, int num_con)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_con) return;
    
    double beta = *beta_ptr; // Read from device pointer
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


__global__ void dual_feasibility_update_moment(
    double *lambda_bar, double *lambda_moment, double *lambda,
    double *lambda_prev, const double *beta_ptr, int num_var)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_var) return;
    
    double beta = *beta_ptr; // Read from device pointer
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


static double get_step_factor(void)
{
    /* Step is `step_size^2 * STEP_FACTOR`. Default 0.5 = safe 1/(2L). 1.0 = 1/L. */
    static double cached = -1.0;
    if (cached < 0.0) {
        const char *s = getenv("SCHEME1_STEP_FACTOR");
        cached = (s && *s) ? atof(s) : 0.5;
        if (!isfinite(cached) || cached <= 0.0) cached = 0.5;
    }
    return cached;
}

void primal_feasibility_iterate_once(
    pdhg_solver_state_t *state,
    double *d1, double *d2, double *d1_prev, double *d2_prev, double *d1_bar,
    double *d2_bar, double *d1_moment, double *d2_moment, double *grad_d1,
    double *grad_d2, double *delta_d, double *direction,
    double *d_t_curr, double *d_t_prev, double *d_beta, double *d_d1_dot, double *d_d2_dot)
{
    double step = state->step_size * state->step_size * get_step_factor();

    update_t_and_beta_kernel<<<1, 1, 0, state->stream>>>(d_t_curr, d_t_prev, d_beta);
    if (g_use_sc) {
        overwrite_beta_sc_kernel<<<1, 1, 0, state->stream>>>(d_beta, g_beta_sc);
    }

    primal_feasibility_update_moment<<<state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream>>>(
        d1_bar, d2_bar, d1_moment, d2_moment, delta_d, d1, d2, d1_prev, d2_prev,
        d_beta, state->num_constraints);

    cupdlpx_spmv_ATx(state->sparse_handle, state->spmv_ctx, delta_d, direction);

    proj_grad_update_primal_solution<<<state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream>>>(
        state->pdhg_primal_solution, state->initial_primal_solution, direction,
        state->variable_lower_bound, state->variable_upper_bound, state->num_variables);

    cupdlpx_spmv_Ax(state->sparse_handle, state->spmv_ctx, state->pdhg_primal_solution, state->primal_product);

    primal_feasibility_iterate_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream>>>(
        d1, d2, d1_prev, d2_prev, grad_d1, grad_d2, step, d1_bar,
        d2_bar, state->primal_product, state->constraint_lower_bound,
        state->constraint_upper_bound, state->num_constraints);

    CUBLAS_CHECK(cublasDdot(state->blas_handle, state->num_constraints, grad_d1, 1, d1_moment, 1, d_d1_dot));
    CUBLAS_CHECK(cublasDdot(state->blas_handle, state->num_constraints, grad_d2, 1, d2_moment, 1, d_d2_dot));

    update_t_prev_primal_kernel<<<1, 1, 0, state->stream>>>(d_t_curr, d_t_prev, d_d1_dot, d_d2_dot);
}

/* helper: enable SC mode based on env at start of polish */
static void maybe_enable_sc(double step_size)
{
    const char *s = getenv("SCHEME1_SC_BETA");
    g_use_sc = false;
    g_beta_sc = 0.0;
    if (!s || !*s) return;
    /* Allow either an explicit float ("0.9") or "auto" => derive from step_size. */
    if (strcmp(s, "auto") == 0) {
        /* mu=1, L = 2/step^2. sqrt(mu/L) = step_size / sqrt(2). */
        double sqrt_mu_over_L = step_size / sqrt(2.0);
        g_beta_sc = (1.0 - sqrt_mu_over_L) / (1.0 + sqrt_mu_over_L);
    } else {
        g_beta_sc = atof(s);
    }
    if (g_beta_sc <= 0.0 || g_beta_sc >= 1.0) { g_beta_sc = 0.0; return; }
    g_use_sc = true;
    fprintf(stderr, "[SC] strongly-convex AGD enabled, beta_sc=%.6f\n", g_beta_sc);
}

void dual_feasibility_iterate_once(
    pdhg_solver_state_t *state,
    double *lambda, double *lambda_prev, double *lambda_bar,
    double *lambda_moment, double *grad_lambda, double *direction_y, double *original_dual_slack,
    double *d_t_curr, double *d_t_prev, double *d_beta, double *d_lambda_dot)
{
    update_t_and_beta_kernel<<<1, 1, 0, state->stream>>>(d_t_curr, d_t_prev, d_beta);
    if (g_use_sc) {
        overwrite_beta_sc_kernel<<<1, 1, 0, state->stream>>>(d_beta, g_beta_sc);
    }

    dual_feasibility_update_moment<<<state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream>>>(
        lambda_bar, lambda_moment, lambda, lambda_prev, d_beta, state->num_variables);

    cupdlpx_spmv_Ax(state->sparse_handle, state->spmv_ctx, lambda_bar, direction_y);

    proj_grad_update_dual_solution<<<state->num_blocks_primal_dual, THREADS_PER_BLOCK, 0, state->stream>>>(
        state->pdhg_dual_solution, state->dual_slack, state->initial_dual_solution, original_dual_slack, direction_y, lambda_bar,
        state->constraint_lower_bound, state->constraint_upper_bound, state->variable_lower_bound, state->variable_upper_bound,
        state->num_constraints, state->num_variables);

    cupdlpx_spmv_ATx(state->sparse_handle, state->spmv_ctx, state->pdhg_dual_solution, state->dual_product);

    dual_feasibility_iterate_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream>>>(
        lambda, lambda_prev, lambda_bar, grad_lambda, state->step_size * state->step_size * get_step_factor(),
        state->dual_product, state->objective_vector, state->dual_slack,
        state->num_variables);

    CUBLAS_CHECK(cublasDdot(state->blas_handle, state->num_variables, grad_lambda, 1, lambda_moment, 1, d_lambda_dot));
    update_t_prev_dual_kernel<<<1, 1, 0, state->stream>>>(d_t_curr, d_t_prev, d_lambda_dot);
}

feas_polish_result_t proj_scheme_primal_feasibility_polishing(
    pdhg_solver_state_t *state,
    const pdhg_parameters_t *params)
{
    maybe_enable_sc(state->step_size);
    int num_con = state->num_constraints;
    int num_var = state->num_variables;
    
    feas_polish_result_t result = {false, 0, 0.0, TERMINATION_REASON_UNSPECIFIED};

    // --- 0. Backup Original State Data ---
    double *backup_primal_solution;
    ALLOC_ZERO(backup_primal_solution, num_var * sizeof(double));
    CUDA_CHECK(cudaMemcpy(backup_primal_solution, state->pdhg_primal_solution, num_var * sizeof(double), cudaMemcpyDeviceToDevice));
    
    double backup_abs_residual = state->absolute_primal_residual;
    double backup_rel_residual = state->relative_primal_residual;
    double backup_obj_value = state->primal_objective_value;


    // --- 1. Array Allocations ---
    double *d1, *d2, *d1_prev, *d2_prev, *d1_bar, *d2_bar, *grad_d1, *grad_d2, *d1_moment, *d2_moment, *delta_d, *direction;
    ALLOC_ZERO(d1, num_con * sizeof(double));
    ALLOC_ZERO(d2, num_con * sizeof(double));
    ALLOC_ZERO(d1_prev, num_con * sizeof(double));
    ALLOC_ZERO(d2_prev, num_con * sizeof(double));
    ALLOC_ZERO(d1_bar, num_con * sizeof(double));
    ALLOC_ZERO(d2_bar, num_con * sizeof(double));
    ALLOC_ZERO(grad_d1, num_con * sizeof(double));
    ALLOC_ZERO(grad_d2, num_con * sizeof(double));
    ALLOC_ZERO(d1_moment, num_con * sizeof(double));
    ALLOC_ZERO(d2_moment, num_con * sizeof(double));
    ALLOC_ZERO(delta_d, num_con * sizeof(double));
    ALLOC_ZERO(direction, num_var * sizeof(double));

    // --- 2. Device Scalars for Graph Capture ---
    double *d_t_curr, *d_t_prev, *d_beta, *d_d1_dot, *d_d2_dot;
    ALLOC_ZERO(d_t_curr, sizeof(double));
    ALLOC_ZERO(d_t_prev, sizeof(double));
    ALLOC_ZERO(d_beta, sizeof(double));
    ALLOC_ZERO(d_d1_dot, sizeof(double));
    ALLOC_ZERO(d_d2_dot, sizeof(double));

    double init_t = 1.0;
    CUDA_CHECK(cudaMemcpy(d_t_curr, &init_t, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_t_prev, &init_t, sizeof(double), cudaMemcpyHostToDevice));

    // --- 3. State Initialization ---
    state->total_count = 0;
    state->cumulative_time_sec = 0.0;
    state->termination_reason = TERMINATION_REASON_UNSPECIFIED;
    
    CUDA_CHECK(cudaMemcpy(
        state->initial_primal_solution, state->pdhg_primal_solution,
        state->num_variables * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(
        state->current_primal_solution, state->pdhg_primal_solution,
        state->num_variables * sizeof(double), cudaMemcpyDeviceToDevice));
        
    /* Warm-start d1, d2. Modes (env SCHEME1_WARM_D):
     *   "" / unset / "0": no warm-start, d1=d2=0 (default)
     *   "y": use LP warm-start dual y splitted into (max(y,0), max(-y,0))
     *   "x": use Ax_warm constraint violation as d (default ON when no env set)
     * The "x" mode is the KKT-consistent choice for the projection problem:
     * d_init reflects how far each constraint is from feasibility on x_warm. */
    {
        const char *mode = getenv("SCHEME1_WARM_D");
        if (!mode) mode = "x";  /* default to x-based warm start */
        if (strcmp(mode, "y") == 0) {
            warm_start_d_from_y_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream>>>(
                state->pdhg_dual_solution, d1, d2,
                state->constraint_lower_bound, state->constraint_upper_bound,
                state->num_constraints);
        } else if (strcmp(mode, "x") == 0) {
            /* Need Ax_warm: warm x is in state->pdhg_primal_solution, compute A*x. */
            cupdlpx_spmv_Ax(state->sparse_handle, state->spmv_ctx,
                            state->pdhg_primal_solution, state->primal_product);
            warm_start_d_from_x_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream>>>(
                state->primal_product, d1, d2,
                state->constraint_lower_bound, state->constraint_upper_bound,
                state->num_constraints);
        }
        /* else: leave d1=d2=0 (already zero from ALLOC_ZERO). */
        if (strcmp(mode, "y") == 0 || strcmp(mode, "x") == 0) {
            CUDA_CHECK(cudaMemcpyAsync(d1_prev, d1, num_con * sizeof(double), cudaMemcpyDeviceToDevice, state->stream));
            CUDA_CHECK(cudaMemcpyAsync(d2_prev, d2, num_con * sizeof(double), cudaMemcpyDeviceToDevice, state->stream));
        }
    }

    print_initial_feas_polish_info(true, params);
    clock_t start_time = clock();

    // --- 4. Set cuBLAS to DEVICE mode ---
    cublasSetPointerMode(state->blas_handle, CUBLAS_POINTER_MODE_DEVICE);

    cudaGraphExec_t graphExec = NULL;
    bool graph_created = false;

    // --- 5. Main Iteration Loop ---
    while (state->termination_reason == TERMINATION_REASON_UNSPECIFIED)
    {
        if ((state->is_this_major_iteration || state->total_count == 0) || (state->total_count % get_print_frequency(state->total_count) == 0))
        {
            compute_primal_feas_polish_residual(state, state->objective_vector, params->optimality_norm);
            check_feas_polishing_termination_criteria(state, state, &params->termination_criteria, true);
            state->cumulative_time_sec = (double)(clock() - start_time) / CLOCKS_PER_SEC;
            display_feas_polish_iteration_stats(state, params->verbose, true);
            
            if (state->termination_reason != TERMINATION_REASON_UNSPECIFIED) break;
        }

        if (!graph_created)
        {
            CUDA_CHECK(cudaStreamBeginCapture(state->stream, cudaStreamCaptureModeGlobal));
            
            for (int i = 0; i < params->termination_evaluation_frequency; i++)
            {
                primal_feasibility_iterate_once(
                    state, d1, d2, d1_prev, d2_prev, d1_bar, d2_bar, d1_moment, d2_moment, 
                    grad_d1, grad_d2, delta_d, direction, 
                    d_t_curr, d_t_prev, d_beta, d_d1_dot, d_d2_dot);
            }
            
            cudaGraph_t graph;
            CUDA_CHECK(cudaStreamEndCapture(state->stream, &graph));
            CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
            CUDA_CHECK(cudaGraphDestroy(graph));
            graph_created = true;
        }
        
        CUDA_CHECK(cudaGraphLaunch(graphExec, state->stream));
        state->total_count += params->termination_evaluation_frequency;
    }

    // --- 6. Restore cuBLAS to HOST mode ---
    cublasSetPointerMode(state->blas_handle, CUBLAS_POINTER_MODE_HOST);

    // --- 7. Handle Results & Fallback ---
    result.iterations = state->total_count;
    result.time_sec = state->cumulative_time_sec;
    result.termination_reason = state->termination_reason;

    /* Keep the polished iterate on SUCCESS / ITER_LIMIT / TIME_LIMIT — only
     * roll back to the warm-start backup if the polish blew up (e.g., flagged
     * INFEASIBLE / numerical error). The previous behavior rolled back on any
     * non-SUCCESS termination, which made budget-limited runs indistinguishable
     * from "no polish" and unfair to compare against baseline. */
    if (state->termination_reason == TERMINATION_REASON_FEAS_POLISH_SUCCESS ||
        state->termination_reason == TERMINATION_REASON_ITERATION_LIMIT ||
        state->termination_reason == TERMINATION_REASON_TIME_LIMIT)
    {
        result.success = (state->termination_reason == TERMINATION_REASON_FEAS_POLISH_SUCCESS);
        CUDA_CHECK(cudaMemcpy(
            state->current_primal_solution, state->pdhg_primal_solution,
            state->num_variables * sizeof(double), cudaMemcpyDeviceToDevice));
    }
    else
    {
        result.success = false;
        CUDA_CHECK(cudaMemcpy(
            state->pdhg_primal_solution, backup_primal_solution,
            state->num_variables * sizeof(double), cudaMemcpyDeviceToDevice));
        state->absolute_primal_residual = backup_abs_residual;
        state->relative_primal_residual = backup_rel_residual;
        state->primal_objective_value = backup_obj_value;
    }

    // --- 8. Cleanup Memory ---
    if (graphExec) CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK(cudaFree(backup_primal_solution)); // Free the backup buffer
    CUDA_CHECK(cudaFree(d1));
    CUDA_CHECK(cudaFree(d2));
    CUDA_CHECK(cudaFree(d1_prev));
    CUDA_CHECK(cudaFree(d2_prev));
    CUDA_CHECK(cudaFree(d1_bar));
    CUDA_CHECK(cudaFree(d2_bar));
    CUDA_CHECK(cudaFree(grad_d1));
    CUDA_CHECK(cudaFree(grad_d2));
    CUDA_CHECK(cudaFree(d1_moment));
    CUDA_CHECK(cudaFree(d2_moment));
    CUDA_CHECK(cudaFree(delta_d));
    CUDA_CHECK(cudaFree(direction));
    CUDA_CHECK(cudaFree(d_t_curr));
    CUDA_CHECK(cudaFree(d_t_prev));
    CUDA_CHECK(cudaFree(d_beta));
    CUDA_CHECK(cudaFree(d_d1_dot));
    CUDA_CHECK(cudaFree(d_d2_dot));
    
    return result;
}


feas_polish_result_t proj_scheme_dual_feasibility_polishing(
    pdhg_solver_state_t *state,
    const pdhg_parameters_t *params,
    const double *warm_primal_x)
{
    int num_con = state->num_constraints;
    int num_var = state->num_variables;
    
    feas_polish_result_t result = {false, 0, 0.0, TERMINATION_REASON_UNSPECIFIED};

    // --- 0. Backup Original State Data ---
    double *backup_dual_solution;
    double *backup_dual_slack;
    ALLOC_ZERO(backup_dual_solution, num_con * sizeof(double));
    ALLOC_ZERO(backup_dual_slack, num_var * sizeof(double));
    
    CUDA_CHECK(cudaMemcpy(backup_dual_solution, state->pdhg_dual_solution, num_con * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(backup_dual_slack, state->dual_slack, num_var * sizeof(double), cudaMemcpyDeviceToDevice));
    
    double backup_abs_residual = state->absolute_dual_residual;
    double backup_rel_residual = state->relative_dual_residual;
    double backup_obj_value = state->dual_objective_value;

    // --- 1. Array Allocations ---
    double *lambda, *lambda_prev, *lambda_bar, *grad_lambda, *lambda_moment, *direction_y, *original_dual_slack;
    ALLOC_ZERO(lambda, num_var * sizeof(double));
    ALLOC_ZERO(lambda_prev, num_var * sizeof(double));
    ALLOC_ZERO(lambda_bar, num_var * sizeof(double));
    ALLOC_ZERO(grad_lambda, num_var * sizeof(double));
    ALLOC_ZERO(lambda_moment, num_var * sizeof(double));
    ALLOC_ZERO(original_dual_slack, num_var * sizeof(double));
    ALLOC_ZERO(direction_y, num_con * sizeof(double));

    // --- 2. Device Scalars for Graph Capture ---
    double *d_t_curr, *d_t_prev, *d_beta, *d_lambda_dot;
    ALLOC_ZERO(d_t_curr, sizeof(double));
    ALLOC_ZERO(d_t_prev, sizeof(double));
    ALLOC_ZERO(d_beta, sizeof(double));
    ALLOC_ZERO(d_lambda_dot, sizeof(double));

    double init_t = 1.0;
    CUDA_CHECK(cudaMemcpy(d_t_curr, &init_t, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_t_prev, &init_t, sizeof(double), cudaMemcpyHostToDevice));

    // --- 3. State Initialization ---
    state->total_count = 0;
    state->cumulative_time_sec = 0.0;
    state->termination_reason = TERMINATION_REASON_UNSPECIFIED;
    
    CUDA_CHECK(cudaMemcpy(
        state->initial_dual_solution, state->pdhg_dual_solution,
        state->num_constraints * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(
        state->current_dual_solution, state->pdhg_dual_solution,
        state->num_constraints * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(
        original_dual_slack, state->dual_slack,
        state->num_variables * sizeof(double), cudaMemcpyDeviceToDevice));
        
    print_initial_feas_polish_info(false, params);
    clock_t start_time = clock();

    // --- 4. Set cuBLAS to DEVICE mode ---
    cublasSetPointerMode(state->blas_handle, CUBLAS_POINTER_MODE_DEVICE);

    cudaGraphExec_t graphExec = NULL;
    bool graph_created = false;

    // --- 5. Main Iteration Loop ---
    while (state->termination_reason == TERMINATION_REASON_UNSPECIFIED)
    {
        if ((state->is_this_major_iteration || state->total_count == 0) || (state->total_count % get_print_frequency(state->total_count) == 0))
        {
            /* Use the WARM-START primal for the s^T x term in the dual objective.
             * This evaluates the dual lower bound consistently across schemes
             * regardless of how each scheme polished the primal — eliminating
             * the x-shift artifact that would otherwise pollute the reported
             * dual_obj when a scheme moves x significantly during polish. */
            compute_dual_feas_polish_residual(state,
                                              state->constraint_lower_bound_finite_val,
                                              state->constraint_upper_bound_finite_val,
                                              warm_primal_x,
                                              params->optimality_norm);
                                              
            check_feas_polishing_termination_criteria(state, state, &params->termination_criteria, false);
            state->cumulative_time_sec = (double)(clock() - start_time) / CLOCKS_PER_SEC;
            display_feas_polish_iteration_stats(state, params->verbose, false);
            
            if (state->termination_reason != TERMINATION_REASON_UNSPECIFIED) break;
        }

        if (!graph_created)
        {
            CUDA_CHECK(cudaStreamBeginCapture(state->stream, cudaStreamCaptureModeGlobal));
            
            for (int i = 0; i < params->termination_evaluation_frequency; i++)
            {
                dual_feasibility_iterate_once(
                    state, lambda, lambda_prev, lambda_bar, lambda_moment, 
                    grad_lambda, direction_y, original_dual_slack,
                    d_t_curr, d_t_prev, d_beta, d_lambda_dot);
            }
            
            cudaGraph_t graph;
            CUDA_CHECK(cudaStreamEndCapture(state->stream, &graph));
            CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
            CUDA_CHECK(cudaGraphDestroy(graph));
            graph_created = true;
        }
        
        CUDA_CHECK(cudaGraphLaunch(graphExec, state->stream));
        state->total_count += params->termination_evaluation_frequency;
    }

    // --- 6. Restore cuBLAS to HOST mode ---
    cublasSetPointerMode(state->blas_handle, CUBLAS_POINTER_MODE_HOST);

    // --- 7. Handle Results & Fallback ---
    result.iterations = state->total_count;
    result.time_sec = state->cumulative_time_sec;
    result.termination_reason = state->termination_reason;

    /* Same policy as primal: keep polished on SUCCESS/ITER_LIMIT/TIME_LIMIT. */
    if (state->termination_reason == TERMINATION_REASON_FEAS_POLISH_SUCCESS ||
        state->termination_reason == TERMINATION_REASON_ITERATION_LIMIT ||
        state->termination_reason == TERMINATION_REASON_TIME_LIMIT)
    {
        result.success = (state->termination_reason == TERMINATION_REASON_FEAS_POLISH_SUCCESS);
    }
    else
    {
        result.success = false;
        CUDA_CHECK(cudaMemcpy(
            state->pdhg_dual_solution, backup_dual_solution,
            state->num_constraints * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(
            state->dual_slack, backup_dual_slack,
            state->num_variables * sizeof(double), cudaMemcpyDeviceToDevice));
        state->absolute_dual_residual = backup_abs_residual;
        state->relative_dual_residual = backup_rel_residual;
        state->dual_objective_value = backup_obj_value;
    }

    // --- 8. Cleanup Memory ---
    if (graphExec) CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK(cudaFree(backup_dual_solution));
    CUDA_CHECK(cudaFree(backup_dual_slack));
    CUDA_CHECK(cudaFree(lambda));
    CUDA_CHECK(cudaFree(lambda_prev));
    CUDA_CHECK(cudaFree(lambda_bar));
    CUDA_CHECK(cudaFree(grad_lambda));
    CUDA_CHECK(cudaFree(lambda_moment));
    CUDA_CHECK(cudaFree(original_dual_slack));
    CUDA_CHECK(cudaFree(direction_y));
    CUDA_CHECK(cudaFree(d_t_curr));
    CUDA_CHECK(cudaFree(d_t_prev));
    CUDA_CHECK(cudaFree(d_beta));
    CUDA_CHECK(cudaFree(d_lambda_dot));
    
    return result;
}

void proj_scheme_feas_polish_final_log(const feas_polish_result_t *primal_res,
                                       const feas_polish_result_t *dual_res,
                                       const pdhg_solver_state_t *state,
                                       bool verbose)
{
    if (verbose)
    {
        printf("---------------------------------------------------------------------------------------\n");
    }
    printf("Feasibility Polishing Summary\n");
    printf("  Primal Status        : %s\n", termination_reason_to_string(primal_res->termination_reason));
    printf("  Primal Iterations    : %d\n", primal_res->iterations);
    printf("  Primal Time Usage    : %.3g sec\n", primal_res->time_sec);
    printf("  Dual Status          : %s\n", termination_reason_to_string(dual_res->termination_reason));
    printf("  Dual Iterations      : %d\n", dual_res->iterations);
    printf("  Dual Time Usage      : %.3g sec\n", dual_res->time_sec);
    printf("  Primal Residual      : %.3e\n", state->relative_primal_residual);
    printf("  Dual Residual        : %.3e\n", state->relative_dual_residual);
    printf("  Primal Dual Gap      : %.3e\n", 
           fabs(state->primal_objective_value - state->dual_objective_value) /
               (1.0 + fabs(state->primal_objective_value) + fabs(state->dual_objective_value)));
}

void proj_scheme_feasibility_polish(const pdhg_parameters_t *params, pdhg_solver_state_t *state)
{
    clock_t feasibility_polishing_start_time = clock();
    
    if (state->relative_primal_residual < params->termination_criteria.eps_feas_polish_relative &&
        state->relative_dual_residual < params->termination_criteria.eps_feas_polish_relative)
    {
        printf("Skipping feasibility polishing as the solution is already sufficiently feasible.\n");
        return;
    }

    /* Snapshot the warm-start primal x BEFORE primal polish overwrites it.
     * The dual phase uses this fixed reference x to compute s^T x in the
     * dual_obj formula, so the reported dual_obj is comparable across schemes
     * regardless of how each scheme's primal phase moved x. */
    double *warm_primal_x_backup;
    CUDA_CHECK(cudaMalloc(&warm_primal_x_backup, state->num_variables * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(warm_primal_x_backup, state->pdhg_primal_solution,
                          state->num_variables * sizeof(double), cudaMemcpyDeviceToDevice));

    feas_polish_result_t primal_res = proj_scheme_primal_feasibility_polishing(state, params);

    state->feasibility_iteration += primal_res.iterations;
    state->primal_polish_iterations = primal_res.iterations;
    state->primal_polish_time_sec = primal_res.time_sec;
    state->primal_polish_termination = primal_res.termination_reason;
    state->primal_polish_residual = state->relative_primal_residual;

    feas_polish_result_t dual_res = proj_scheme_dual_feasibility_polishing(state, params, warm_primal_x_backup);

    state->feasibility_iteration += dual_res.iterations;
    state->dual_polish_iterations = dual_res.iterations;
    state->dual_polish_time_sec = dual_res.time_sec;
    state->dual_polish_termination = dual_res.termination_reason;
    state->dual_polish_residual = state->relative_dual_residual;

    state->objective_gap = fabs(state->primal_objective_value - state->dual_objective_value);
    state->relative_objective_gap =
        state->objective_gap / (1.0 + fabs(state->primal_objective_value) + fabs(state->dual_objective_value));
    state->polish_relative_gap = state->relative_objective_gap;

    proj_scheme_feas_polish_final_log(&primal_res, &dual_res, state, params->verbose);

    CUDA_CHECK(cudaFree(warm_primal_x_backup));

    state->feasibility_polishing_time = (double)(clock() - feasibility_polishing_start_time) / CLOCKS_PER_SEC;
    return;
}