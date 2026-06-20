/*
 * Scheme 1 (Projection) + adaptive AGD restart.
 *
 * Reuses Scheme 1's per-iteration kernels (proj_feasibility_polish.cu).
 * Restart logic:
 *
 *   - Gradient-direction restart (kept from Scheme 1): t_prev = 1 when
 *     <grad, Δd> < 0.
 *   - Sufficient-reduction adaptive period restart (added here): every
 *     K_curr iters we measure the squared violation
 *         r^2 = ||(AL - Ax)_+||^2 + ||(Ax - AU)_+||^2
 *     and compare to r^2 at the previous restart point. If r^2 dropped by at
 *     least a factor beta_suff (sufficient progress) we *grow* K_curr; if it
 *     barely moved, we *shrink* K_curr. In both cases we force t_prev = 1 at
 *     the restart point. K_curr is clamped to [K_MIN, K_MAX].
 *
 * This auto-tunes the restart period per instance: well-conditioned dual-ascent
 * problems converge fast between restarts and K_curr grows to ~K_MAX (rare
 * forced restarts; behaves like Scheme 1); ill-conditioned problems with
 * AGD ringing get aggressive short restart intervals.
 */
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

#define PR_ALLOC_ZERO(dest, bytes)                                                                                     \
    CUDA_CHECK(cudaMalloc(&dest, bytes));                                                                              \
    CUDA_CHECK(cudaMemset(dest, 0, bytes));

#define PR_K_INIT 256
#define PR_K_MIN  16
#define PR_K_MAX  8192
#define PR_BETA_SUFF       0.9
#define PR_GROW_NUM        3       /* K *= 3/2 */
#define PR_GROW_DEN        2
#define PR_SHRINK_NUM      7       /* K *= 7/10 */
#define PR_SHRINK_DEN      10

/* Reuse Scheme 1 iterate primitives. */
extern void primal_feasibility_iterate_once(
    pdhg_solver_state_t *state,
    double *d1, double *d2, double *d1_prev, double *d2_prev, double *d1_bar,
    double *d2_bar, double *d1_moment, double *d2_moment, double *grad_d1,
    double *grad_d2, double *delta_d, double *direction,
    double *d_t_curr, double *d_t_prev, double *d_beta, double *d_d1_dot, double *d_d2_dot);

extern void dual_feasibility_iterate_once(
    pdhg_solver_state_t *state,
    double *lambda, double *lambda_prev, double *lambda_bar,
    double *lambda_moment, double *grad_lambda, double *direction_y,
    double *original_dual_slack,
    double *d_t_curr, double *d_t_prev, double *d_beta, double *d_lambda_dot);

/* Compute per-element squared violation: viol_sq[i] = max(0, AL-Ax)^2 + max(0, Ax-AU)^2.
 * The kernels in proj_feasibility_polish.cu already wrote grad_d1 = (AL-Ax) * is_AL_fin
 * and grad_d2 = (Ax-AU) * is_AU_fin; the positive parts of these are exactly the
 * primal residual contributions at unbounded sides we get 0 from the mask.
 */
__global__ void pr_violation_sq_kernel(double *viol_sq,
                                       const double *__restrict__ grad_d1,
                                       const double *__restrict__ grad_d2,
                                       int num_con)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_con) return;
    double v1 = fmax(0.0, grad_d1[idx]);
    double v2 = fmax(0.0, grad_d2[idx]);
    viol_sq[idx] = v1 * v1 + v2 * v2;
}

/* Same for the dual phase: lambda gradient violation.
 * grad_lambda = c - A^T y - s. Magnitude squared per element. */
__global__ void pr_lambda_sq_kernel(double *out_sq,
                                    const double *__restrict__ grad_lambda,
                                    int num_var)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_var) return;
    double g = grad_lambda[idx];
    out_sq[idx] = g * g;
}

/* Adaptive restart: increment iter_since_restart; if it hit K, compare r_curr to r_last
 * and update K and t_prev. */
/* Optional c-shift: anchor <- x0 - alpha*c (same as proj-obj). When alpha=0,
 * proj-restart reduces to Scheme 1 + adaptive restart (no objective shift). */
__global__ void pr_subtract_obj_kernel(double *x, const double *__restrict__ c, double alpha, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    x[idx] -= alpha * c[idx];
}

static double pr_get_alpha(double step_size)
{
    const char *s = getenv("PROJ_OBJ_ALPHA");
    if (!s || !*s) return 0.0;
    if (strcmp(s, "step2") == 0) return step_size * step_size;
    if (strcmp(s, "step")  == 0) return step_size;
    double v = atof(s);
    if (v == 0.0 && s[0] != '0') return 0.0;
    return v;
}

__global__ void pr_adaptive_restart_kernel(double *t_prev,
                                           int *d_K,
                                           int *d_iter_since_restart,
                                           double *d_r_last,
                                           const double *d_r_curr,
                                           int K_min, int K_max,
                                           double beta_suff,
                                           int grow_num, int grow_den,
                                           int shrink_num, int shrink_den)
{
    int it = ++(*d_iter_since_restart);
    if (it < *d_K) return;

    double r_curr = *d_r_curr;
    double r_last = *d_r_last;
    int K = *d_K;
    if (r_curr <= beta_suff * r_last) {
        K = (K * grow_num) / grow_den;
        if (K > K_max) K = K_max;
    } else {
        K = (K * shrink_num) / shrink_den;
        if (K < K_min) K = K_min;
    }
    *d_K = K;
    *d_r_last = r_curr;
    *d_iter_since_restart = 0;
    *t_prev = 1.0;
}

/* -----------------------------------------------------------------------
 * Primal driver
 * --------------------------------------------------------------------- */
static feas_polish_result_t proj_restart_primal_polishing(pdhg_solver_state_t *state,
                                                           const pdhg_parameters_t *params)
{
    int num_con = state->num_constraints;
    int num_var = state->num_variables;
    feas_polish_result_t result = {false, 0, 0.0, TERMINATION_REASON_UNSPECIFIED};

    /* Backup. */
    double *backup_primal_solution;
    PR_ALLOC_ZERO(backup_primal_solution, num_var * sizeof(double));
    CUDA_CHECK(cudaMemcpy(backup_primal_solution, state->pdhg_primal_solution,
                          num_var * sizeof(double), cudaMemcpyDeviceToDevice));
    double backup_abs_residual = state->absolute_primal_residual;
    double backup_rel_residual = state->relative_primal_residual;
    double backup_obj_value    = state->primal_objective_value;

    double *d1, *d2, *d1_prev, *d2_prev, *d1_bar, *d2_bar, *grad_d1, *grad_d2;
    double *d1_moment, *d2_moment, *delta_d, *direction;
    PR_ALLOC_ZERO(d1,        num_con * sizeof(double));
    PR_ALLOC_ZERO(d2,        num_con * sizeof(double));
    PR_ALLOC_ZERO(d1_prev,   num_con * sizeof(double));
    PR_ALLOC_ZERO(d2_prev,   num_con * sizeof(double));
    PR_ALLOC_ZERO(d1_bar,    num_con * sizeof(double));
    PR_ALLOC_ZERO(d2_bar,    num_con * sizeof(double));
    PR_ALLOC_ZERO(grad_d1,   num_con * sizeof(double));
    PR_ALLOC_ZERO(grad_d2,   num_con * sizeof(double));
    PR_ALLOC_ZERO(d1_moment, num_con * sizeof(double));
    PR_ALLOC_ZERO(d2_moment, num_con * sizeof(double));
    PR_ALLOC_ZERO(delta_d,   num_con * sizeof(double));
    PR_ALLOC_ZERO(direction, num_var * sizeof(double));

    double *d_t_curr, *d_t_prev, *d_beta, *d_d1_dot, *d_d2_dot;
    PR_ALLOC_ZERO(d_t_curr, sizeof(double));
    PR_ALLOC_ZERO(d_t_prev, sizeof(double));
    PR_ALLOC_ZERO(d_beta,   sizeof(double));
    PR_ALLOC_ZERO(d_d1_dot, sizeof(double));
    PR_ALLOC_ZERO(d_d2_dot, sizeof(double));

    /* Adaptive-restart state. */
    double *viol_sq, *d_r_curr, *d_r_last;
    int *d_K, *d_iter_since_restart;
    PR_ALLOC_ZERO(viol_sq,              num_con * sizeof(double));
    PR_ALLOC_ZERO(d_r_curr,             sizeof(double));
    PR_ALLOC_ZERO(d_r_last,             sizeof(double));
    PR_ALLOC_ZERO(d_K,                  sizeof(int));
    PR_ALLOC_ZERO(d_iter_since_restart, sizeof(int));
    int K_init = PR_K_INIT;
    double r_init = INFINITY;
    CUDA_CHECK(cudaMemcpy(d_K,      &K_init, sizeof(int),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_r_last, &r_init, sizeof(double), cudaMemcpyHostToDevice));

    double init_t = 1.0;
    CUDA_CHECK(cudaMemcpy(d_t_curr, &init_t, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_t_prev, &init_t, sizeof(double), cudaMemcpyHostToDevice));

    state->total_count         = 0;
    state->cumulative_time_sec = 0.0;
    state->termination_reason  = TERMINATION_REASON_UNSPECIFIED;

    CUDA_CHECK(cudaMemcpy(state->initial_primal_solution, state->pdhg_primal_solution,
                          num_var * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state->current_primal_solution, state->pdhg_primal_solution,
                          num_var * sizeof(double), cudaMemcpyDeviceToDevice));

    /* Optional proj-obj-style c-shift: anchor <- x0 - alpha*c when PROJ_OBJ_ALPHA env is set. */
    {
        double alpha = pr_get_alpha(state->step_size);
        if (alpha != 0.0) {
            int blocks_primal = (num_var + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            pr_subtract_obj_kernel<<<blocks_primal, THREADS_PER_BLOCK, 0, state->stream>>>(
                state->initial_primal_solution, state->objective_vector, alpha, num_var);
        }
    }

    print_initial_feas_polish_info(true, params);
    clock_t start_time = clock();

    cublasSetPointerMode(state->blas_handle, CUBLAS_POINTER_MODE_DEVICE);

    cudaGraphExec_t graphExec = NULL;
    bool graph_created = false;

    while (state->termination_reason == TERMINATION_REASON_UNSPECIFIED) {
        if ((state->is_this_major_iteration || state->total_count == 0) ||
            (state->total_count % get_print_frequency(state->total_count) == 0)) {
            compute_primal_feas_polish_residual(state, state->objective_vector, params->optimality_norm);
            check_feas_polishing_termination_criteria(state, state, &params->termination_criteria, true);
            state->cumulative_time_sec = (double)(clock() - start_time) / CLOCKS_PER_SEC;
            display_feas_polish_iteration_stats(state, params->verbose, true);
            if (state->termination_reason != TERMINATION_REASON_UNSPECIFIED) break;
        }

        if (!graph_created) {
            CUDA_CHECK(cudaStreamBeginCapture(state->stream, cudaStreamCaptureModeGlobal));
            for (int i = 0; i < params->termination_evaluation_frequency; i++) {
                primal_feasibility_iterate_once(state, d1, d2, d1_prev, d2_prev,
                                                d1_bar, d2_bar, d1_moment, d2_moment,
                                                grad_d1, grad_d2, delta_d, direction,
                                                d_t_curr, d_t_prev, d_beta, d_d1_dot, d_d2_dot);
                /* Compute squared violation (only positive parts of grad_d1, grad_d2). */
                pr_violation_sq_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream>>>(
                    viol_sq, grad_d1, grad_d2, num_con);
                CUBLAS_CHECK(cublasDasum(state->blas_handle, num_con, viol_sq, 1, d_r_curr));
                /* Adaptive restart with sufficient-reduction rule. */
                pr_adaptive_restart_kernel<<<1, 1, 0, state->stream>>>(
                    d_t_prev, d_K, d_iter_since_restart, d_r_last, d_r_curr,
                    PR_K_MIN, PR_K_MAX, PR_BETA_SUFF,
                    PR_GROW_NUM, PR_GROW_DEN, PR_SHRINK_NUM, PR_SHRINK_DEN);
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

    cublasSetPointerMode(state->blas_handle, CUBLAS_POINTER_MODE_HOST);

    result.iterations         = state->total_count;
    result.time_sec           = state->cumulative_time_sec;
    result.termination_reason = state->termination_reason;

    if (state->termination_reason == TERMINATION_REASON_FEAS_POLISH_SUCCESS ||
        state->termination_reason == TERMINATION_REASON_ITERATION_LIMIT ||
        state->termination_reason == TERMINATION_REASON_TIME_LIMIT) {
        result.success = (state->termination_reason == TERMINATION_REASON_FEAS_POLISH_SUCCESS);
        CUDA_CHECK(cudaMemcpy(state->current_primal_solution, state->pdhg_primal_solution,
                              num_var * sizeof(double), cudaMemcpyDeviceToDevice));
    } else {
        result.success = false;
        CUDA_CHECK(cudaMemcpy(state->pdhg_primal_solution, backup_primal_solution,
                              num_var * sizeof(double), cudaMemcpyDeviceToDevice));
        state->absolute_primal_residual = backup_abs_residual;
        state->relative_primal_residual = backup_rel_residual;
        state->primal_objective_value   = backup_obj_value;
    }

    if (graphExec) CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK(cudaFree(backup_primal_solution));
    CUDA_CHECK(cudaFree(d1));         CUDA_CHECK(cudaFree(d2));
    CUDA_CHECK(cudaFree(d1_prev));    CUDA_CHECK(cudaFree(d2_prev));
    CUDA_CHECK(cudaFree(d1_bar));     CUDA_CHECK(cudaFree(d2_bar));
    CUDA_CHECK(cudaFree(grad_d1));    CUDA_CHECK(cudaFree(grad_d2));
    CUDA_CHECK(cudaFree(d1_moment));  CUDA_CHECK(cudaFree(d2_moment));
    CUDA_CHECK(cudaFree(delta_d));    CUDA_CHECK(cudaFree(direction));
    CUDA_CHECK(cudaFree(d_t_curr));   CUDA_CHECK(cudaFree(d_t_prev));
    CUDA_CHECK(cudaFree(d_beta));
    CUDA_CHECK(cudaFree(d_d1_dot));   CUDA_CHECK(cudaFree(d_d2_dot));
    CUDA_CHECK(cudaFree(viol_sq));
    CUDA_CHECK(cudaFree(d_r_curr));   CUDA_CHECK(cudaFree(d_r_last));
    CUDA_CHECK(cudaFree(d_K));        CUDA_CHECK(cudaFree(d_iter_since_restart));
    return result;
}

/* -----------------------------------------------------------------------
 * Dual driver
 * --------------------------------------------------------------------- */
static feas_polish_result_t proj_restart_dual_polishing(pdhg_solver_state_t *state,
                                                         const pdhg_parameters_t *params,
                                                         const double *warm_primal_x)
{
    int num_con = state->num_constraints;
    int num_var = state->num_variables;
    feas_polish_result_t result = {false, 0, 0.0, TERMINATION_REASON_UNSPECIFIED};

    double *backup_dual_solution, *backup_dual_slack;
    PR_ALLOC_ZERO(backup_dual_solution, num_con * sizeof(double));
    PR_ALLOC_ZERO(backup_dual_slack,    num_var * sizeof(double));
    CUDA_CHECK(cudaMemcpy(backup_dual_solution, state->pdhg_dual_solution,
                          num_con * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(backup_dual_slack, state->dual_slack,
                          num_var * sizeof(double), cudaMemcpyDeviceToDevice));
    double backup_abs_residual = state->absolute_dual_residual;
    double backup_rel_residual = state->relative_dual_residual;
    double backup_obj_value    = state->dual_objective_value;

    double *lambda, *lambda_prev, *lambda_bar, *grad_lambda, *lambda_moment;
    double *direction_y, *original_dual_slack;
    PR_ALLOC_ZERO(lambda,              num_var * sizeof(double));
    PR_ALLOC_ZERO(lambda_prev,         num_var * sizeof(double));
    PR_ALLOC_ZERO(lambda_bar,          num_var * sizeof(double));
    PR_ALLOC_ZERO(grad_lambda,         num_var * sizeof(double));
    PR_ALLOC_ZERO(lambda_moment,       num_var * sizeof(double));
    PR_ALLOC_ZERO(original_dual_slack, num_var * sizeof(double));
    PR_ALLOC_ZERO(direction_y,         num_con * sizeof(double));

    double *d_t_curr, *d_t_prev, *d_beta, *d_lambda_dot;
    PR_ALLOC_ZERO(d_t_curr,     sizeof(double));
    PR_ALLOC_ZERO(d_t_prev,     sizeof(double));
    PR_ALLOC_ZERO(d_beta,       sizeof(double));
    PR_ALLOC_ZERO(d_lambda_dot, sizeof(double));

    double *viol_sq, *d_r_curr, *d_r_last;
    int *d_K, *d_iter_since_restart;
    PR_ALLOC_ZERO(viol_sq,              num_var * sizeof(double));
    PR_ALLOC_ZERO(d_r_curr,             sizeof(double));
    PR_ALLOC_ZERO(d_r_last,             sizeof(double));
    PR_ALLOC_ZERO(d_K,                  sizeof(int));
    PR_ALLOC_ZERO(d_iter_since_restart, sizeof(int));
    int K_init = PR_K_INIT;
    double r_init = INFINITY;
    CUDA_CHECK(cudaMemcpy(d_K,      &K_init, sizeof(int),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_r_last, &r_init, sizeof(double), cudaMemcpyHostToDevice));

    double init_t = 1.0;
    CUDA_CHECK(cudaMemcpy(d_t_curr, &init_t, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_t_prev, &init_t, sizeof(double), cudaMemcpyHostToDevice));

    state->total_count         = 0;
    state->cumulative_time_sec = 0.0;
    state->termination_reason  = TERMINATION_REASON_UNSPECIFIED;

    CUDA_CHECK(cudaMemcpy(state->initial_dual_solution, state->pdhg_dual_solution,
                          num_con * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state->current_dual_solution, state->pdhg_dual_solution,
                          num_con * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(original_dual_slack, state->dual_slack,
                          num_var * sizeof(double), cudaMemcpyDeviceToDevice));

    print_initial_feas_polish_info(false, params);
    clock_t start_time = clock();

    cublasSetPointerMode(state->blas_handle, CUBLAS_POINTER_MODE_DEVICE);

    cudaGraphExec_t graphExec = NULL;
    bool graph_created = false;

    while (state->termination_reason == TERMINATION_REASON_UNSPECIFIED) {
        if ((state->is_this_major_iteration || state->total_count == 0) ||
            (state->total_count % get_print_frequency(state->total_count) == 0)) {
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

        if (!graph_created) {
            CUDA_CHECK(cudaStreamBeginCapture(state->stream, cudaStreamCaptureModeGlobal));
            for (int i = 0; i < params->termination_evaluation_frequency; i++) {
                dual_feasibility_iterate_once(state, lambda, lambda_prev, lambda_bar, lambda_moment,
                                              grad_lambda, direction_y, original_dual_slack,
                                              d_t_curr, d_t_prev, d_beta, d_lambda_dot);
                pr_lambda_sq_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream>>>(
                    viol_sq, grad_lambda, num_var);
                CUBLAS_CHECK(cublasDasum(state->blas_handle, num_var, viol_sq, 1, d_r_curr));
                pr_adaptive_restart_kernel<<<1, 1, 0, state->stream>>>(
                    d_t_prev, d_K, d_iter_since_restart, d_r_last, d_r_curr,
                    PR_K_MIN, PR_K_MAX, PR_BETA_SUFF,
                    PR_GROW_NUM, PR_GROW_DEN, PR_SHRINK_NUM, PR_SHRINK_DEN);
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

    cublasSetPointerMode(state->blas_handle, CUBLAS_POINTER_MODE_HOST);

    result.iterations         = state->total_count;
    result.time_sec           = state->cumulative_time_sec;
    result.termination_reason = state->termination_reason;

    if (state->termination_reason == TERMINATION_REASON_FEAS_POLISH_SUCCESS ||
        state->termination_reason == TERMINATION_REASON_ITERATION_LIMIT ||
        state->termination_reason == TERMINATION_REASON_TIME_LIMIT) {
        result.success = (state->termination_reason == TERMINATION_REASON_FEAS_POLISH_SUCCESS);
    } else {
        result.success = false;
        CUDA_CHECK(cudaMemcpy(state->pdhg_dual_solution, backup_dual_solution,
                              num_con * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(state->dual_slack, backup_dual_slack,
                              num_var * sizeof(double), cudaMemcpyDeviceToDevice));
        state->absolute_dual_residual = backup_abs_residual;
        state->relative_dual_residual = backup_rel_residual;
        state->dual_objective_value   = backup_obj_value;
    }

    if (graphExec) CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK(cudaFree(backup_dual_solution));
    CUDA_CHECK(cudaFree(backup_dual_slack));
    CUDA_CHECK(cudaFree(lambda));               CUDA_CHECK(cudaFree(lambda_prev));
    CUDA_CHECK(cudaFree(lambda_bar));           CUDA_CHECK(cudaFree(grad_lambda));
    CUDA_CHECK(cudaFree(lambda_moment));        CUDA_CHECK(cudaFree(original_dual_slack));
    CUDA_CHECK(cudaFree(direction_y));
    CUDA_CHECK(cudaFree(d_t_curr));             CUDA_CHECK(cudaFree(d_t_prev));
    CUDA_CHECK(cudaFree(d_beta));               CUDA_CHECK(cudaFree(d_lambda_dot));
    CUDA_CHECK(cudaFree(viol_sq));
    CUDA_CHECK(cudaFree(d_r_curr));             CUDA_CHECK(cudaFree(d_r_last));
    CUDA_CHECK(cudaFree(d_K));                  CUDA_CHECK(cudaFree(d_iter_since_restart));
    return result;
}

static void proj_restart_final_log(const feas_polish_result_t *primal_res,
                                   const feas_polish_result_t *dual_res,
                                   const pdhg_solver_state_t *state,
                                   bool verbose)
{
    if (verbose) {
        printf("---------------------------------------------------------------------------------------\n");
    }
    printf("Feasibility Polishing Summary (Proj + Adaptive AGD Restart)\n");
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

void proj_restart_scheme_feasibility_polish(const pdhg_parameters_t *params, pdhg_solver_state_t *state)
{
    clock_t feasibility_polishing_start_time = clock();

    if (state->relative_primal_residual < params->termination_criteria.eps_feas_polish_relative &&
        state->relative_dual_residual   < params->termination_criteria.eps_feas_polish_relative) {
        printf("Skipping feasibility polishing as the solution is already sufficiently feasible.\n");
        return;
    }

    /* Save warm-start primal x for cross-scheme-comparable dual_obj reporting. */
    double *warm_primal_x_backup;
    CUDA_CHECK(cudaMalloc(&warm_primal_x_backup, state->num_variables * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(warm_primal_x_backup, state->pdhg_primal_solution,
                          state->num_variables * sizeof(double), cudaMemcpyDeviceToDevice));

    feas_polish_result_t primal_res = proj_restart_primal_polishing(state, params);
    state->feasibility_iteration   += primal_res.iterations;
    state->primal_polish_iterations = primal_res.iterations;
    state->primal_polish_time_sec   = primal_res.time_sec;
    state->primal_polish_termination = primal_res.termination_reason;
    state->primal_polish_residual   = state->relative_primal_residual;

    feas_polish_result_t dual_res = proj_restart_dual_polishing(state, params, warm_primal_x_backup);
    state->feasibility_iteration += dual_res.iterations;
    state->dual_polish_iterations = dual_res.iterations;
    state->dual_polish_time_sec   = dual_res.time_sec;
    state->dual_polish_termination = dual_res.termination_reason;
    state->dual_polish_residual   = state->relative_dual_residual;

    state->objective_gap = fabs(state->primal_objective_value - state->dual_objective_value);
    state->relative_objective_gap =
        state->objective_gap / (1.0 + fabs(state->primal_objective_value) + fabs(state->dual_objective_value));
    state->polish_relative_gap = state->relative_objective_gap;

    proj_restart_final_log(&primal_res, &dual_res, state, params->verbose);
    CUDA_CHECK(cudaFree(warm_primal_x_backup));
    state->feasibility_polishing_time = (double)(clock() - feasibility_polishing_start_time) / CLOCKS_PER_SEC;
}
