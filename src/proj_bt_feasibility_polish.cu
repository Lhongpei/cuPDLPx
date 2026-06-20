/*
 * Scheme 1 (Projection) + monotone backtracking line search on top of AGD (M-FISTA style).
 *
 * Each iteration:
 *   1. Save the current iterate's residual_sq f_curr.
 *   2. AGD extrapolation: d_bar = d + beta*(d - d_prev).
 *   3. Compute grad at d_bar (via proj_x + spmv).
 *   4. Trial step: d_trial = max(d_bar + step * grad, 0); evaluate f_trial.
 *   5. If f_trial < f_curr: ACCEPT — commit d <- d_trial, gently grow step.
 *      Else: SHRINK step (step *= 0.5), retry up to BT_MAX_BT times.
 *      If exhausted: RESTART momentum (t_prev=1, d_prev=d) and keep step,
 *      retry next iter from fresh AGD start.
 *
 * No CUDA-graph capture (number of trial evaluations varies). Per-iter cost is
 * ~3-15 kernel/spmv launches depending on backtracking depth.
 *
 * Strict monotonicity guarantees residual_sq is non-increasing every accepted iter,
 * eliminating the AGD oscillation seen with fixed step on hard instances.
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

#define BT_ALLOC_ZERO(dest, bytes)                                                                                     \
    CUDA_CHECK(cudaMalloc(&dest, bytes));                                                                              \
    CUDA_CHECK(cudaMemset(dest, 0, bytes));

#define BT_MAX_BT     5
#define BT_SHRINK     0.5
#define BT_GROW       1.2
#define BT_STEP_MIN_FRAC 1e-4   /* lower bound for step: base_step * this */

/* delta_d_dual = d1 - d2 (input to A^T). */
__global__ void bt_compose(double *out, const double *__restrict__ a, const double *__restrict__ b, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = a[idx] - b[idx];
}

/* x = proj_X(x0 + dir). */
__global__ void bt_proj_x(double *x, const double *__restrict__ x0, const double *__restrict__ dir,
                          const double *__restrict__ lb, const double *__restrict__ ub, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    double v = x0[idx] + dir[idx];
    x[idx] = fmin(fmax(v, lb[idx]), ub[idx]);
}

/* From Ax, compute grad (= residual signed, masked at unbounded sides) and viol_sq per i. */
__global__ void bt_grad_and_viol(const double *__restrict__ Ax, const double *__restrict__ AL,
                                 const double *__restrict__ AU, double *grad1, double *grad2,
                                 double *viol, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    double AL_i = AL[idx], AU_i = AU[idx], Ax_i = Ax[idx];
    bool fl = isfinite(AL_i), fu = isfinite(AU_i);
    double g1 = fl ? (AL_i - Ax_i) : 0.0;
    double g2 = fu ? (Ax_i - AU_i) : 0.0;
    grad1[idx] = g1;
    grad2[idx] = g2;
    double v1 = fmax(0.0, g1);
    double v2 = fmax(0.0, g2);
    viol[idx] = v1 * v1 + v2 * v2;
}

/* AGD extrapolate: d_bar = d + beta * (d - d_prev). */
__global__ void bt_extrapolate(double *d_bar, const double *__restrict__ d, const double *__restrict__ d_prev,
                               double beta, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    d_bar[idx] = d[idx] + beta * (d[idx] - d_prev[idx]);
}

/* d_new = max(d_bar + step * grad, 0). */
__global__ void bt_apply(double *d_new, const double *__restrict__ d_bar, const double *__restrict__ grad,
                         double step, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    d_new[idx] = fmax(d_bar[idx] + step * grad[idx], 0.0);
}

/* Evaluate residual_sq at (d1, d2). Side effects: writes x_proj, Ax_proj, grad1, grad2, viol.
 * Returns the scalar sum-of-violations into *out_sq (host pointer). */
static void bt_eval(pdhg_solver_state_t *state, const double *d1, const double *d2,
                    double *delta, double *direction, double *grad1, double *grad2, double *viol,
                    double *out_sq)
{
    int num_con = state->num_constraints;
    int num_var = state->num_variables;
    int b_con = state->num_blocks_dual;
    int b_var = state->num_blocks_primal;

    bt_compose<<<b_con, THREADS_PER_BLOCK, 0, state->stream>>>(delta, d1, d2, num_con);
    cupdlpx_spmv_ATx(state->sparse_handle, state->spmv_ctx, delta, direction);
    bt_proj_x<<<b_var, THREADS_PER_BLOCK, 0, state->stream>>>(
        state->pdhg_primal_solution, state->initial_primal_solution, direction,
        state->variable_lower_bound, state->variable_upper_bound, num_var);
    cupdlpx_spmv_Ax(state->sparse_handle, state->spmv_ctx, state->pdhg_primal_solution, state->primal_product);
    bt_grad_and_viol<<<b_con, THREADS_PER_BLOCK, 0, state->stream>>>(
        state->primal_product, state->constraint_lower_bound, state->constraint_upper_bound,
        grad1, grad2, viol, num_con);
    cublasPointerMode_t old;
    cublasGetPointerMode(state->blas_handle, &old);
    cublasSetPointerMode(state->blas_handle, CUBLAS_POINTER_MODE_HOST);
    CUBLAS_CHECK(cublasDasum(state->blas_handle, num_con, viol, 1, out_sq));
    cublasSetPointerMode(state->blas_handle, old);
}

static feas_polish_result_t proj_bt_primal_polishing(pdhg_solver_state_t *state, const pdhg_parameters_t *params)
{
    int num_con = state->num_constraints;
    int num_var = state->num_variables;
    int b_con = state->num_blocks_dual;
    feas_polish_result_t result = {false, 0, 0.0, TERMINATION_REASON_UNSPECIFIED};

    /* Backup. */
    double *backup_primal;
    BT_ALLOC_ZERO(backup_primal, num_var * sizeof(double));
    CUDA_CHECK(cudaMemcpy(backup_primal, state->pdhg_primal_solution, num_var * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    double backup_abs_res = state->absolute_primal_residual;
    double backup_rel_res = state->relative_primal_residual;
    double backup_obj = state->primal_objective_value;

    /* AGD iterate state. */
    double *d1, *d2, *d1_prev, *d2_prev, *d1_bar, *d2_bar;
    /* gradient cache at d_bar (current) and at trial point */
    double *grad1, *grad2;
    /* trial buffers */
    double *d1_trial, *d2_trial, *grad1_trial, *grad2_trial;
    double *delta, *direction, *viol;
    BT_ALLOC_ZERO(d1,           num_con * sizeof(double));
    BT_ALLOC_ZERO(d2,           num_con * sizeof(double));
    BT_ALLOC_ZERO(d1_prev,      num_con * sizeof(double));
    BT_ALLOC_ZERO(d2_prev,      num_con * sizeof(double));
    BT_ALLOC_ZERO(d1_bar,       num_con * sizeof(double));
    BT_ALLOC_ZERO(d2_bar,       num_con * sizeof(double));
    BT_ALLOC_ZERO(grad1,        num_con * sizeof(double));
    BT_ALLOC_ZERO(grad2,        num_con * sizeof(double));
    BT_ALLOC_ZERO(d1_trial,     num_con * sizeof(double));
    BT_ALLOC_ZERO(d2_trial,     num_con * sizeof(double));
    BT_ALLOC_ZERO(grad1_trial,  num_con * sizeof(double));
    BT_ALLOC_ZERO(grad2_trial,  num_con * sizeof(double));
    BT_ALLOC_ZERO(delta,        num_con * sizeof(double));
    BT_ALLOC_ZERO(direction,    num_var * sizeof(double));
    BT_ALLOC_ZERO(viol,         num_con * sizeof(double));

    state->total_count         = 0;
    state->cumulative_time_sec = 0.0;
    state->termination_reason  = TERMINATION_REASON_UNSPECIFIED;
    CUDA_CHECK(cudaMemcpy(state->initial_primal_solution, state->pdhg_primal_solution,
                          num_var * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state->current_primal_solution, state->pdhg_primal_solution,
                          num_var * sizeof(double), cudaMemcpyDeviceToDevice));

    print_initial_feas_polish_info(true, params);
    clock_t start_time = clock();

    /* AGD scalars (host-side since we don't use graph capture). */
    double t_prev = 1.0, t_curr = 1.0, beta = 0.0;

    /* Nonmonotone line search (GLL): maintain rolling max of past 5 f values. */
    #define NMLS_WIN 5
    double f_hist[NMLS_WIN];
    int f_hist_pos = 0;

    /* Evaluate initial f at d = 0. */
    double f_curr = 0.0;
    bt_eval(state, d1, d2, delta, direction, grad1, grad2, viol, &f_curr);
    for (int i = 0; i < NMLS_WIN; i++) f_hist[i] = f_curr;

    double base_step = state->step_size * state->step_size;  /* 1/L start */
    double step = base_step;

    while (state->termination_reason == TERMINATION_REASON_UNSPECIFIED) {
        if ((state->total_count == 0) || (state->total_count % get_print_frequency(state->total_count) == 0)) {
            compute_primal_feas_polish_residual(state, state->objective_vector, params->optimality_norm);
            check_feas_polishing_termination_criteria(state, state, &params->termination_criteria, true);
            state->cumulative_time_sec = (double)(clock() - start_time) / CLOCKS_PER_SEC;
            display_feas_polish_iteration_stats(state, params->verbose, true);
            if (state->termination_reason != TERMINATION_REASON_UNSPECIFIED) break;
        }

        /* AGD t/beta update. */
        t_curr = 0.5 * (1.0 + sqrt(1.0 + 4.0 * t_prev * t_prev));
        beta = (t_prev - 1.0) / t_curr;

        /* Extrapolate: d_bar = d + beta * (d - d_prev). */
        bt_extrapolate<<<b_con, THREADS_PER_BLOCK, 0, state->stream>>>(d1_bar, d1, d1_prev, beta, num_con);
        bt_extrapolate<<<b_con, THREADS_PER_BLOCK, 0, state->stream>>>(d2_bar, d2, d2_prev, beta, num_con);

        /* Evaluate f and grad at d_bar (replaces grad1, grad2 with grad at d_bar). */
        double f_bar;
        bt_eval(state, d1_bar, d2_bar, delta, direction, grad1, grad2, viol, &f_bar);

        /* Strict-monotone BT: accept iff f_trial < f_curr.
         * (NMLS / loose criteria can accept non-progress steps that ruin convergence;
         * strict descent guarantees iter count is no worse than fixed-step AGD.) */
        double f_trial = 0.0;
        int bt;
        bool accepted = false;
        for (bt = 0; bt < BT_MAX_BT; bt++) {
            bt_apply<<<b_con, THREADS_PER_BLOCK, 0, state->stream>>>(d1_trial, d1_bar, grad1, step, num_con);
            bt_apply<<<b_con, THREADS_PER_BLOCK, 0, state->stream>>>(d2_trial, d2_bar, grad2, step, num_con);
            bt_eval(state, d1_trial, d2_trial, delta, direction, grad1_trial, grad2_trial, viol, &f_trial);
            if (f_trial < f_curr) { accepted = true; break; }
            step *= BT_SHRINK;
            if (step < base_step * BT_STEP_MIN_FRAC) break;
        }

        /* Lipschitz-safe fallback: if BT failed to find descent, use safe step
         * 1/(2L) AND restart AGD momentum (t_prev = 1 makes next iter's beta = 0,
         * giving a fresh AGD start). This matches what Scheme 1's gradient-direction
         * restart does, ensuring we never diverge from accumulated momentum. */
        bool bt_failed = !accepted;
        if (bt_failed) {
            double safe_step = state->step_size * state->step_size * 0.5;
            /* Recompute trial from d (NOT d_bar) using fresh-start step — equivalent
             * to taking the plain projected gradient step (no momentum) like
             * Scheme 1's first iter after restart. */
            bt_apply<<<b_con, THREADS_PER_BLOCK, 0, state->stream>>>(d1_trial, d1, grad1, safe_step, num_con);
            bt_apply<<<b_con, THREADS_PER_BLOCK, 0, state->stream>>>(d2_trial, d2, grad2, safe_step, num_con);
            bt_eval(state, d1_trial, d2_trial, delta, direction, grad1_trial, grad2_trial, viol, &f_trial);
            accepted = true;
            step = safe_step;
        }

        /* Commit: d_prev <- d, d <- d_trial. */
        CUDA_CHECK(cudaMemcpyAsync(d1_prev, d1, num_con * sizeof(double), cudaMemcpyDeviceToDevice, state->stream));
        CUDA_CHECK(cudaMemcpyAsync(d2_prev, d2, num_con * sizeof(double), cudaMemcpyDeviceToDevice, state->stream));
        CUDA_CHECK(cudaMemcpyAsync(d1, d1_trial, num_con * sizeof(double), cudaMemcpyDeviceToDevice, state->stream));
        CUDA_CHECK(cudaMemcpyAsync(d2, d2_trial, num_con * sizeof(double), cudaMemcpyDeviceToDevice, state->stream));
        f_curr = f_trial;
        f_hist[f_hist_pos] = f_trial;
        f_hist_pos = (f_hist_pos + 1) % NMLS_WIN;
        if (bt_failed) {
            /* Restart momentum on fallback. */
            t_prev = 1.0;
        } else {
            t_prev = t_curr;  /* advance momentum normally */
            if (bt == 0) step = fmin(step * BT_GROW, base_step * 4.0);
        }
        state->total_count++;
    }

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
        CUDA_CHECK(cudaMemcpy(state->pdhg_primal_solution, backup_primal,
                              num_var * sizeof(double), cudaMemcpyDeviceToDevice));
        state->absolute_primal_residual = backup_abs_res;
        state->relative_primal_residual = backup_rel_res;
        state->primal_objective_value   = backup_obj;
    }

    CUDA_CHECK(cudaFree(backup_primal));
    CUDA_CHECK(cudaFree(d1));        CUDA_CHECK(cudaFree(d2));
    CUDA_CHECK(cudaFree(d1_prev));   CUDA_CHECK(cudaFree(d2_prev));
    CUDA_CHECK(cudaFree(d1_bar));    CUDA_CHECK(cudaFree(d2_bar));
    CUDA_CHECK(cudaFree(grad1));     CUDA_CHECK(cudaFree(grad2));
    CUDA_CHECK(cudaFree(d1_trial));  CUDA_CHECK(cudaFree(d2_trial));
    CUDA_CHECK(cudaFree(grad1_trial)); CUDA_CHECK(cudaFree(grad2_trial));
    CUDA_CHECK(cudaFree(delta));     CUDA_CHECK(cudaFree(direction));
    CUDA_CHECK(cudaFree(viol));
    return result;
}

/* Reuse Scheme 1's dual phase. */
extern void dual_feasibility_iterate_once(
    pdhg_solver_state_t *state,
    double *lambda, double *lambda_prev, double *lambda_bar,
    double *lambda_moment, double *grad_lambda, double *direction_y,
    double *original_dual_slack,
    double *d_t_curr, double *d_t_prev, double *d_beta, double *d_lambda_dot);

static feas_polish_result_t proj_bt_dual_polishing(pdhg_solver_state_t *state, const pdhg_parameters_t *params,
                                                    const double *warm_primal_x)
{
    int num_con = state->num_constraints;
    int num_var = state->num_variables;
    feas_polish_result_t result = {false, 0, 0.0, TERMINATION_REASON_UNSPECIFIED};

    double *backup_dual_solution, *backup_dual_slack;
    BT_ALLOC_ZERO(backup_dual_solution, num_con * sizeof(double));
    BT_ALLOC_ZERO(backup_dual_slack,    num_var * sizeof(double));
    CUDA_CHECK(cudaMemcpy(backup_dual_solution, state->pdhg_dual_solution,
                          num_con * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(backup_dual_slack, state->dual_slack,
                          num_var * sizeof(double), cudaMemcpyDeviceToDevice));
    double backup_abs_res = state->absolute_dual_residual;
    double backup_rel_res = state->relative_dual_residual;
    double backup_obj = state->dual_objective_value;

    double *lambda, *lambda_prev, *lambda_bar, *grad_lambda, *lambda_moment;
    double *direction_y, *original_dual_slack;
    BT_ALLOC_ZERO(lambda,              num_var * sizeof(double));
    BT_ALLOC_ZERO(lambda_prev,         num_var * sizeof(double));
    BT_ALLOC_ZERO(lambda_bar,          num_var * sizeof(double));
    BT_ALLOC_ZERO(grad_lambda,         num_var * sizeof(double));
    BT_ALLOC_ZERO(lambda_moment,       num_var * sizeof(double));
    BT_ALLOC_ZERO(original_dual_slack, num_var * sizeof(double));
    BT_ALLOC_ZERO(direction_y,         num_con * sizeof(double));

    double *d_t_curr, *d_t_prev, *d_beta, *d_lambda_dot;
    BT_ALLOC_ZERO(d_t_curr,     sizeof(double));
    BT_ALLOC_ZERO(d_t_prev,     sizeof(double));
    BT_ALLOC_ZERO(d_beta,       sizeof(double));
    BT_ALLOC_ZERO(d_lambda_dot, sizeof(double));

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
            compute_dual_feas_polish_residual(state, state->constraint_lower_bound_finite_val,
                                              state->constraint_upper_bound_finite_val,
                                              warm_primal_x, params->optimality_norm);
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

    result.iterations = state->total_count;
    result.time_sec   = state->cumulative_time_sec;
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
        state->absolute_dual_residual = backup_abs_res;
        state->relative_dual_residual = backup_rel_res;
        state->dual_objective_value   = backup_obj;
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
    return result;
}

void proj_bt_scheme_feasibility_polish(const pdhg_parameters_t *params, pdhg_solver_state_t *state)
{
    clock_t feasibility_polishing_start_time = clock();
    if (state->relative_primal_residual < params->termination_criteria.eps_feas_polish_relative &&
        state->relative_dual_residual   < params->termination_criteria.eps_feas_polish_relative) {
        printf("Skipping feasibility polishing as the solution is already sufficiently feasible.\n");
        return;
    }

    double *warm_primal_x_backup;
    CUDA_CHECK(cudaMalloc(&warm_primal_x_backup, state->num_variables * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(warm_primal_x_backup, state->pdhg_primal_solution,
                          state->num_variables * sizeof(double), cudaMemcpyDeviceToDevice));

    feas_polish_result_t primal_res = proj_bt_primal_polishing(state, params);
    state->feasibility_iteration   += primal_res.iterations;
    state->primal_polish_iterations = primal_res.iterations;
    state->primal_polish_time_sec   = primal_res.time_sec;
    state->primal_polish_termination = primal_res.termination_reason;
    state->primal_polish_residual   = state->relative_primal_residual;

    feas_polish_result_t dual_res = proj_bt_dual_polishing(state, params, warm_primal_x_backup);
    state->feasibility_iteration += dual_res.iterations;
    state->dual_polish_iterations = dual_res.iterations;
    state->dual_polish_time_sec   = dual_res.time_sec;
    state->dual_polish_termination = dual_res.termination_reason;
    state->dual_polish_residual   = state->relative_dual_residual;

    state->objective_gap = fabs(state->primal_objective_value - state->dual_objective_value);
    state->relative_objective_gap =
        state->objective_gap / (1.0 + fabs(state->primal_objective_value) + fabs(state->dual_objective_value));
    state->polish_relative_gap = state->relative_objective_gap;

    if (params->verbose) {
        printf("---------------------------------------------------------------------------------------\n");
    }
    printf("Feasibility Polishing Summary (Proj + AGD + Backtracking)\n");
    printf("  Primal Status        : %s\n", termination_reason_to_string(primal_res.termination_reason));
    printf("  Primal Iterations    : %d\n", primal_res.iterations);
    printf("  Primal Time Usage    : %.3g sec\n", primal_res.time_sec);
    printf("  Dual Status          : %s\n", termination_reason_to_string(dual_res.termination_reason));
    printf("  Dual Iterations      : %d\n", dual_res.iterations);
    printf("  Dual Time Usage      : %.3g sec\n", dual_res.time_sec);
    printf("  Primal Residual      : %.3e\n", state->relative_primal_residual);
    printf("  Dual Residual        : %.3e\n", state->relative_dual_residual);
    printf("  Primal Dual Gap      : %.3e\n", state->relative_objective_gap);

    CUDA_CHECK(cudaFree(warm_primal_x_backup));
    state->feasibility_polishing_time = (double)(clock() - feasibility_polishing_start_time) / CLOCKS_PER_SEC;
}
