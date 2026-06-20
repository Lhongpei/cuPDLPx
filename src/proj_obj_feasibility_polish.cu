/*
 * Scheme 1 (Projection) + objective term c^T x in the primal phase.
 *
 * Primal subproblem becomes
 *     min_{x in X}  c^T x + (1/2)||x - x0||^2 + (1/2)||residual(Ax)||^2
 * which is equivalent (up to a constant) to
 *     min_{x in X}  (1/2)||x - (x0 - c)||^2 + (1/2)||residual(Ax)||^2.
 *
 * So we keep Scheme 1's dual-ascent kernels unchanged and merely shift the
 * primal anchor: initial_primal_solution <- x0 - c at the start of primal
 * polish. The strong convexity (and therefore the dual-ascent derivation) is
 * preserved.
 *
 * The dual phase reuses Scheme 1's dual polish unchanged.
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

#define PO_ALLOC_ZERO(dest, bytes)                                                                                     \
    CUDA_CHECK(cudaMalloc(&dest, bytes));                                                                              \
    CUDA_CHECK(cudaMemset(dest, 0, bytes));

/* Reuse Scheme 1 primitives. */
extern void primal_feasibility_iterate_once(
    pdhg_solver_state_t *state,
    double *d1, double *d2, double *d1_prev, double *d2_prev, double *d1_bar,
    double *d2_bar, double *d1_moment, double *d2_moment, double *grad_d1,
    double *grad_d2, double *delta_d, double *direction,
    double *d_t_curr, double *d_t_prev, double *d_beta, double *d_d1_dot, double *d_d2_dot);

/* The dual phase is identical to Scheme 1. */
extern void proj_scheme_feasibility_polish(const pdhg_parameters_t *params, pdhg_solver_state_t *state);

/* x = x - alpha * c.  Applied to initial_primal_solution after the warm-start copy. */
__global__ void po_subtract_obj_kernel(double *x, const double *__restrict__ c, double alpha, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    x[idx] -= alpha * c[idx];
}

/* Read step-size weight alpha (PG-style step on the c-shift) from env. Defaults to 1.0. */
static double po_get_alpha(double step_size)
{
    const char *s = getenv("PROJ_OBJ_ALPHA");
    if (!s || !*s) return 1.0;
    if (strcmp(s, "step2") == 0) return step_size * step_size;   /* = 1/L approx */
    if (strcmp(s, "step")  == 0) return step_size;
    double v = atof(s);
    if (v == 0.0 && s[0] != '0') return 1.0;
    return v;
}

/* Dual-side anchor shifts that encode "+dual_obj" into the regularized dual
 * feasibility prox:
 *
 *   q(y, s) = sum_i [ max(y_i,0)*con_lb_fin + min(y_i,0)*con_ub_fin ]
 *           + sum_j [ max(-s_j,0)*var_lb_fin + min(-s_j,0)*var_ub_fin ]
 *
 * is piecewise linear; within the cone selected by sign of the warm-start
 * iterate the slope is constant, so we shift the anchor accordingly. For
 * range constraints (y free) and free vars the sign of the warm-start picks
 * the side that the iterate is most likely to stay on during polish.
 */
__global__ void po_shift_y_anchor_kernel(double *y_anchor,
                                          const double *__restrict__ con_lb_fin,
                                          const double *__restrict__ con_ub_fin,
                                          double alpha,
                                          int num_con)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_con) return;
    double y = y_anchor[i];
    double shift = (y >= 0.0) ? con_lb_fin[i] : con_ub_fin[i];
    y_anchor[i] = y + alpha * shift;
}

__global__ void po_shift_s_anchor_kernel(double *s_anchor,
                                          const double *__restrict__ var_lb_fin,
                                          const double *__restrict__ var_ub_fin,
                                          double alpha,
                                          int num_var)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= num_var) return;
    double s = s_anchor[j];
    double shift = (s >= 0.0) ? -var_ub_fin[j] : -var_lb_fin[j];
    s_anchor[j] = s + alpha * shift;
}

static double po_get_dual_alpha(double step_size)
{
    const char *s = getenv("PROJ_OBJ_DUAL_ALPHA");
    if (!s || !*s) return 0.0; /* disabled by default; b/u/l scale is too large */
    if (strcmp(s, "step2") == 0) return step_size * step_size;
    if (strcmp(s, "step")  == 0) return step_size;
    double v = atof(s);
    if (v == 0.0 && s[0] != '0') return 0.0;
    return v;
}

/* -----------------------------------------------------------------------
 * Primal driver: clone of proj_scheme_primal_feasibility_polishing
 * (proj_feasibility_polish.cu) with one extra kernel that subtracts c
 * from initial_primal_solution.
 * --------------------------------------------------------------------- */
static feas_polish_result_t proj_obj_primal_polishing(pdhg_solver_state_t *state,
                                                       const pdhg_parameters_t *params)
{
    int num_con = state->num_constraints;
    int num_var = state->num_variables;
    feas_polish_result_t result = {false, 0, 0.0, TERMINATION_REASON_UNSPECIFIED};

    /* Backup. */
    double *backup_primal_solution;
    PO_ALLOC_ZERO(backup_primal_solution, num_var * sizeof(double));
    CUDA_CHECK(cudaMemcpy(backup_primal_solution, state->pdhg_primal_solution,
                          num_var * sizeof(double), cudaMemcpyDeviceToDevice));
    double backup_abs_residual = state->absolute_primal_residual;
    double backup_rel_residual = state->relative_primal_residual;
    double backup_obj_value    = state->primal_objective_value;

    double *d1, *d2, *d1_prev, *d2_prev, *d1_bar, *d2_bar, *grad_d1, *grad_d2;
    double *d1_moment, *d2_moment, *delta_d, *direction;
    PO_ALLOC_ZERO(d1,        num_con * sizeof(double));
    PO_ALLOC_ZERO(d2,        num_con * sizeof(double));
    PO_ALLOC_ZERO(d1_prev,   num_con * sizeof(double));
    PO_ALLOC_ZERO(d2_prev,   num_con * sizeof(double));
    PO_ALLOC_ZERO(d1_bar,    num_con * sizeof(double));
    PO_ALLOC_ZERO(d2_bar,    num_con * sizeof(double));
    PO_ALLOC_ZERO(grad_d1,   num_con * sizeof(double));
    PO_ALLOC_ZERO(grad_d2,   num_con * sizeof(double));
    PO_ALLOC_ZERO(d1_moment, num_con * sizeof(double));
    PO_ALLOC_ZERO(d2_moment, num_con * sizeof(double));
    PO_ALLOC_ZERO(delta_d,   num_con * sizeof(double));
    PO_ALLOC_ZERO(direction, num_var * sizeof(double));

    double *d_t_curr, *d_t_prev, *d_beta, *d_d1_dot, *d_d2_dot;
    PO_ALLOC_ZERO(d_t_curr, sizeof(double));
    PO_ALLOC_ZERO(d_t_prev, sizeof(double));
    PO_ALLOC_ZERO(d_beta,   sizeof(double));
    PO_ALLOC_ZERO(d_d1_dot, sizeof(double));
    PO_ALLOC_ZERO(d_d2_dot, sizeof(double));

    double init_t = 1.0;
    CUDA_CHECK(cudaMemcpy(d_t_curr, &init_t, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_t_prev, &init_t, sizeof(double), cudaMemcpyHostToDevice));

    state->total_count         = 0;
    state->cumulative_time_sec = 0.0;
    state->termination_reason  = TERMINATION_REASON_UNSPECIFIED;

    /* Set anchor = warm-start primal x0. */
    CUDA_CHECK(cudaMemcpy(state->initial_primal_solution, state->pdhg_primal_solution,
                          num_var * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state->current_primal_solution, state->pdhg_primal_solution,
                          num_var * sizeof(double), cudaMemcpyDeviceToDevice));

    /* === The only difference from Scheme 1: anchor <- x0 - alpha*c. ===
     * Equivalent to adding alpha*c^T x to the regularized projection objective.
     * This is a single projected-gradient step with step size alpha on the
     * primal objective. alpha is read from PROJ_OBJ_ALPHA env var (default 1.0;
     * "step2" => state->step_size^2 (~1/L), "step" => state->step_size). */
    int blocks_primal = (num_var + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    double alpha = po_get_alpha(state->step_size);
    po_subtract_obj_kernel<<<blocks_primal, THREADS_PER_BLOCK, 0, state->stream>>>(
        state->initial_primal_solution, state->objective_vector, alpha, num_var);

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
    return result;
}

/* Reuse Scheme 1's dual polish driver (no obj term — dual side mirrors b^T y + ...,
 * which is more invasive to inject; left for a separate experiment). */
extern void proj_scheme_feas_polish_final_log_compat(const feas_polish_result_t *primal_res,
                                                     const feas_polish_result_t *dual_res,
                                                     const pdhg_solver_state_t *state,
                                                     bool verbose);

/* We need access to Scheme 1's dual_feasibility_iterate_once for the dual phase. */
extern void dual_feasibility_iterate_once(
    pdhg_solver_state_t *state,
    double *lambda, double *lambda_prev, double *lambda_bar,
    double *lambda_moment, double *grad_lambda, double *direction_y,
    double *original_dual_slack,
    double *d_t_curr, double *d_t_prev, double *d_beta, double *d_lambda_dot);

/* And we need to inline a dual polish driver that mirrors Scheme 1. Easiest:
 * extract Scheme 1's dual driver. Re-implement here for clarity. */
static feas_polish_result_t proj_obj_dual_polishing(pdhg_solver_state_t *state,
                                                     const pdhg_parameters_t *params,
                                                     const double *warm_primal_x)
{
    int num_con = state->num_constraints;
    int num_var = state->num_variables;
    feas_polish_result_t result = {false, 0, 0.0, TERMINATION_REASON_UNSPECIFIED};

    double *backup_dual_solution, *backup_dual_slack;
    PO_ALLOC_ZERO(backup_dual_solution, num_con * sizeof(double));
    PO_ALLOC_ZERO(backup_dual_slack,    num_var * sizeof(double));
    CUDA_CHECK(cudaMemcpy(backup_dual_solution, state->pdhg_dual_solution,
                          num_con * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(backup_dual_slack, state->dual_slack,
                          num_var * sizeof(double), cudaMemcpyDeviceToDevice));
    double backup_abs_residual = state->absolute_dual_residual;
    double backup_rel_residual = state->relative_dual_residual;
    double backup_obj_value    = state->dual_objective_value;

    double *lambda, *lambda_prev, *lambda_bar, *grad_lambda, *lambda_moment;
    double *direction_y, *original_dual_slack;
    PO_ALLOC_ZERO(lambda,              num_var * sizeof(double));
    PO_ALLOC_ZERO(lambda_prev,         num_var * sizeof(double));
    PO_ALLOC_ZERO(lambda_bar,          num_var * sizeof(double));
    PO_ALLOC_ZERO(grad_lambda,         num_var * sizeof(double));
    PO_ALLOC_ZERO(lambda_moment,       num_var * sizeof(double));
    PO_ALLOC_ZERO(original_dual_slack, num_var * sizeof(double));
    PO_ALLOC_ZERO(direction_y,         num_con * sizeof(double));

    double *d_t_curr, *d_t_prev, *d_beta, *d_lambda_dot;
    PO_ALLOC_ZERO(d_t_curr,     sizeof(double));
    PO_ALLOC_ZERO(d_t_prev,     sizeof(double));
    PO_ALLOC_ZERO(d_beta,       sizeof(double));
    PO_ALLOC_ZERO(d_lambda_dot, sizeof(double));

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

    /* Dual-side analog of the primal c-shift: pre-shift (y, s) anchors so the
     * regularized dual feasibility prox effectively maximizes the dual obj.
     * DISABLED BY DEFAULT — set PROJ_OBJ_DUAL_ALPHA to enable (raw b/u/l scales
     * are usually too large; step2 / step / small constants more sensible). */
    double dual_alpha = po_get_dual_alpha(state->step_size);
    if (dual_alpha != 0.0) {
        int b_con = (num_con + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        int b_var = (num_var + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        po_shift_y_anchor_kernel<<<b_con, THREADS_PER_BLOCK, 0, state->stream>>>(
            state->initial_dual_solution,
            state->constraint_lower_bound_finite_val,
            state->constraint_upper_bound_finite_val,
            dual_alpha,
            num_con);
        po_shift_s_anchor_kernel<<<b_var, THREADS_PER_BLOCK, 0, state->stream>>>(
            original_dual_slack,
            state->variable_lower_bound_finite_val,
            state->variable_upper_bound_finite_val,
            dual_alpha,
            num_var);
    }

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
    return result;
}

static void proj_obj_final_log(const feas_polish_result_t *primal_res,
                                const feas_polish_result_t *dual_res,
                                const pdhg_solver_state_t *state,
                                bool verbose)
{
    if (verbose) {
        printf("---------------------------------------------------------------------------------------\n");
    }
    printf("Feasibility Polishing Summary (Proj + c^T x in primal)\n");
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

void proj_obj_scheme_feasibility_polish(const pdhg_parameters_t *params, pdhg_solver_state_t *state)
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

    feas_polish_result_t primal_res = proj_obj_primal_polishing(state, params);
    state->feasibility_iteration   += primal_res.iterations;
    state->primal_polish_iterations = primal_res.iterations;
    state->primal_polish_time_sec   = primal_res.time_sec;
    state->primal_polish_termination = primal_res.termination_reason;
    state->primal_polish_residual   = state->relative_primal_residual;

    feas_polish_result_t dual_res = proj_obj_dual_polishing(state, params, warm_primal_x_backup);
    state->feasibility_iteration += dual_res.iterations;
    state->dual_polish_iterations = dual_res.iterations;
    state->dual_polish_time_sec   = dual_res.time_sec;
    state->dual_polish_termination = dual_res.termination_reason;
    state->dual_polish_residual   = state->relative_dual_residual;

    state->objective_gap = fabs(state->primal_objective_value - state->dual_objective_value);
    state->relative_objective_gap =
        state->objective_gap / (1.0 + fabs(state->primal_objective_value) + fabs(state->dual_objective_value));
    state->polish_relative_gap = state->relative_objective_gap;

    proj_obj_final_log(&primal_res, &dual_res, state, params->verbose);
    CUDA_CHECK(cudaFree(warm_primal_x_backup));
    state->feasibility_polishing_time = (double)(clock() - feasibility_polishing_start_time) / CLOCKS_PER_SEC;
}
