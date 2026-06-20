/*
 * Scheme 2 (Norm Minimization) feasibility polishing.
 *
 * Primal:  min_{x in X}     ||(l_c - A x)_+||^2 + ||(A x - u_c)_+||^2
 * Dual:    min_{y in Y,
 *              s in R}      ||c - A^T y - s||^2
 *
 * Both subproblems are solved by Restart Accelerated Gradient Descent (rAGD)
 * with projection onto the respective box/cone constraints. Restart is
 * triggered when the inner product of the gradient direction and the
 * iterate displacement disagrees (O'Donoghue-Candes style).
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

#define S2_ALLOC_ZERO(dest, bytes)                                                                                     \
    CUDA_CHECK(cudaMalloc(&dest, bytes));                                                                              \
    CUDA_CHECK(cudaMemset(dest, 0, bytes));

/* -----------------------------------------------------------------------
 * Shared device kernels (suffix s2_ to avoid clash with proj_feasibility_polish.cu)
 * --------------------------------------------------------------------- */

__global__ void s2_update_t_and_beta_kernel(double *t_curr, double *t_prev, double *beta)
{
    *t_curr = 0.5 * (1.0 + sqrt(1.0 + 4.0 * (*t_prev) * (*t_prev)));
    *beta = (*t_prev - 1.0) / (*t_curr);
}

__global__ void s2_restart_if_negative_kernel(double *t_curr, double *t_prev, const double *dot)
{
    /* Restart when the dot product of "descent-direction-from-x_k" and the
     * achieved step direction is negative, i.e., the momentum is fighting
     * the gradient. Matches the convention used by Scheme 1.
     */
    *t_prev = (*dot < 0.0) ? 1.0 : *t_curr;
}

__global__ void s2_restart_if_negative_sum_kernel(double *t_curr, double *t_prev, const double *dot1,
                                                   const double *dot2)
{
    *t_prev = ((*dot1 + *dot2) < 0.0) ? 1.0 : *t_curr;
}

/* -----------------------------------------------------------------------
 * Primal Scheme 2 kernels
 * --------------------------------------------------------------------- */

__global__ void s2_primal_extrapolate_kernel(double *__restrict__ x_bar,
                                              const double *__restrict__ x,
                                              const double *__restrict__ x_prev,
                                              const double *beta_ptr,
                                              int num_var)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_var)
        return;
    double b = *beta_ptr;
    x_bar[idx] = x[idx] + b * (x[idx] - x_prev[idx]);
}

/* In-place transform: input  buffer[i] = (A x_bar)_i
 *                     output buffer[i] = 2 * ((A x_bar)_i - clip((A x_bar)_i, l_i, u_i))
 *
 * The sign of the result is signed:  negative if (A x_bar)_i < l_i,
 *                                    positive if (A x_bar)_i > u_i,
 *                                    zero otherwise.
 * SpMV-ing A^T against this buffer yields gradient/2.
 */
__global__ void s2_primal_violation_kernel(double *__restrict__ buf,
                                            const double *__restrict__ lb,
                                            const double *__restrict__ ub,
                                            int num_con)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_con)
        return;
    double v = buf[idx];
    double lb_i = lb[idx];
    double ub_i = ub[idx];
    double clamped = fmax(lb_i, fmin(v, ub_i));
    buf[idx] = 2.0 * (v - clamped);
}

/* x_new = proj_X(x_bar - step * grad)
 * Also: x_prev <- old x,  dx <- x_new - old x
 *       descent_dir <- x_new - x_bar (used for restart test)
 */
__global__ void s2_primal_step_kernel(double *__restrict__ x,
                                       double *__restrict__ x_prev,
                                       double *__restrict__ dx,
                                       double *__restrict__ descent_dir,
                                       const double *__restrict__ x_bar,
                                       const double *__restrict__ grad,
                                       const double *__restrict__ c,
                                       double alpha,
                                       const double *__restrict__ vlb,
                                       const double *__restrict__ vub,
                                       double step,
                                       int num_var)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_var)
        return;
    double x_old = x[idx];
    x_prev[idx] = x_old;
    /* Adds alpha * c to the gradient: min ||residual||^2 + alpha * c^T x  */
    double cand = x_bar[idx] - step * (grad[idx] + alpha * c[idx]);
    double x_new = fmin(fmax(cand, vlb[idx]), vub[idx]);
    x[idx] = x_new;
    dx[idx] = x_new - x_old;
    descent_dir[idx] = x_new - x_bar[idx];
}

static double s2_get_alpha(double step_size)
{
    const char *s = getenv("PROJ_OBJ_ALPHA");
    if (!s || !*s) return 0.0;
    if (strcmp(s, "step2") == 0) return step_size * step_size;
    if (strcmp(s, "step")  == 0) return step_size;
    double v = atof(s);
    if (v == 0.0 && s[0] != '0') return 0.0;
    return v;
}

static void s2_primal_iterate_once(pdhg_solver_state_t *state, double *x_bar, double *x_prev, double *dx,
                                    double *descent_dir, double *grad, double *r_buf, double step,
                                    double alpha,
                                    double *d_t_curr, double *d_t_prev, double *d_beta, double *d_dot)
{
    /* 1. update t,beta on device */
    s2_update_t_and_beta_kernel<<<1, 1, 0, state->stream>>>(d_t_curr, d_t_prev, d_beta);

    /* 2. x_bar = x + beta * (x - x_prev) */
    s2_primal_extrapolate_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream>>>(
        x_bar, state->pdhg_primal_solution, x_prev, d_beta, state->num_variables);

    /* 3. r_buf = A * x_bar */
    cupdlpx_spmv_Ax(state->sparse_handle, state->spmv_ctx, x_bar, r_buf);

    /* 4. r_buf <- 2 * (A x_bar - clip(A x_bar, lb_c, ub_c)) */
    s2_primal_violation_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream>>>(
        r_buf, state->constraint_lower_bound, state->constraint_upper_bound, state->num_constraints);

    /* 5. grad = A^T * r_buf */
    cupdlpx_spmv_ATx(state->sparse_handle, state->spmv_ctx, r_buf, grad);

    /* 6. x_new = proj(x_bar - step * (grad + alpha * c)); update x_prev, dx, descent_dir */
    s2_primal_step_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream>>>(
        state->pdhg_primal_solution, x_prev, dx, descent_dir, x_bar, grad,
        state->objective_vector, alpha,
        state->variable_lower_bound, state->variable_upper_bound,
        step, state->num_variables);

    /* 7. dot = descent_dir . dx (sign-aware for restart) */
    CUBLAS_CHECK(cublasDdot(state->blas_handle, state->num_variables, descent_dir, 1, dx, 1, d_dot));

    /* 8. restart if dot < 0 (descent direction disagrees with momentum) */
    s2_restart_if_negative_kernel<<<1, 1, 0, state->stream>>>(d_t_curr, d_t_prev, d_dot);
}

feas_polish_result_t norm_scheme_primal_feasibility_polishing(pdhg_solver_state_t *state,
                                                                const pdhg_parameters_t *params)
{
    int num_var = state->num_variables;
    int num_con = state->num_constraints;

    feas_polish_result_t result = {false, 0, 0.0, TERMINATION_REASON_UNSPECIFIED};

    /* ---- 0. Backup ---- */
    double *backup_primal_solution;
    S2_ALLOC_ZERO(backup_primal_solution, num_var * sizeof(double));
    CUDA_CHECK(cudaMemcpy(backup_primal_solution, state->pdhg_primal_solution, num_var * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    double backup_abs_residual = state->absolute_primal_residual;
    double backup_rel_residual = state->relative_primal_residual;
    double backup_obj_value = state->primal_objective_value;

    /* ---- 1. Allocations ---- */
    double *x_bar, *x_prev, *dx, *descent_dir, *grad, *r_buf;
    S2_ALLOC_ZERO(x_bar, num_var * sizeof(double));
    S2_ALLOC_ZERO(x_prev, num_var * sizeof(double));
    S2_ALLOC_ZERO(dx, num_var * sizeof(double));
    S2_ALLOC_ZERO(descent_dir, num_var * sizeof(double));
    S2_ALLOC_ZERO(grad, num_var * sizeof(double));
    S2_ALLOC_ZERO(r_buf, num_con * sizeof(double));

    /* x_prev starts as x (so first extrapolation is a no-op) */
    CUDA_CHECK(cudaMemcpy(x_prev, state->pdhg_primal_solution, num_var * sizeof(double), cudaMemcpyDeviceToDevice));

    double *d_t_curr, *d_t_prev, *d_beta, *d_dot;
    S2_ALLOC_ZERO(d_t_curr, sizeof(double));
    S2_ALLOC_ZERO(d_t_prev, sizeof(double));
    S2_ALLOC_ZERO(d_beta, sizeof(double));
    S2_ALLOC_ZERO(d_dot, sizeof(double));
    double init_t = 1.0;
    CUDA_CHECK(cudaMemcpy(d_t_curr, &init_t, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_t_prev, &init_t, sizeof(double), cudaMemcpyHostToDevice));

    /* ---- 2. State init ---- */
    state->total_count = 0;
    state->cumulative_time_sec = 0.0;
    state->termination_reason = TERMINATION_REASON_UNSPECIFIED;
    CUDA_CHECK(cudaMemcpy(state->initial_primal_solution, state->pdhg_primal_solution, num_var * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state->current_primal_solution, state->pdhg_primal_solution, num_var * sizeof(double),
                          cudaMemcpyDeviceToDevice));

    /* Optional c-shift step: gradient becomes (grad_residual + alpha * c). */
    double alpha = s2_get_alpha(state->step_size);
    /* F(x) = ||(l - Ax)_+||^2 + ||(Ax - u)_+||^2 has Lipschitz L = 2 ||A||^2.
     * PDHG keeps eta^2 ||A||^2 < 1, so step = eta^2 / 2 satisfies step * L <= 1
     * (1/L is the AGD-safe upper bound). */
    double step = 0.5 * state->step_size * state->step_size;

    print_initial_feas_polish_info(true, params);
    clock_t start_time = clock();
    cublasSetPointerMode(state->blas_handle, CUBLAS_POINTER_MODE_DEVICE);

    cudaGraphExec_t graphExec = NULL;
    bool graph_created = false;

    while (state->termination_reason == TERMINATION_REASON_UNSPECIFIED)
    {
        if ((state->is_this_major_iteration || state->total_count == 0) ||
            (state->total_count % get_print_frequency(state->total_count) == 0))
        {
            compute_primal_feas_polish_residual(state, state->objective_vector, params->optimality_norm);
            check_feas_polishing_termination_criteria(state, state, &params->termination_criteria, true);
            state->cumulative_time_sec = (double)(clock() - start_time) / CLOCKS_PER_SEC;
            display_feas_polish_iteration_stats(state, params->verbose, true);
            if (state->termination_reason != TERMINATION_REASON_UNSPECIFIED)
                break;
        }

        if (!graph_created)
        {
            CUDA_CHECK(cudaStreamBeginCapture(state->stream, cudaStreamCaptureModeGlobal));
            for (int i = 0; i < params->termination_evaluation_frequency; i++)
            {
                s2_primal_iterate_once(state, x_bar, x_prev, dx, descent_dir, grad, r_buf, step, alpha,
                                        d_t_curr, d_t_prev, d_beta, d_dot);
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
    result.time_sec = state->cumulative_time_sec;
    result.termination_reason = state->termination_reason;

    /* Keep polished iterate on SUCCESS / ITER_LIMIT / TIME_LIMIT. */
    if (state->termination_reason == TERMINATION_REASON_FEAS_POLISH_SUCCESS ||
        state->termination_reason == TERMINATION_REASON_ITERATION_LIMIT ||
        state->termination_reason == TERMINATION_REASON_TIME_LIMIT)
    {
        result.success = (state->termination_reason == TERMINATION_REASON_FEAS_POLISH_SUCCESS);
        CUDA_CHECK(cudaMemcpy(state->current_primal_solution, state->pdhg_primal_solution, num_var * sizeof(double),
                              cudaMemcpyDeviceToDevice));
    }
    else
    {
        result.success = false;
        CUDA_CHECK(cudaMemcpy(state->pdhg_primal_solution, backup_primal_solution, num_var * sizeof(double),
                              cudaMemcpyDeviceToDevice));
        state->absolute_primal_residual = backup_abs_residual;
        state->relative_primal_residual = backup_rel_residual;
        state->primal_objective_value = backup_obj_value;
    }

    if (graphExec)
        CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK(cudaFree(backup_primal_solution));
    CUDA_CHECK(cudaFree(x_bar));
    CUDA_CHECK(cudaFree(x_prev));
    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(descent_dir));
    CUDA_CHECK(cudaFree(grad));
    CUDA_CHECK(cudaFree(r_buf));
    CUDA_CHECK(cudaFree(d_t_curr));
    CUDA_CHECK(cudaFree(d_t_prev));
    CUDA_CHECK(cudaFree(d_beta));
    CUDA_CHECK(cudaFree(d_dot));

    return result;
}

/* -----------------------------------------------------------------------
 * Dual Scheme 2 kernels
 * --------------------------------------------------------------------- */

__global__ void s2_dual_extrapolate_y_kernel(double *__restrict__ y_bar,
                                              const double *__restrict__ y,
                                              const double *__restrict__ y_prev,
                                              const double *beta_ptr,
                                              int num_con)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_con)
        return;
    double b = *beta_ptr;
    y_bar[idx] = y[idx] + b * (y[idx] - y_prev[idx]);
}

__global__ void s2_dual_extrapolate_s_kernel(double *__restrict__ s_bar,
                                              const double *__restrict__ s,
                                              const double *__restrict__ s_prev,
                                              const double *beta_ptr,
                                              int num_var)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_var)
        return;
    double b = *beta_ptr;
    s_bar[idx] = s[idx] + b * (s[idx] - s_prev[idx]);
}

/* z = c - A^T y_bar - s_bar, written into the dual_product slot (num_var) */
__global__ void s2_dual_residual_kernel(double *__restrict__ z,
                                         const double *__restrict__ c,
                                         const double *__restrict__ ATy_bar,
                                         const double *__restrict__ s_bar,
                                         int num_var)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_var)
        return;
    z[idx] = c[idx] - ATy_bar[idx] - s_bar[idx];
}

/* y_new = proj_Y(y_bar + 2*step * Az),
 * dy = y_new - y_cur,  descent_dir_y = y_new - y_bar (signed),
 * y_prev = old y_cur.
 *
 * Y is determined by which constraint bounds are finite:
 *   l_c=-inf, u_c=inf  -> Y = {0}
 *   l_c=-inf, u_c<inf  -> Y = (-inf, 0]
 *   l_c<inf,  u_c=inf  -> Y = [0, +inf)
 *   both finite        -> Y = R
 */
__global__ void s2_dual_step_y_kernel(double *__restrict__ y,
                                       double *__restrict__ y_prev,
                                       double *__restrict__ dy,
                                       double *__restrict__ descent_dir_y,
                                       const double *__restrict__ y_bar,
                                       const double *__restrict__ Az,
                                       const double *__restrict__ con_lb,
                                       const double *__restrict__ con_ub,
                                       double step2,
                                       int num_con)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_con)
        return;
    double y_old = y[idx];
    y_prev[idx] = y_old;
    double cand = y_bar[idx] + step2 * Az[idx];
    double ub_finite = isfinite(con_lb[idx]) ? INFINITY : 0.0;
    double lb_finite = isfinite(con_ub[idx]) ? -INFINITY : 0.0;
    double y_new = fmin(fmax(cand, lb_finite), ub_finite);
    y[idx] = y_new;
    dy[idx] = y_new - y_old;
    descent_dir_y[idx] = y_new - y_bar[idx];
}

/* s_new = proj_R(s_bar + 2*step * z), and book-keeping.
 *
 * R is determined by which variable bounds are finite (analogous to Y):
 *   l_v=-inf, u_v=inf  -> R = {0}
 *   l_v=-inf, u_v<inf  -> R = (-inf, 0]
 *   l_v<inf,  u_v=inf  -> R = [0, +inf)
 *   both finite        -> R = R
 */
__global__ void s2_dual_step_s_kernel(double *__restrict__ s,
                                       double *__restrict__ s_prev,
                                       double *__restrict__ ds,
                                       double *__restrict__ descent_dir_s,
                                       const double *__restrict__ s_bar,
                                       const double *__restrict__ z,
                                       const double *__restrict__ var_lb,
                                       const double *__restrict__ var_ub,
                                       double step2,
                                       int num_var)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_var)
        return;
    double s_old = s[idx];
    s_prev[idx] = s_old;
    double cand = s_bar[idx] + step2 * z[idx];
    double ub_finite = isfinite(var_lb[idx]) ? INFINITY : 0.0;
    double lb_finite = isfinite(var_ub[idx]) ? -INFINITY : 0.0;
    double s_new = fmin(fmax(cand, lb_finite), ub_finite);
    s[idx] = s_new;
    ds[idx] = s_new - s_old;
    descent_dir_s[idx] = s_new - s_bar[idx];
}

static void s2_dual_iterate_once(pdhg_solver_state_t *state, double *y_bar, double *y_prev, double *dy,
                                  double *descent_dir_y, double *s_bar, double *s_prev, double *ds,
                                  double *descent_dir_s, double *z_buf, double *Az_buf, double step,
                                  double *d_t_curr, double *d_t_prev, double *d_beta, double *d_dot_y,
                                  double *d_dot_s)
{
    /* 1. update t,beta */
    s2_update_t_and_beta_kernel<<<1, 1, 0, state->stream>>>(d_t_curr, d_t_prev, d_beta);

    /* 2. extrapolate y_bar, s_bar */
    s2_dual_extrapolate_y_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream>>>(
        y_bar, state->pdhg_dual_solution, y_prev, d_beta, state->num_constraints);
    s2_dual_extrapolate_s_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream>>>(
        s_bar, state->dual_slack, s_prev, d_beta, state->num_variables);

    /* 3. dual_product = A^T * y_bar */
    cupdlpx_spmv_ATx(state->sparse_handle, state->spmv_ctx, y_bar, state->dual_product);

    /* 4. z = c - dual_product - s_bar */
    s2_dual_residual_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream>>>(
        z_buf, state->objective_vector, state->dual_product, s_bar, state->num_variables);

    /* 5. Az = A * z */
    cupdlpx_spmv_Ax(state->sparse_handle, state->spmv_ctx, z_buf, Az_buf);

    /* 6. y_new = proj_Y(y_bar + 2*step * Az);  s_new = proj_R(s_bar + 2*step * z) */
    double step2 = 2.0 * step;
    s2_dual_step_y_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream>>>(
        state->pdhg_dual_solution, y_prev, dy, descent_dir_y, y_bar, Az_buf, state->constraint_lower_bound,
        state->constraint_upper_bound, step2, state->num_constraints);
    s2_dual_step_s_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream>>>(
        state->dual_slack, s_prev, ds, descent_dir_s, s_bar, z_buf, state->variable_lower_bound,
        state->variable_upper_bound, step2, state->num_variables);

    /* 7. restart dot = descent_dir_y . dy + descent_dir_s . ds */
    CUBLAS_CHECK(cublasDdot(state->blas_handle, state->num_constraints, descent_dir_y, 1, dy, 1, d_dot_y));
    CUBLAS_CHECK(cublasDdot(state->blas_handle, state->num_variables, descent_dir_s, 1, ds, 1, d_dot_s));

    /* 8. restart if sum < 0 */
    s2_restart_if_negative_sum_kernel<<<1, 1, 0, state->stream>>>(d_t_curr, d_t_prev, d_dot_y, d_dot_s);
}

feas_polish_result_t norm_scheme_dual_feasibility_polishing(pdhg_solver_state_t *state,
                                                              const pdhg_parameters_t *params,
                                                              const double *warm_primal_x)
{
    int num_var = state->num_variables;
    int num_con = state->num_constraints;

    feas_polish_result_t result = {false, 0, 0.0, TERMINATION_REASON_UNSPECIFIED};

    /* ---- 0. Backup ---- */
    double *backup_dual_solution, *backup_dual_slack;
    S2_ALLOC_ZERO(backup_dual_solution, num_con * sizeof(double));
    S2_ALLOC_ZERO(backup_dual_slack, num_var * sizeof(double));
    CUDA_CHECK(cudaMemcpy(backup_dual_solution, state->pdhg_dual_solution, num_con * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(backup_dual_slack, state->dual_slack, num_var * sizeof(double), cudaMemcpyDeviceToDevice));
    double backup_abs_residual = state->absolute_dual_residual;
    double backup_rel_residual = state->relative_dual_residual;
    double backup_obj_value = state->dual_objective_value;

    /* ---- 1. Allocations ---- */
    double *y_bar, *y_prev, *dy, *descent_dir_y;
    double *s_bar, *s_prev, *ds, *descent_dir_s;
    double *z_buf, *Az_buf;
    S2_ALLOC_ZERO(y_bar, num_con * sizeof(double));
    S2_ALLOC_ZERO(y_prev, num_con * sizeof(double));
    S2_ALLOC_ZERO(dy, num_con * sizeof(double));
    S2_ALLOC_ZERO(descent_dir_y, num_con * sizeof(double));
    S2_ALLOC_ZERO(s_bar, num_var * sizeof(double));
    S2_ALLOC_ZERO(s_prev, num_var * sizeof(double));
    S2_ALLOC_ZERO(ds, num_var * sizeof(double));
    S2_ALLOC_ZERO(descent_dir_s, num_var * sizeof(double));
    S2_ALLOC_ZERO(z_buf, num_var * sizeof(double));
    S2_ALLOC_ZERO(Az_buf, num_con * sizeof(double));

    /* y_prev = y, s_prev = s so first extrapolation is identity */
    CUDA_CHECK(
        cudaMemcpy(y_prev, state->pdhg_dual_solution, num_con * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(s_prev, state->dual_slack, num_var * sizeof(double), cudaMemcpyDeviceToDevice));

    double *d_t_curr, *d_t_prev, *d_beta, *d_dot_y, *d_dot_s;
    S2_ALLOC_ZERO(d_t_curr, sizeof(double));
    S2_ALLOC_ZERO(d_t_prev, sizeof(double));
    S2_ALLOC_ZERO(d_beta, sizeof(double));
    S2_ALLOC_ZERO(d_dot_y, sizeof(double));
    S2_ALLOC_ZERO(d_dot_s, sizeof(double));
    double init_t = 1.0;
    CUDA_CHECK(cudaMemcpy(d_t_curr, &init_t, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_t_prev, &init_t, sizeof(double), cudaMemcpyHostToDevice));

    /* ---- 2. State init ---- */
    state->total_count = 0;
    state->cumulative_time_sec = 0.0;
    state->termination_reason = TERMINATION_REASON_UNSPECIFIED;
    CUDA_CHECK(cudaMemcpy(state->initial_dual_solution, state->pdhg_dual_solution, num_con * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state->current_dual_solution, state->pdhg_dual_solution, num_con * sizeof(double),
                          cudaMemcpyDeviceToDevice));

    /* F(y, s) = ||c - A^T y - s||^2 has Lipschitz L = 2(||A||^2 + 1).
     * Safe AGD step is eta^2 / (2 * (1 + eta^2)). */
    double eta2 = state->step_size * state->step_size;
    double step = 0.5 * eta2 / (1.0 + eta2);

    print_initial_feas_polish_info(false, params);
    clock_t start_time = clock();
    cublasSetPointerMode(state->blas_handle, CUBLAS_POINTER_MODE_DEVICE);

    cudaGraphExec_t graphExec = NULL;
    bool graph_created = false;

    while (state->termination_reason == TERMINATION_REASON_UNSPECIFIED)
    {
        if ((state->is_this_major_iteration || state->total_count == 0) ||
            (state->total_count % get_print_frequency(state->total_count) == 0))
        {
            /* Use the WARM-START primal x in the s^T x term so the reported
             * dual_obj is comparable across schemes (see note in proj_feasibility_polish.cu). */
            compute_dual_feas_polish_residual(state, state->constraint_lower_bound_finite_val,
                                              state->constraint_upper_bound_finite_val,
                                              warm_primal_x, params->optimality_norm);
            check_feas_polishing_termination_criteria(state, state, &params->termination_criteria, false);
            state->cumulative_time_sec = (double)(clock() - start_time) / CLOCKS_PER_SEC;
            display_feas_polish_iteration_stats(state, params->verbose, false);
            if (state->termination_reason != TERMINATION_REASON_UNSPECIFIED)
                break;
        }

        if (!graph_created)
        {
            CUDA_CHECK(cudaStreamBeginCapture(state->stream, cudaStreamCaptureModeGlobal));
            for (int i = 0; i < params->termination_evaluation_frequency; i++)
            {
                s2_dual_iterate_once(state, y_bar, y_prev, dy, descent_dir_y, s_bar, s_prev, ds, descent_dir_s,
                                      z_buf, Az_buf, step, d_t_curr, d_t_prev, d_beta, d_dot_y, d_dot_s);
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
    result.time_sec = state->cumulative_time_sec;
    result.termination_reason = state->termination_reason;

    /* Keep polished iterate on SUCCESS / ITER_LIMIT / TIME_LIMIT. */
    if (state->termination_reason == TERMINATION_REASON_FEAS_POLISH_SUCCESS ||
        state->termination_reason == TERMINATION_REASON_ITERATION_LIMIT ||
        state->termination_reason == TERMINATION_REASON_TIME_LIMIT)
    {
        result.success = (state->termination_reason == TERMINATION_REASON_FEAS_POLISH_SUCCESS);
    }
    else
    {
        result.success = false;
        CUDA_CHECK(cudaMemcpy(state->pdhg_dual_solution, backup_dual_solution, num_con * sizeof(double),
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(state->dual_slack, backup_dual_slack, num_var * sizeof(double),
                              cudaMemcpyDeviceToDevice));
        state->absolute_dual_residual = backup_abs_residual;
        state->relative_dual_residual = backup_rel_residual;
        state->dual_objective_value = backup_obj_value;
    }

    if (graphExec)
        CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK(cudaFree(backup_dual_solution));
    CUDA_CHECK(cudaFree(backup_dual_slack));
    CUDA_CHECK(cudaFree(y_bar));
    CUDA_CHECK(cudaFree(y_prev));
    CUDA_CHECK(cudaFree(dy));
    CUDA_CHECK(cudaFree(descent_dir_y));
    CUDA_CHECK(cudaFree(s_bar));
    CUDA_CHECK(cudaFree(s_prev));
    CUDA_CHECK(cudaFree(ds));
    CUDA_CHECK(cudaFree(descent_dir_s));
    CUDA_CHECK(cudaFree(z_buf));
    CUDA_CHECK(cudaFree(Az_buf));
    CUDA_CHECK(cudaFree(d_t_curr));
    CUDA_CHECK(cudaFree(d_t_prev));
    CUDA_CHECK(cudaFree(d_beta));
    CUDA_CHECK(cudaFree(d_dot_y));
    CUDA_CHECK(cudaFree(d_dot_s));

    return result;
}

/* -----------------------------------------------------------------------
 * Final logger + driver
 * --------------------------------------------------------------------- */

static void norm_scheme_feas_polish_final_log(const feas_polish_result_t *primal_res,
                                                const feas_polish_result_t *dual_res,
                                                const pdhg_solver_state_t *state, bool verbose)
{
    if (verbose)
    {
        printf("---------------------------------------------------------------------------------------\n");
    }
    printf("Feasibility Polishing Summary (Scheme 2: Norm Minimization)\n");
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

void norm_scheme_feasibility_polish(const pdhg_parameters_t *params, pdhg_solver_state_t *state)
{
    clock_t feasibility_polishing_start_time = clock();

    if (state->relative_primal_residual < params->termination_criteria.eps_feas_polish_relative &&
        state->relative_dual_residual < params->termination_criteria.eps_feas_polish_relative)
    {
        printf("Skipping feasibility polishing as the solution is already sufficiently feasible.\n");
        return;
    }

    /* Save warm-start primal x for cross-scheme-comparable dual_obj reporting. */
    double *warm_primal_x_backup;
    CUDA_CHECK(cudaMalloc(&warm_primal_x_backup, state->num_variables * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(warm_primal_x_backup, state->pdhg_primal_solution,
                          state->num_variables * sizeof(double), cudaMemcpyDeviceToDevice));

    feas_polish_result_t primal_res = norm_scheme_primal_feasibility_polishing(state, params);
    state->feasibility_iteration += primal_res.iterations;
    state->primal_polish_iterations = primal_res.iterations;
    state->primal_polish_time_sec = primal_res.time_sec;
    state->primal_polish_termination = primal_res.termination_reason;
    state->primal_polish_residual = state->relative_primal_residual;

    feas_polish_result_t dual_res = norm_scheme_dual_feasibility_polishing(state, params, warm_primal_x_backup);
    state->feasibility_iteration += dual_res.iterations;
    state->dual_polish_iterations = dual_res.iterations;
    state->dual_polish_time_sec = dual_res.time_sec;
    state->dual_polish_termination = dual_res.termination_reason;
    state->dual_polish_residual = state->relative_dual_residual;

    state->objective_gap = fabs(state->primal_objective_value - state->dual_objective_value);
    state->relative_objective_gap =
        state->objective_gap / (1.0 + fabs(state->primal_objective_value) + fabs(state->dual_objective_value));
    state->polish_relative_gap = state->relative_objective_gap;

    norm_scheme_feas_polish_final_log(&primal_res, &dual_res, state, params->verbose);

    CUDA_CHECK(cudaFree(warm_primal_x_backup));
    state->feasibility_polishing_time = (double)(clock() - feasibility_polishing_start_time) / CLOCKS_PER_SEC;
}
