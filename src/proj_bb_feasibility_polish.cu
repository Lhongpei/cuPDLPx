/*
 * Scheme 1 (Projection) + Barzilai-Borwein adaptive stepsize ON TOP of AGD.
 *
 * Structure mirrors proj_feasibility_polish.cu (Halpern/Nesterov extrapolation
 * with restart on negative inner product), but the fixed safe step
 *     h_safe = step_size^2 * 0.5  ( = 1/L for L = 2||A||^2 )
 * is replaced by a per-iteration BB-adaptive step:
 *
 *     h_BB1 = -<Delta d, Delta g> / ||Delta g||^2
 *     h_BB2 = ||Delta d||^2 / (-<Delta d, Delta g>)
 *
 * where Delta d = d_k - d_{k-1}  (the AGD iterate increment d_moment_k)
 * and   Delta g = g_k - g_{k-1}  (gradient evaluated at the extrapolated point
 * before each iterate update). Both are computed and reduced fully on device
 * so the inner loop can still be CUDA-graph-captured.
 *
 * BB1/BB2 alternate by parity of the iteration counter (a common BB stabilizer).
 * The step is clamped to [h_safe * 0.1, h_safe * 50] and falls back to h_safe
 * on a degenerate (negative or NaN) BB ratio.
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

#define BB_ALLOC_ZERO(dest, bytes)                                                                                     \
    CUDA_CHECK(cudaMalloc(&dest, bytes));                                                                              \
    CUDA_CHECK(cudaMemset(dest, 0, bytes));

/* Periodic-restart interval (iterations) for BB. Small enough to limit
 * blow-up windows on ill-conditioned instances, large enough to let BB
 * harvest acceleration on well-behaved ones. */
#define BB_RESTART_K 32

/* -----------------------------------------------------------------------
 * AGD scalar updates (graph-capture-safe; mirror proj_feasibility_polish.cu)
 * --------------------------------------------------------------------- */

__global__ void bb_update_t_and_beta_kernel(double *t_curr, double *t_prev, double *beta)
{
    *t_curr = 0.5 * (1.0 + sqrt(1.0 + 4.0 * (*t_prev) * (*t_prev)));
    *beta = (*t_prev - 1.0) / (*t_curr);
}

__global__ void bb_update_t_prev_primal_kernel(double *t_curr, double *t_prev, const double *d1_dot, const double *d2_dot)
{
    *t_prev = (*d1_dot + *d2_dot < 0.0) ? 1.0 : *t_curr;
}

__global__ void bb_update_t_prev_dual_kernel(double *t_curr, double *t_prev, const double *lam_dot)
{
    *t_prev = (*lam_dot < 0.0) ? 1.0 : *t_curr;
}

__global__ void bb_inc_counter_kernel(int *cnt) { (*cnt)++; }
__global__ void bb_add_scalar_kernel(double *dst, const double *src) { *dst += *src; }
__global__ void bb_set_step_kernel(double *dst, double val) { *dst = val; }

/* Periodic AGD restart safeguard: every K iters, force t_prev = 1 and reset
 * the step to h_safe. Without this, an unlucky BB step can blow up
 * the iterate on ill-conditioned instances (triptim1) and the gradient-direction
 * restart alone cannot recover (the momentum keeps amplifying the bad direction).
 */
__global__ void bb_periodic_restart_kernel(double *t_prev, double *d_step,
                                           const int *iter_counter, int K, double h_safe)
{
    int it = *iter_counter;
    if (it > 0 && (it % K) == 0) {
        *t_prev = 1.0;
        *d_step = h_safe;
    }
}

/* BB step from device-side scalars. Alternates BB1/BB2 by iteration parity.
 * On a "restart" (t_prev was reset to 1, which we detect via beta == 0) we
 * also fall back to h_safe so the BB ratio doesn't carry stale info across
 * the restart boundary.
 */
__global__ void bb_compute_step_kernel(double *d_step_out,
                                       const double *d_dot_dg,
                                       const double *d_norm_g2,
                                       const double *d_norm_d2,
                                       const double *d_beta_now,
                                       double h_safe,
                                       double h_min,
                                       double h_max,
                                       const int *iter_counter)
{
    (void)d_beta_now;
    double dot_dg = *d_dot_dg;
    double ng2    = *d_norm_g2;
    double nd2    = *d_norm_d2;
    int it        = *iter_counter;

    /* Concave ascent: <Δd, Δg> ≤ 0 (anti-correlated). |dot_dg|/ng2 is the BB1 step. */
    double abs_dot = fabs(dot_dg);
    double h;
    bool ok;
    if ((it & 1) == 0) {
        /* BB1: h = |<Δd,Δg>| / ||Δg||² */
        ok = (ng2 > 1e-300) && (abs_dot > 1e-30);
        h  = ok ? (abs_dot / ng2) : h_safe;
    } else {
        /* BB2: h = ||Δd||² / |<Δd,Δg>| */
        ok = (abs_dot > 1e-30) && (nd2 > 1e-300);
        h  = ok ? (nd2 / abs_dot) : h_safe;
    }
    if (!isfinite(h) || h <= 0.0) h = h_safe;
    h = fmin(fmax(h, h_min), h_max);
    *d_step_out = h;
}

/* -----------------------------------------------------------------------
 * Primal kernels (AGD body, device-pointer step)
 * --------------------------------------------------------------------- */

/* Extrapolation: d_bar_i = d_i + beta * (d_i - d_prev_i)
 * Writes d_moment_i  = d_i - d_prev_i (Δd between successive iterates).
 * Writes delta_d_dual = d1_bar - d2_bar (input to A^T spmv).
 */
__global__ void bb_primal_extrap_kernel(double *d1_bar, double *d2_bar,
                                        double *d1_moment, double *d2_moment,
                                        double *delta_d_dual,
                                        const double *__restrict__ d1, const double *__restrict__ d2,
                                        const double *__restrict__ d1_prev, const double *__restrict__ d2_prev,
                                        const double *beta_ptr, int num_con)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_con) return;
    double beta = *beta_ptr;
    double d1_delta = d1[idx] - d1_prev[idx];
    double d2_delta = d2[idx] - d2_prev[idx];
    double d1_bar_i = d1[idx] + beta * d1_delta;
    double d2_bar_i = d2[idx] + beta * d2_delta;
    delta_d_dual[idx] = d1_bar_i - d2_bar_i;
    d1_moment[idx] = d1_delta;
    d2_moment[idx] = d2_delta;
    d1_bar[idx] = d1_bar_i;
    d2_bar[idx] = d2_bar_i;
}

/* x = proj_X(x0 + direction). */
__global__ void bb_primal_proj_x_kernel(double *current_primal_solution,
                                        const double *__restrict__ initial_primal,
                                        const double *__restrict__ direction,
                                        const double *__restrict__ var_lb,
                                        const double *__restrict__ var_ub,
                                        int num_var)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_var) return;
    double v = initial_primal[idx] + direction[idx];
    current_primal_solution[idx] = fmin(fmax(v, var_lb[idx]), var_ub[idx]);
}

/* Snapshot grad_prev_bb = grad (saves the gradient computed at the PREVIOUS
 * iteration's extrapolated point, before this iter's iterate_kernel overwrites). */
__global__ void bb_save_grad_prev_kernel(double *g1_prev, double *g2_prev,
                                         const double *__restrict__ g1, const double *__restrict__ g2,
                                         int num_con)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_con) return;
    g1_prev[idx] = g1[idx];
    g2_prev[idx] = g2[idx];
}

/* d_prev = d (save before update); compute new grad from Ax; update d = max(d_bar + step*grad, 0). */
__global__ void bb_primal_iterate_kernel(double *d1, double *d2,
                                         double *d1_prev, double *d2_prev,
                                         double *grad_d1, double *grad_d2,
                                         const double *step_size_ptr,
                                         const double *__restrict__ d1_bar,
                                         const double *__restrict__ d2_bar,
                                         const double *__restrict__ Ax,
                                         const double *__restrict__ AL,
                                         const double *__restrict__ AU,
                                         int num_con)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_con) return;
    d1_prev[idx] = d1[idx];
    d2_prev[idx] = d2[idx];

    double AL_i = AL[idx];
    double AU_i = AU[idx];
    double Ax_i = Ax[idx];
    bool is_AL_fin = isfinite(AL_i);
    bool is_AU_fin = isfinite(AU_i);
    /* Mask BEFORE forming the difference so the gradient stays finite at
     * unbounded entries (otherwise inf - finite gives inf, and inf * 0 -> NaN
     * which would poison the BB dot products). */
    double g1_i = is_AL_fin ? (AL_i - Ax_i) : 0.0;
    double g2_i = is_AU_fin ? (Ax_i - AU_i) : 0.0;
    double step = *step_size_ptr;

    double d1_new = d1_bar[idx] + step * g1_i;
    d1[idx] = is_AL_fin ? fmax(d1_new, 0.0) : 0.0;

    double d2_new = d2_bar[idx] + step * g2_i;
    d2[idx] = is_AU_fin ? fmax(d2_new, 0.0) : 0.0;

    grad_d1[idx] = g1_i;
    grad_d2[idx] = g2_i;
}

/* delta_g_i = grad_i - grad_prev_bb_i. */
__global__ void bb_primal_delta_g_kernel(double *dg1, double *dg2,
                                         const double *__restrict__ g1, const double *__restrict__ g2,
                                         const double *__restrict__ g1_prev, const double *__restrict__ g2_prev,
                                         int num_con)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_con) return;
    dg1[idx] = g1[idx] - g1_prev[idx];
    dg2[idx] = g2[idx] - g2_prev[idx];
}

/* -----------------------------------------------------------------------
 * Dual kernels (AGD body, device-pointer step)
 * --------------------------------------------------------------------- */

__global__ void bb_dual_extrap_kernel(double *lambda_bar, double *lambda_moment,
                                      const double *__restrict__ lambda,
                                      const double *__restrict__ lambda_prev,
                                      const double *beta_ptr, int num_var)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_var) return;
    double beta = *beta_ptr;
    double delta = lambda[idx] - lambda_prev[idx];
    lambda_moment[idx] = delta;
    lambda_bar[idx]    = lambda[idx] + beta * delta;
}

__global__ void bb_dual_proj_ys_kernel(double *current_dual_solution,
                                       double *current_dual_slack,
                                       const double *__restrict__ target_dual_solution,
                                       const double *__restrict__ original_dual_slack,
                                       const double *__restrict__ direction_y,
                                       const double *__restrict__ direction_s,
                                       const double *__restrict__ con_lb,
                                       const double *__restrict__ con_ub,
                                       const double *__restrict__ var_lb,
                                       const double *__restrict__ var_ub,
                                       int num_con, int num_var)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_con + num_var) return;
    if (idx < num_var) {
        double s_min = isfinite(var_ub[idx]) ? -INFINITY : 0.0;
        double s_max = isfinite(var_lb[idx]) ?  INFINITY : 0.0;
        double v = original_dual_slack[idx] + direction_s[idx];
        current_dual_slack[idx] = fmin(fmax(v, s_min), s_max);
    } else {
        int ci = idx - num_var;
        double y_min = isfinite(con_ub[ci]) ? -INFINITY : 0.0;
        double y_max = isfinite(con_lb[ci]) ?  INFINITY : 0.0;
        double v = target_dual_solution[ci] + direction_y[ci];
        current_dual_solution[ci] = fmin(fmax(v, y_min), y_max);
    }
}

__global__ void bb_dual_save_grad_prev_kernel(double *g_prev, const double *__restrict__ g, int num_var)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_var) return;
    g_prev[idx] = g[idx];
}

__global__ void bb_dual_iterate_kernel(double *lambda, double *lambda_prev,
                                       const double *__restrict__ lambda_bar,
                                       double *grad_lambda,
                                       const double *step_size_ptr,
                                       const double *__restrict__ dual_product,
                                       const double *__restrict__ objective_vector,
                                       const double *__restrict__ dual_slack,
                                       int num_var)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_var) return;
    lambda_prev[idx] = lambda[idx];
    double g = objective_vector[idx] - dual_product[idx] - dual_slack[idx];
    lambda[idx] = lambda_bar[idx] + (*step_size_ptr) * g;
    grad_lambda[idx] = g;
}

__global__ void bb_dual_delta_g_kernel(double *dg, const double *__restrict__ g, const double *__restrict__ g_prev,
                                       int num_var)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_var) return;
    dg[idx] = g[idx] - g_prev[idx];
}

/* -----------------------------------------------------------------------
 * One BB+AGD primal iteration
 * --------------------------------------------------------------------- */
static void primal_iterate_once_bb(pdhg_solver_state_t *state,
                                   double *d1, double *d2, double *d1_prev, double *d2_prev,
                                   double *d1_bar, double *d2_bar,
                                   double *d1_moment, double *d2_moment,
                                   double *grad_d1, double *grad_d2,
                                   double *grad_d1_prev_bb, double *grad_d2_prev_bb,
                                   double *delta_g1, double *delta_g2,
                                   double *delta_d_dual, double *direction,
                                   double *d_t_curr, double *d_t_prev, double *d_beta,
                                   double *d_d1_dot, double *d_d2_dot,
                                   double *d_step, double *d_step_next,
                                   double *d_dot_dg, double *d_norm_g2, double *d_norm_d2,
                                   double *d_tmp1, double *d_tmp2, double *d_tmp3,
                                   int *d_iter_counter,
                                   double h_safe, double h_min, double h_max)
{
    /* AGD t/beta update. */
    bb_update_t_and_beta_kernel<<<1, 1, 0, state->stream>>>(d_t_curr, d_t_prev, d_beta);

    /* Extrapolate d_bar; record d_moment. */
    bb_primal_extrap_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream>>>(
        d1_bar, d2_bar, d1_moment, d2_moment, delta_d_dual, d1, d2, d1_prev, d2_prev,
        d_beta, state->num_constraints);

    /* x = proj_X(x0 + A^T (d1_bar - d2_bar)). */
    cupdlpx_spmv_ATx(state->sparse_handle, state->spmv_ctx, delta_d_dual, direction);
    bb_primal_proj_x_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream>>>(
        state->pdhg_primal_solution, state->initial_primal_solution, direction,
        state->variable_lower_bound, state->variable_upper_bound, state->num_variables);

    /* Ax. */
    cupdlpx_spmv_Ax(state->sparse_handle, state->spmv_ctx, state->pdhg_primal_solution, state->primal_product);

    /* Snapshot OLD grad (the one computed at previous iter's d_bar) into grad_prev_bb. */
    bb_save_grad_prev_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream>>>(
        grad_d1_prev_bb, grad_d2_prev_bb, grad_d1, grad_d2, state->num_constraints);

    /* Iterate: writes d_prev = d, then d = max(d_bar + step*grad, 0), and new grad. */
    bb_primal_iterate_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream>>>(
        d1, d2, d1_prev, d2_prev, grad_d1, grad_d2, d_step,
        d1_bar, d2_bar, state->primal_product,
        state->constraint_lower_bound, state->constraint_upper_bound, state->num_constraints);

    /* AGD restart check: <grad_new, d_moment>. */
    CUBLAS_CHECK(cublasDdot(state->blas_handle, state->num_constraints, grad_d1, 1, d1_moment, 1, d_d1_dot));
    CUBLAS_CHECK(cublasDdot(state->blas_handle, state->num_constraints, grad_d2, 1, d2_moment, 1, d_d2_dot));
    bb_update_t_prev_primal_kernel<<<1, 1, 0, state->stream>>>(d_t_curr, d_t_prev, d_d1_dot, d_d2_dot);
    /* Periodic AGD restart on top of gradient-direction restart. */
    bb_periodic_restart_kernel<<<1, 1, 0, state->stream>>>(d_t_prev, d_step, d_iter_counter,
                                                           BB_RESTART_K, h_safe);

    /* BB step computation for NEXT iter.
     * delta_g = grad_new - grad_prev_bb; <d_moment, delta_g>, ||delta_g||^2, ||d_moment||^2 */
    bb_primal_delta_g_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK, 0, state->stream>>>(
        delta_g1, delta_g2, grad_d1, grad_d2, grad_d1_prev_bb, grad_d2_prev_bb, state->num_constraints);

    CUBLAS_CHECK(cublasDdot(state->blas_handle, state->num_constraints, d1_moment, 1, delta_g1, 1, d_dot_dg));
    CUBLAS_CHECK(cublasDdot(state->blas_handle, state->num_constraints, d2_moment, 1, delta_g2, 1, d_tmp1));
    bb_add_scalar_kernel<<<1, 1, 0, state->stream>>>(d_dot_dg, d_tmp1);

    CUBLAS_CHECK(cublasDdot(state->blas_handle, state->num_constraints, delta_g1, 1, delta_g1, 1, d_norm_g2));
    CUBLAS_CHECK(cublasDdot(state->blas_handle, state->num_constraints, delta_g2, 1, delta_g2, 1, d_tmp2));
    bb_add_scalar_kernel<<<1, 1, 0, state->stream>>>(d_norm_g2, d_tmp2);

    CUBLAS_CHECK(cublasDdot(state->blas_handle, state->num_constraints, d1_moment, 1, d1_moment, 1, d_norm_d2));
    CUBLAS_CHECK(cublasDdot(state->blas_handle, state->num_constraints, d2_moment, 1, d2_moment, 1, d_tmp3));
    bb_add_scalar_kernel<<<1, 1, 0, state->stream>>>(d_norm_d2, d_tmp3);

    bb_compute_step_kernel<<<1, 1, 0, state->stream>>>(d_step_next,
                                                       d_dot_dg, d_norm_g2, d_norm_d2,
                                                       d_beta, h_safe, h_min, h_max, d_iter_counter);
    bb_inc_counter_kernel<<<1, 1, 0, state->stream>>>(d_iter_counter);

    /* Promote step. */
    CUDA_CHECK(cudaMemcpyAsync(d_step, d_step_next, sizeof(double), cudaMemcpyDeviceToDevice, state->stream));
}

/* -----------------------------------------------------------------------
 * One BB+AGD dual iteration
 * --------------------------------------------------------------------- */
static void dual_iterate_once_bb(pdhg_solver_state_t *state,
                                 double *lambda, double *lambda_prev, double *lambda_bar, double *lambda_moment,
                                 double *grad_lambda, double *grad_lambda_prev_bb, double *delta_g,
                                 double *direction_y, double *original_dual_slack,
                                 double *d_t_curr, double *d_t_prev, double *d_beta,
                                 double *d_lambda_dot,
                                 double *d_step, double *d_step_next,
                                 double *d_dot_dg, double *d_norm_g2, double *d_norm_d2,
                                 int *d_iter_counter,
                                 double h_safe, double h_min, double h_max)
{
    bb_update_t_and_beta_kernel<<<1, 1, 0, state->stream>>>(d_t_curr, d_t_prev, d_beta);

    bb_dual_extrap_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream>>>(
        lambda_bar, lambda_moment, lambda, lambda_prev, d_beta, state->num_variables);

    cupdlpx_spmv_Ax(state->sparse_handle, state->spmv_ctx, lambda_bar, direction_y);

    bb_dual_proj_ys_kernel<<<state->num_blocks_primal_dual, THREADS_PER_BLOCK, 0, state->stream>>>(
        state->pdhg_dual_solution, state->dual_slack,
        state->initial_dual_solution, original_dual_slack,
        direction_y, lambda_bar,
        state->constraint_lower_bound, state->constraint_upper_bound,
        state->variable_lower_bound, state->variable_upper_bound,
        state->num_constraints, state->num_variables);

    cupdlpx_spmv_ATx(state->sparse_handle, state->spmv_ctx, state->pdhg_dual_solution, state->dual_product);

    bb_dual_save_grad_prev_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream>>>(
        grad_lambda_prev_bb, grad_lambda, state->num_variables);

    bb_dual_iterate_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream>>>(
        lambda, lambda_prev, lambda_bar, grad_lambda, d_step,
        state->dual_product, state->objective_vector, state->dual_slack, state->num_variables);

    /* AGD restart check + periodic restart. */
    CUBLAS_CHECK(cublasDdot(state->blas_handle, state->num_variables, grad_lambda, 1, lambda_moment, 1, d_lambda_dot));
    bb_update_t_prev_dual_kernel<<<1, 1, 0, state->stream>>>(d_t_curr, d_t_prev, d_lambda_dot);
    bb_periodic_restart_kernel<<<1, 1, 0, state->stream>>>(d_t_prev, d_step, d_iter_counter,
                                                           BB_RESTART_K, h_safe);

    /* BB step. */
    bb_dual_delta_g_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK, 0, state->stream>>>(
        delta_g, grad_lambda, grad_lambda_prev_bb, state->num_variables);
    CUBLAS_CHECK(cublasDdot(state->blas_handle, state->num_variables, lambda_moment, 1, delta_g, 1, d_dot_dg));
    CUBLAS_CHECK(cublasDdot(state->blas_handle, state->num_variables, delta_g,      1, delta_g, 1, d_norm_g2));
    CUBLAS_CHECK(cublasDdot(state->blas_handle, state->num_variables, lambda_moment,1, lambda_moment, 1, d_norm_d2));

    bb_compute_step_kernel<<<1, 1, 0, state->stream>>>(d_step_next,
                                                       d_dot_dg, d_norm_g2, d_norm_d2,
                                                       d_beta, h_safe, h_min, h_max, d_iter_counter);
    bb_inc_counter_kernel<<<1, 1, 0, state->stream>>>(d_iter_counter);

    CUDA_CHECK(cudaMemcpyAsync(d_step, d_step_next, sizeof(double), cudaMemcpyDeviceToDevice, state->stream));
}

/* -----------------------------------------------------------------------
 * Primal driver
 * --------------------------------------------------------------------- */
static feas_polish_result_t proj_bb_primal_polishing(pdhg_solver_state_t *state, const pdhg_parameters_t *params)
{
    int num_con = state->num_constraints;
    int num_var = state->num_variables;
    feas_polish_result_t result = {false, 0, 0.0, TERMINATION_REASON_UNSPECIFIED};

    /* Backup. */
    double *backup_primal_solution;
    BB_ALLOC_ZERO(backup_primal_solution, num_var * sizeof(double));
    CUDA_CHECK(cudaMemcpy(backup_primal_solution, state->pdhg_primal_solution,
                          num_var * sizeof(double), cudaMemcpyDeviceToDevice));
    double backup_abs_residual = state->absolute_primal_residual;
    double backup_rel_residual = state->relative_primal_residual;
    double backup_obj_value    = state->primal_objective_value;

    /* AGD buffers. */
    double *d1, *d2, *d1_prev, *d2_prev, *d1_bar, *d2_bar;
    double *grad_d1, *grad_d2, *d1_moment, *d2_moment;
    double *delta_d_dual, *direction;
    BB_ALLOC_ZERO(d1,           num_con * sizeof(double));
    BB_ALLOC_ZERO(d2,           num_con * sizeof(double));
    BB_ALLOC_ZERO(d1_prev,      num_con * sizeof(double));
    BB_ALLOC_ZERO(d2_prev,      num_con * sizeof(double));
    BB_ALLOC_ZERO(d1_bar,       num_con * sizeof(double));
    BB_ALLOC_ZERO(d2_bar,       num_con * sizeof(double));
    BB_ALLOC_ZERO(grad_d1,      num_con * sizeof(double));
    BB_ALLOC_ZERO(grad_d2,      num_con * sizeof(double));
    BB_ALLOC_ZERO(d1_moment,    num_con * sizeof(double));
    BB_ALLOC_ZERO(d2_moment,    num_con * sizeof(double));
    BB_ALLOC_ZERO(delta_d_dual, num_con * sizeof(double));
    BB_ALLOC_ZERO(direction,    num_var * sizeof(double));

    /* BB extras. */
    double *grad_d1_prev_bb, *grad_d2_prev_bb, *delta_g1, *delta_g2;
    BB_ALLOC_ZERO(grad_d1_prev_bb, num_con * sizeof(double));
    BB_ALLOC_ZERO(grad_d2_prev_bb, num_con * sizeof(double));
    BB_ALLOC_ZERO(delta_g1,        num_con * sizeof(double));
    BB_ALLOC_ZERO(delta_g2,        num_con * sizeof(double));

    /* Device scalars. */
    double *d_t_curr, *d_t_prev, *d_beta, *d_d1_dot, *d_d2_dot;
    double *d_step, *d_step_next, *d_dot_dg, *d_norm_g2, *d_norm_d2;
    double *d_tmp1, *d_tmp2, *d_tmp3;
    int *d_iter_counter;
    BB_ALLOC_ZERO(d_t_curr,       sizeof(double));
    BB_ALLOC_ZERO(d_t_prev,       sizeof(double));
    BB_ALLOC_ZERO(d_beta,         sizeof(double));
    BB_ALLOC_ZERO(d_d1_dot,       sizeof(double));
    BB_ALLOC_ZERO(d_d2_dot,       sizeof(double));
    BB_ALLOC_ZERO(d_step,         sizeof(double));
    BB_ALLOC_ZERO(d_step_next,    sizeof(double));
    BB_ALLOC_ZERO(d_dot_dg,       sizeof(double));
    BB_ALLOC_ZERO(d_norm_g2,      sizeof(double));
    BB_ALLOC_ZERO(d_norm_d2,      sizeof(double));
    BB_ALLOC_ZERO(d_tmp1,         sizeof(double));
    BB_ALLOC_ZERO(d_tmp2,         sizeof(double));
    BB_ALLOC_ZERO(d_tmp3,         sizeof(double));
    BB_ALLOC_ZERO(d_iter_counter, sizeof(int));

    double init_t = 1.0;
    CUDA_CHECK(cudaMemcpy(d_t_curr, &init_t, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_t_prev, &init_t, sizeof(double), cudaMemcpyHostToDevice));

    double h_safe = state->step_size * state->step_size * 0.5;
    double h_min  = h_safe * 0.1;
    double h_max  = h_safe * 20.0;
    bb_set_step_kernel<<<1, 1, 0, state->stream>>>(d_step, h_safe);

    state->total_count         = 0;
    state->cumulative_time_sec = 0.0;
    state->termination_reason  = TERMINATION_REASON_UNSPECIFIED;

    CUDA_CHECK(cudaMemcpy(state->initial_primal_solution, state->pdhg_primal_solution,
                          num_var * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state->current_primal_solution, state->pdhg_primal_solution,
                          num_var * sizeof(double), cudaMemcpyDeviceToDevice));

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
                primal_iterate_once_bb(state, d1, d2, d1_prev, d2_prev, d1_bar, d2_bar,
                                       d1_moment, d2_moment, grad_d1, grad_d2,
                                       grad_d1_prev_bb, grad_d2_prev_bb, delta_g1, delta_g2,
                                       delta_d_dual, direction,
                                       d_t_curr, d_t_prev, d_beta, d_d1_dot, d_d2_dot,
                                       d_step, d_step_next,
                                       d_dot_dg, d_norm_g2, d_norm_d2,
                                       d_tmp1, d_tmp2, d_tmp3,
                                       d_iter_counter, h_safe, h_min, h_max);
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
    CUDA_CHECK(cudaFree(d1));            CUDA_CHECK(cudaFree(d2));
    CUDA_CHECK(cudaFree(d1_prev));       CUDA_CHECK(cudaFree(d2_prev));
    CUDA_CHECK(cudaFree(d1_bar));        CUDA_CHECK(cudaFree(d2_bar));
    CUDA_CHECK(cudaFree(grad_d1));       CUDA_CHECK(cudaFree(grad_d2));
    CUDA_CHECK(cudaFree(d1_moment));     CUDA_CHECK(cudaFree(d2_moment));
    CUDA_CHECK(cudaFree(delta_d_dual));  CUDA_CHECK(cudaFree(direction));
    CUDA_CHECK(cudaFree(grad_d1_prev_bb)); CUDA_CHECK(cudaFree(grad_d2_prev_bb));
    CUDA_CHECK(cudaFree(delta_g1));      CUDA_CHECK(cudaFree(delta_g2));
    CUDA_CHECK(cudaFree(d_t_curr));      CUDA_CHECK(cudaFree(d_t_prev));
    CUDA_CHECK(cudaFree(d_beta));
    CUDA_CHECK(cudaFree(d_d1_dot));      CUDA_CHECK(cudaFree(d_d2_dot));
    CUDA_CHECK(cudaFree(d_step));        CUDA_CHECK(cudaFree(d_step_next));
    CUDA_CHECK(cudaFree(d_dot_dg));      CUDA_CHECK(cudaFree(d_norm_g2));
    CUDA_CHECK(cudaFree(d_norm_d2));
    CUDA_CHECK(cudaFree(d_tmp1));        CUDA_CHECK(cudaFree(d_tmp2));
    CUDA_CHECK(cudaFree(d_tmp3));        CUDA_CHECK(cudaFree(d_iter_counter));
    return result;
}

/* -----------------------------------------------------------------------
 * Dual driver
 * --------------------------------------------------------------------- */
static feas_polish_result_t proj_bb_dual_polishing(pdhg_solver_state_t *state, const pdhg_parameters_t *params)
{
    int num_con = state->num_constraints;
    int num_var = state->num_variables;
    feas_polish_result_t result = {false, 0, 0.0, TERMINATION_REASON_UNSPECIFIED};

    double *backup_dual_solution, *backup_dual_slack;
    BB_ALLOC_ZERO(backup_dual_solution, num_con * sizeof(double));
    BB_ALLOC_ZERO(backup_dual_slack,    num_var * sizeof(double));
    CUDA_CHECK(cudaMemcpy(backup_dual_solution, state->pdhg_dual_solution,
                          num_con * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(backup_dual_slack, state->dual_slack,
                          num_var * sizeof(double), cudaMemcpyDeviceToDevice));
    double backup_abs_residual = state->absolute_dual_residual;
    double backup_rel_residual = state->relative_dual_residual;
    double backup_obj_value    = state->dual_objective_value;

    double *lambda, *lambda_prev, *lambda_bar, *grad_lambda, *lambda_moment;
    double *grad_lambda_prev_bb, *delta_g, *direction_y, *original_dual_slack;
    BB_ALLOC_ZERO(lambda,              num_var * sizeof(double));
    BB_ALLOC_ZERO(lambda_prev,         num_var * sizeof(double));
    BB_ALLOC_ZERO(lambda_bar,          num_var * sizeof(double));
    BB_ALLOC_ZERO(grad_lambda,         num_var * sizeof(double));
    BB_ALLOC_ZERO(lambda_moment,       num_var * sizeof(double));
    BB_ALLOC_ZERO(grad_lambda_prev_bb, num_var * sizeof(double));
    BB_ALLOC_ZERO(delta_g,             num_var * sizeof(double));
    BB_ALLOC_ZERO(direction_y,         num_con * sizeof(double));
    BB_ALLOC_ZERO(original_dual_slack, num_var * sizeof(double));

    double *d_t_curr, *d_t_prev, *d_beta, *d_lambda_dot;
    double *d_step, *d_step_next, *d_dot_dg, *d_norm_g2, *d_norm_d2;
    int *d_iter_counter;
    BB_ALLOC_ZERO(d_t_curr,       sizeof(double));
    BB_ALLOC_ZERO(d_t_prev,       sizeof(double));
    BB_ALLOC_ZERO(d_beta,         sizeof(double));
    BB_ALLOC_ZERO(d_lambda_dot,   sizeof(double));
    BB_ALLOC_ZERO(d_step,         sizeof(double));
    BB_ALLOC_ZERO(d_step_next,    sizeof(double));
    BB_ALLOC_ZERO(d_dot_dg,       sizeof(double));
    BB_ALLOC_ZERO(d_norm_g2,      sizeof(double));
    BB_ALLOC_ZERO(d_norm_d2,      sizeof(double));
    BB_ALLOC_ZERO(d_iter_counter, sizeof(int));

    double init_t = 1.0;
    CUDA_CHECK(cudaMemcpy(d_t_curr, &init_t, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_t_prev, &init_t, sizeof(double), cudaMemcpyHostToDevice));

    double h_safe = state->step_size * state->step_size * 0.5;
    double h_min  = h_safe * 0.1;
    double h_max  = h_safe * 20.0;
    bb_set_step_kernel<<<1, 1, 0, state->stream>>>(d_step, h_safe);

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
                                              state->pdhg_primal_solution,
                                              params->optimality_norm);
            check_feas_polishing_termination_criteria(state, state, &params->termination_criteria, false);
            state->cumulative_time_sec = (double)(clock() - start_time) / CLOCKS_PER_SEC;
            display_feas_polish_iteration_stats(state, params->verbose, false);
            if (state->termination_reason != TERMINATION_REASON_UNSPECIFIED) break;
        }

        if (!graph_created) {
            CUDA_CHECK(cudaStreamBeginCapture(state->stream, cudaStreamCaptureModeGlobal));
            for (int i = 0; i < params->termination_evaluation_frequency; i++) {
                dual_iterate_once_bb(state, lambda, lambda_prev, lambda_bar, lambda_moment,
                                     grad_lambda, grad_lambda_prev_bb, delta_g,
                                     direction_y, original_dual_slack,
                                     d_t_curr, d_t_prev, d_beta, d_lambda_dot,
                                     d_step, d_step_next,
                                     d_dot_dg, d_norm_g2, d_norm_d2,
                                     d_iter_counter, h_safe, h_min, h_max);
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
    CUDA_CHECK(cudaFree(lambda_moment));        CUDA_CHECK(cudaFree(grad_lambda_prev_bb));
    CUDA_CHECK(cudaFree(delta_g));              CUDA_CHECK(cudaFree(direction_y));
    CUDA_CHECK(cudaFree(original_dual_slack));
    CUDA_CHECK(cudaFree(d_t_curr));             CUDA_CHECK(cudaFree(d_t_prev));
    CUDA_CHECK(cudaFree(d_beta));               CUDA_CHECK(cudaFree(d_lambda_dot));
    CUDA_CHECK(cudaFree(d_step));               CUDA_CHECK(cudaFree(d_step_next));
    CUDA_CHECK(cudaFree(d_dot_dg));             CUDA_CHECK(cudaFree(d_norm_g2));
    CUDA_CHECK(cudaFree(d_norm_d2));            CUDA_CHECK(cudaFree(d_iter_counter));
    return result;
}

static void proj_bb_final_log(const feas_polish_result_t *primal_res,
                              const feas_polish_result_t *dual_res,
                              const pdhg_solver_state_t *state,
                              bool verbose)
{
    if (verbose) {
        printf("---------------------------------------------------------------------------------------\n");
    }
    printf("Feasibility Polishing Summary (BB+AGD)\n");
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

void proj_bb_scheme_feasibility_polish(const pdhg_parameters_t *params, pdhg_solver_state_t *state)
{
    clock_t feasibility_polishing_start_time = clock();

    if (state->relative_primal_residual < params->termination_criteria.eps_feas_polish_relative &&
        state->relative_dual_residual   < params->termination_criteria.eps_feas_polish_relative) {
        printf("Skipping feasibility polishing as the solution is already sufficiently feasible.\n");
        return;
    }

    feas_polish_result_t primal_res = proj_bb_primal_polishing(state, params);
    state->feasibility_iteration   += primal_res.iterations;
    state->primal_polish_iterations = primal_res.iterations;
    state->primal_polish_time_sec   = primal_res.time_sec;
    state->primal_polish_termination = primal_res.termination_reason;
    state->primal_polish_residual   = state->relative_primal_residual;

    feas_polish_result_t dual_res = proj_bb_dual_polishing(state, params);
    state->feasibility_iteration += dual_res.iterations;
    state->dual_polish_iterations = dual_res.iterations;
    state->dual_polish_time_sec   = dual_res.time_sec;
    state->dual_polish_termination = dual_res.termination_reason;
    state->dual_polish_residual   = state->relative_dual_residual;

    state->objective_gap = fabs(state->primal_objective_value - state->dual_objective_value);
    state->relative_objective_gap =
        state->objective_gap / (1.0 + fabs(state->primal_objective_value) + fabs(state->dual_objective_value));
    state->polish_relative_gap = state->relative_objective_gap;

    proj_bb_final_log(&primal_res, &dual_res, state, params->verbose);
    state->feasibility_polishing_time = (double)(clock() - feasibility_polishing_start_time) / CLOCKS_PER_SEC;
}
