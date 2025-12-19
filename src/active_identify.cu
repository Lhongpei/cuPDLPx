#include "active_identify.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "utils.h"
#include "preconditioner.h"
#include "solver.h"
#include "cupdlpx_types.h"
// --- Helper: Build Index Mapping ---
// Creates a map: old_index -> new_index.
// Returns the size of the new dimension.
// map[i] == -1 means index i is removed.
int build_mapping(int old_dim, const bool *mask, int *mapping)
{
    int new_dim = 0;
    for (int i = 0; i < old_dim; ++i)
    {
        if (mask[i])
        {
            mapping[i] = new_dim++;
        }
        else
        {
            mapping[i] = -1;
        }
    }
    return new_dim;
}

// --- 1. CSR Slicing ---
matrix_desc_t slice_csr(const matrix_desc_t *A, const bool *mask_row, const bool *mask_col)
{
    matrix_desc_t B;
    B.fmt = A->fmt; // CSR
    B.zero_tolerance = A->zero_tolerance;

    // 1. Build Map for Columns (CSR needs to re-index column indices)
    int *col_map = (int *)malloc(A->n * sizeof(int));
    B.n = build_mapping(A->n, mask_col, col_map);

    // Calculate new m (rows)
    B.m = 0;
    for (int i = 0; i < A->m; ++i)
        if (mask_row[i])
            B.m++;

    // 2. Count new NNZ
    int new_nnz = 0;
    for (int i = 0; i < A->m; ++i)
    {
        if (!mask_row[i])
            continue;
        for (int k = A->data.csr.row_ptr[i]; k < A->data.csr.row_ptr[i + 1]; ++k)
        {
            if (mask_col[A->data.csr.col_ind[k]])
                new_nnz++;
        }
    }
    B.data.csr.nnz = new_nnz;

    // 3. Allocate
    int *new_row_ptr = (int *)malloc((B.m + 1) * sizeof(int));
    int *new_col_ind = (int *)malloc(new_nnz * sizeof(int));
    double *new_vals = (double *)malloc(new_nnz * sizeof(double));

    // 4. Populate
    int current_nnz = 0;
    int current_row = 0;
    new_row_ptr[0] = 0;

    for (int i = 0; i < A->m; ++i)
    {
        if (!mask_row[i])
            continue;

        for (int k = A->data.csr.row_ptr[i]; k < A->data.csr.row_ptr[i + 1]; ++k)
        {
            int old_col = A->data.csr.col_ind[k];
            if (mask_col[old_col])
            {
                new_vals[current_nnz] = A->data.csr.vals[k];
                new_col_ind[current_nnz] = col_map[old_col];
                current_nnz++;
            }
        }
        current_row++;
        new_row_ptr[current_row] = current_nnz;
    }

    // Assign const pointers
    B.data.csr.row_ptr = new_row_ptr;
    B.data.csr.col_ind = new_col_ind;
    B.data.csr.vals = new_vals;

    free(col_map);
    return B;
}

// --- 2. CSC Slicing ---
matrix_desc_t slice_csc(const matrix_desc_t *A, const bool *mask_row, const bool *mask_col)
{
    matrix_desc_t B;
    B.fmt = A->fmt; // CSC
    B.zero_tolerance = A->zero_tolerance;

    // 1. Build Map for Rows (CSC needs to re-index row indices)
    int *row_map = (int *)malloc(A->m * sizeof(int));
    B.m = build_mapping(A->m, mask_row, row_map);

    // Calculate new n (cols)
    B.n = 0;
    for (int j = 0; j < A->n; ++j)
        if (mask_col[j])
            B.n++;

    // 2. Count new NNZ
    int new_nnz = 0;
    for (int j = 0; j < A->n; ++j)
    {
        if (!mask_col[j])
            continue; // Skip removed columns
        for (int k = A->data.csc.col_ptr[j]; k < A->data.csc.col_ptr[j + 1]; ++k)
        {
            if (mask_row[A->data.csc.row_ind[k]])
                new_nnz++;
        }
    }
    B.data.csc.nnz = new_nnz;

    // 3. Allocate
    int *new_col_ptr = (int *)malloc((B.n + 1) * sizeof(int));
    int *new_row_ind = (int *)malloc(new_nnz * sizeof(int));
    double *new_vals = (double *)malloc(new_nnz * sizeof(double));

    // 4. Populate
    int current_nnz = 0;
    int current_col = 0;
    new_col_ptr[0] = 0;

    for (int j = 0; j < A->n; ++j)
    {
        if (!mask_col[j])
            continue;

        for (int k = A->data.csc.col_ptr[j]; k < A->data.csc.col_ptr[j + 1]; ++k)
        {
            int old_row = A->data.csc.row_ind[k];
            if (mask_row[old_row])
            {
                new_vals[current_nnz] = A->data.csc.vals[k];
                new_row_ind[current_nnz] = row_map[old_row]; // Re-index row
                current_nnz++;
            }
        }
        current_col++;
        new_col_ptr[current_col] = current_nnz;
    }

    B.data.csc.col_ptr = new_col_ptr;
    B.data.csc.row_ind = new_row_ind;
    B.data.csc.vals = new_vals;

    free(row_map);
    return B;
}

// --- 3. COO Slicing ---
matrix_desc_t slice_coo(const matrix_desc_t *A, const bool *mask_row, const bool *mask_col)
{
    matrix_desc_t B;
    B.fmt = A->fmt; // COO
    B.zero_tolerance = A->zero_tolerance;

    int *row_map = (int *)malloc(A->m * sizeof(int));
    int *col_map = (int *)malloc(A->n * sizeof(int));

    B.m = build_mapping(A->m, mask_row, row_map);
    B.n = build_mapping(A->n, mask_col, col_map);

    // Count NNZ
    int new_nnz = 0;
    for (int k = 0; k < A->data.coo.nnz; ++k)
    {
        if (mask_row[A->data.coo.row_ind[k]] && mask_col[A->data.coo.col_ind[k]])
        {
            new_nnz++;
        }
    }
    B.data.coo.nnz = new_nnz;

    // Allocate
    int *new_row_ind = (int *)malloc(new_nnz * sizeof(int));
    int *new_col_ind = (int *)malloc(new_nnz * sizeof(int));
    double *new_vals = (double *)malloc(new_nnz * sizeof(double));

    // Populate
    int current = 0;
    for (int k = 0; k < A->data.coo.nnz; ++k)
    {
        int r = A->data.coo.row_ind[k];
        int c = A->data.coo.col_ind[k];

        if (mask_row[r] && mask_col[c])
        {
            new_row_ind[current] = row_map[r];
            new_col_ind[current] = col_map[c];
            new_vals[current] = A->data.coo.vals[k];
            current++;
        }
    }

    B.data.coo.row_ind = new_row_ind;
    B.data.coo.col_ind = new_col_ind;
    B.data.coo.vals = new_vals;

    free(row_map);
    free(col_map);
    return B;
}

// --- Wrapper Function ---
matrix_desc_t slice_matrix(const matrix_desc_t *A, const bool *mask_row, const bool *mask_col)
{

    if (A->fmt == 1)
    { // Replace with actual enum for CSR
        return slice_csr(A, mask_row, mask_col);
    }
    else if (A->fmt == 2)
    { // Replace with actual enum for CSC
        return slice_csc(A, mask_row, mask_col);
    }
    else if (A->fmt == 3)
    { // Replace with actual enum for COO
        return slice_coo(A, mask_row, mask_col);
    }
    else
    {
        // Handle Dense or Unknown formats
        fprintf(stderr, "Error: Unsupported matrix format for slicing\n");
        matrix_desc_t empty = {};
        return empty;
    }
}

#define ALLOC_AND_COPY(dest, src, bytes)  \
    CUDA_CHECK(cudaMalloc(&dest, bytes)); \
    CUDA_CHECK(cudaMemcpy(dest, src, bytes, cudaMemcpyHostToDevice));

#define ALLOC_ZERO(dest, bytes)           \
    CUDA_CHECK(cudaMalloc(&dest, bytes)); \
    CUDA_CHECK(cudaMemset(dest, 0, bytes));

#define VAR_ACTIVE 0
#define VAR_FIXED_LB 1
#define VAR_FIXED_UB 2
#define VAR_FIXED_MID 3

__global__ void update_status_snap_nearest_kernel(
    int *__restrict__ col_status,
    double *__restrict__ x,
    const bool *__restrict__ mask,
    const double *__restrict__ lb,
    const double *__restrict__ ub,
    const int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        if (mask[i])
        {
            col_status[i] = VAR_ACTIVE;
        }
        else
        {
            double val = x[i];

            // Use INFINITY for missing bounds (requires <math.h>)
            double dist_lb = fabs(val - lb[i]);
            double dist_ub = fabs(val - ub[i]);

            if (dist_lb <= dist_ub)
            {
                x[i] = lb[i];
                col_status[i] = VAR_FIXED_LB;
            }
            else
            {
                x[i] = ub[i];
                col_status[i] = VAR_FIXED_UB;
            }
        }
    }
}

void free_matrix_contents(matrix_desc_t *mat)
{
    if (mat->fmt == 1)
    { // CSR
        if (mat->data.csr.row_ptr)
            free((void *)mat->data.csr.row_ptr);
        if (mat->data.csr.col_ind)
            free((void *)mat->data.csr.col_ind);
        if (mat->data.csr.vals)
            free((void *)mat->data.csr.vals);
    }
    else if (mat->fmt == 2)
    { // CSC
        if (mat->data.csc.col_ptr)
            free((void *)mat->data.csc.col_ptr);
        if (mat->data.csc.row_ind)
            free((void *)mat->data.csc.row_ind);
        if (mat->data.csc.vals)
            free((void *)mat->data.csc.vals);
    }
    else if (mat->fmt == 3)
    { // COO
        if (mat->data.coo.row_ind)
            free((void *)mat->data.coo.row_ind);
        if (mat->data.coo.col_ind)
            free((void *)mat->data.coo.col_ind);
        if (mat->data.coo.vals)
            free((void *)mat->data.coo.vals);
    }
    // Zero out the struct so it's safe if accessed again
    memset(mat, 0, sizeof(matrix_desc_t));
}


static double get_residual(const cupdlpx_result_t *result)
{
    double prim_res = result->relative_primal_residual;
    double dual_res = result->relative_dual_residual;
    double gap = result->relative_objective_gap;
    return fmax(fmax(prim_res, dual_res), gap);
}

__global__ void zero_out_masked_vars_kernel(
    double *x,
    const bool *mask,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        if (mask[idx])
        {
            x[idx] = 0.0;
        }
    }
}

lp_problem_t *construct_sub_lp(
    const lp_problem_t *complete_lp,
    const bool *mask_row, // Host mask
    const bool *mask_col, // Host mask
    const pdhg_solver_state_t *state,
    int *col_status); 

// --------------------------------------------------------------------------------
// Helper: Generate Initial Mask (Phase 0)
// --------------------------------------------------------------------------------
void generate_initial_mask(
    const lp_problem_t *original_problem,
    const pdhg_parameters_t *params,
    double threshold,
    bool verbose,
    // Outputs
    bool **out_mask_col,
    bool **out_mask_row,
    cupdlpx_result_t **out_phase0_result);


static int count_true(const bool *mask, int size)
{
    int count = 0;
    for (int i = 0; i < size; ++i)
    {
        if (mask[i])
            count++;
    }
    return count;
}

void check_violations_gpu(
    const pdhg_solver_state_t *state,
    const bool *mask_row_host, 
    const bool *mask_col_host,
    const int *col_status_gpu, // Already on GPU
    double tol_row_basic,
    double tol_col_basic,
    // Output pointers (Host side)
    int **out_violated_rows, int *out_num_rows,
    int **out_violated_cols, int *out_num_cols);

double* slice_gpu_vector_to_host(
    const double* d_vector, 
    const bool* h_mask, 
    int n, 
    int* out_len);

void scatter_host_to_gpu_masked(
    double *d_full_vec,       // Destination (GPU)
    const double *h_sub_vec,  // Source (Host)
    const bool *h_mask,       // Mask (Host)
    int full_dim);

cupdlpx_result_t *optimize_two_stage(
    const pdhg_parameters_t *params,
    const lp_problem_t *original_problem,
    const double coarse_tol, 
    const double fine_tol, 
    bool verbose){
    if (verbose) {
        printf("\n>>> Stage 1: Solving to coarse tolerance %.1e...\n", coarse_tol);
    }

    pdhg_parameters_t params_stage1 = *params;
    params_stage1.termination_criteria.eps_optimal_relative = coarse_tol;
    params_stage1.termination_criteria.eps_feasible_relative = coarse_tol;
    
    cupdlpx_result_t *result_stage1 = optimize(&params_stage1, original_problem);

    if (!result_stage1) {
        if (verbose) printf(">>> Stage 1 failed to return a result.\n");
        return NULL;
    }

    if (verbose) {
        printf("\n>>> Stage 2: Solving to fine tolerance %.1e using warm start...\n", fine_tol);
    }

    pdhg_parameters_t params_stage2 = *params;
    params_stage2.termination_criteria.eps_optimal_relative = fine_tol;
    params_stage2.termination_criteria.eps_feasible_relative = fine_tol;
    lp_problem_t problem_stage2 = *original_problem;
    params_stage2.init_primal_weight = result_stage1->primal_weight;
    // params_stage2.init_primal_weight_integral = result_stage1->primal_weight_integral;
    // problem_stage2.primal_start = result_stage1->primal_solution;
    // problem_stage2.dual_start   = result_stage1->dual_solution;
    set_start_values(
        &problem_stage2,
        result_stage1->primal_solution,
        result_stage1->dual_solution);

    cupdlpx_result_t *result_stage2 = optimize(&params_stage2, &problem_stage2);

    return result_stage2;
}

void cupdlpx_copy_solution_stats(cupdlpx_result_t *dest, const cupdlpx_result_t *src) {
    if (dest == NULL || src == NULL) {
        return;
    }

    // --- Residuals ---
    dest->absolute_primal_residual = src->absolute_primal_residual;
    dest->relative_primal_residual = src->relative_primal_residual;
    dest->absolute_dual_residual   = src->absolute_dual_residual;
    dest->relative_dual_residual   = src->relative_dual_residual;

    // --- Objectives ---
    dest->primal_objective_value = src->primal_objective_value;
    dest->dual_objective_value   = src->dual_objective_value;
    dest->objective_gap          = src->objective_gap;
    dest->relative_objective_gap = src->relative_objective_gap;

    // --- Ray / Infeasibility Info ---
    dest->max_primal_ray_infeasibility = src->max_primal_ray_infeasibility;
    dest->max_dual_ray_infeasibility   = src->max_dual_ray_infeasibility;
    dest->primal_ray_linear_objective  = src->primal_ray_linear_objective;
    dest->dual_ray_objective           = src->dual_ray_objective;

    // --- Termination Reason ---
    dest->termination_reason = src->termination_reason;

    // --- Weights (Useful for restarts) ---
    dest->primal_weight          = src->primal_weight;
    dest->primal_weight_integral = src->primal_weight_integral;
}

cupdlpx_result_t *optimize_with_adaptive_active_identify(
    const pdhg_parameters_t *params,
    const lp_problem_t *original_problem,
    const int max_adaptive_iteration,
    const double init_mask_threshold,
    const double tol,
    const double time_limit,
    bool verbose)
{
    // ------------------------------------------------------------------
    // 1. SETUP & PHASE 0
    // ------------------------------------------------------------------

    // Create Non-Rescaling Params
    pdhg_parameters_t *non_rescale_params = new pdhg_parameters_t;
    set_default_parameters(non_rescale_params);
    non_rescale_params->l_inf_ruiz_iterations = 0;
    non_rescale_params->has_pock_chambolle_alpha = false;
    non_rescale_params->bound_objective_rescaling = false;
    print_initial_info(params, original_problem);

    // Stats Struct
    adaptive_active_identify_stats_t *stats =
        (adaptive_active_identify_stats_t *)safe_calloc(1, sizeof(adaptive_active_identify_stats_t));
    stats->orig_n = original_problem->num_variables;
    stats->orig_m = original_problem->num_constraints;

    // Initialize Solver State
    rescale_info_t *rescale_info = rescale_problem(non_rescale_params, original_problem);
    pdhg_solver_state_t *complete_state = initialize_solver_state(original_problem, rescale_info);
    
    // [CLEANUP] Free rescale info immediately after use
    rescale_info_free(rescale_info); 
    // [CLEANUP] Free the temp params
    delete non_rescale_params; 

    // Phase 0: Initial Mask Generation
    bool *mask_col = NULL;
    bool *mask_row = NULL;
    cupdlpx_result_t *main_result = NULL; // Renamed from phase0_result for clarity
    
    generate_initial_mask(
        original_problem,
        params,
        init_mask_threshold,
        verbose,
        &mask_col,
        &mask_row,
        &main_result);

    if (main_result->termination_reason == TERMINATION_REASON_DUAL_INFEASIBLE || main_result->termination_reason == TERMINATION_REASON_PRIMAL_INFEASIBLE)
    {    
        printf(">>> Stopping early (Original Problem Infeasible or Unbouned)\n");
        return main_result;
    }

    stats->init_reduced_n = count_true(mask_col, original_problem->num_variables);
    stats->init_reduced_m = count_true(mask_row, original_problem->num_constraints);
    stats->phase0_iter = main_result->total_count;
    stats->phase0_time = main_result->cumulative_time_sec;

    // Sync Phase 0 result to GPU State
    CUDA_CHECK(cudaMemcpy(
        complete_state->pdhg_primal_solution,
        main_result->primal_solution,
        original_problem->num_variables * sizeof(double),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        complete_state->pdhg_dual_solution,
        main_result->dual_solution,
        original_problem->num_constraints * sizeof(double),
        cudaMemcpyHostToDevice));

    // Allocate GPU Snapshots
    double *start_x_full = NULL;
    ALLOC_AND_COPY(start_x_full, complete_state->pdhg_primal_solution,
                   original_problem->num_variables * sizeof(double));
    double *start_y_full = NULL;
    ALLOC_AND_COPY(start_y_full, complete_state->pdhg_dual_solution,
                   original_problem->num_constraints * sizeof(double));
    
    // Prepare Column Status Array (GPU)
    int *status_col_gpu = NULL;
    ALLOC_ZERO(status_col_gpu, original_problem->num_variables * sizeof(int));

    // Prepare Sub-LP Params
    pdhg_parameters_t *params_for_sublp = new pdhg_parameters_t(*params);
    set_default_parameters(params_for_sublp);
    params_for_sublp->termination_criteria.eps_optimal_relative = tol;
    params_for_sublp->termination_criteria.eps_feasible_relative = tol;
    params_for_sublp->termination_criteria.iteration_limit = 100000;
    params_for_sublp->verbose = false;

    double last_residual = get_residual(main_result);
    double best_residual = last_residual;
    double init_primal_weight = main_result->primal_weight;
    double init_primal_weight_integral = main_result->primal_weight_integral;

    // ------------------------------------------------------------------
    // 2. ADAPTIVE IDENTIFICATION LOOP
    // ------------------------------------------------------------------
    double obtain_global_optimal = false;
    int no_prograss_time = 0;
    for(int iter = 0; iter < max_adaptive_iteration; ++iter)
    {
        stats->adaptive_loops = iter;
        
        int n_sub;
        double *x_start = slice_gpu_vector_to_host(
            start_x_full, mask_col, original_problem->num_variables, &n_sub);
        int m_sub;
        double *y_start = slice_gpu_vector_to_host(
            start_y_full, mask_row, original_problem->num_constraints, &m_sub);
        
        // --- Early Stop Check: Sub-LP too big ---
        if (n_sub > 0.95 * original_problem->num_variables && 
            m_sub > 0.95 * original_problem->num_constraints)
        {
            if (verbose) printf(">>> Stopping early (Sub-LP too large)\n");
            
            for (int i = 0; i < original_problem->num_variables; ++i) mask_col[i] = true;
            for (int i = 0; i < original_problem->num_constraints; ++i) mask_row[i] = true;
            
            // [CLEANUP] Free slice vectors
            free(x_start);
            free(y_start);
            break;
        }

        if (verbose) printf(">>> Adaptive Iteration %d / %d\n", iter + 1, max_adaptive_iteration);

        // Construct & Solve Sub-LP
        lp_problem_t *sub_lp = construct_sub_lp(
            original_problem, mask_row, mask_col, complete_state, status_col_gpu);
        
        set_start_values(sub_lp, x_start, y_start);
        print_initial_info(params, sub_lp);

        params_for_sublp->init_primal_weight = init_primal_weight;
        // params_for_sublp->init_primal_weight_integral = init_primal_weight_integral;

        cupdlpx_result_t *sub_result = optimize(params_for_sublp, sub_lp);

        cupdlpx_copy_solution_stats(main_result, sub_result);
        // Update Stats
        stats->adaptive_iter_sum += sub_result->total_count;
        stats->adaptive_time_sum += sub_result->cumulative_time_sec;
        if (stats->adaptive_time_sum + stats->phase0_time >= time_limit) {
            if (verbose) printf(">>> Time limit reached during adaptive phase.\n");
            break;
        }
        
        
        double residual = get_residual(sub_result);

        // Scatter Solution to Full GPU State
        scatter_host_to_gpu_masked(
            complete_state->pdhg_primal_solution,
            sub_result->primal_solution,
            mask_col,
            original_problem->num_variables);
        
        CUDA_CHECK(cudaMemset(complete_state->pdhg_dual_solution, 0, original_problem->num_constraints * sizeof(double)));
        
        scatter_host_to_gpu_masked(
            complete_state->pdhg_dual_solution,
            sub_result->dual_solution,
            mask_row,
            original_problem->num_constraints);

        // --- Check Global Violations ---
        int *violated_rows = NULL;
        int num_violated_rows = 0;
        int *violated_cols = NULL;
        int num_violated_cols = 0;
        // double tol_row_basic = tol;
        // double tol_col_basic = tol;
        if (residual >= 0.5 * last_residual){
            no_prograss_time += 1;
            // tol_row_basic *= pow(0.1, no_prograss_time);
            // tol_col_basic *= pow(0.1, no_prograss_time);
            
        }

        if (no_prograss_time >= 5) {
            for (int i = 0; i < original_problem->num_constraints; ++i) mask_row[i] = true; 
            for (int i = 0; i < original_problem->num_variables; ++i) mask_col[i] = true;
            printf("No prograss for 5 iterations, adding all rows and columns to the sub-LP.\n");
            break;
        }

        check_violations_gpu(
            complete_state, mask_row, mask_col, status_col_gpu, tol, tol,
            &violated_rows, &num_violated_rows,
            &violated_cols, &num_violated_cols);

        bool should_break = false;
        
        if (num_violated_rows == 0 && num_violated_cols == 0) {
            printf("   >>> No Global Violations detected.\n");
            if (sub_result->termination_reason == TERMINATION_REASON_OPTIMAL) {
                if (verbose) printf(">>> Stopping early (Optimal & Valid)\n");
                should_break = true;
                obtain_global_optimal = true;
            }
        } else {
            printf("   >>> Global Violations: +%d rows, +%d cols.\n", num_violated_rows, num_violated_cols);
        }

        // Logic to update Sub-LP or Skip
        if (!should_break) {
            if (residual < 0.5 * last_residual && sub_result->termination_reason != TERMINATION_REASON_OPTIMAL) {
                printf("   Sub-LP Improved (%.3e -> %.3e), skipping mask update.\n", last_residual, residual);
                
                if (residual < best_residual) {
                    best_residual = residual;
                    CUDA_CHECK(cudaMemcpy(start_x_full, complete_state->pdhg_primal_solution,
                                      original_problem->num_variables * sizeof(double), cudaMemcpyDeviceToDevice));
                    CUDA_CHECK(cudaMemcpy(start_y_full, complete_state->pdhg_dual_solution,
                                        original_problem->num_constraints * sizeof(double), cudaMemcpyDeviceToDevice));
                    init_primal_weight = sub_result->primal_weight;
                    init_primal_weight_integral = sub_result->primal_weight_integral;
                    }
                no_prograss_time = 0;
            } 
            // else {
                printf("   Updating Mask with violations.\n");
                for (int i = 0; i < num_violated_rows; ++i) mask_row[violated_rows[i]] = true;
                for (int i = 0; i < num_violated_cols; ++i) mask_col[violated_cols[i]] = true;
            // }
        }
        
        last_residual = residual;

        // ------------------------------------------------------------------
        // CLEANUP: Loop Iteration Objects
        // ------------------------------------------------------------------
        free(x_start);
        free(y_start);
        if (violated_rows) free(violated_rows);
        if (violated_cols) free(violated_cols);
        
        // Free Sub-LP and Result (Assumes library provides these or standard free)
        // If these are structs with internal pointers, use specific free func.
        // Assuming typical usage:
        lp_problem_free(sub_lp);        // Placeholder: Replace with actual free function
        cupdlpx_result_free(sub_result); // Placeholder: Replace with actual free function
        
        if (should_break) break;
    }

    if (verbose) {
        printf("\n>>> Adaptive Phase Completed. Entering Final Solve.\n");
    }

    // ------------------------------------------------------------------
    // 3. FINAL SOLVE
    // ------------------------------------------------------------------
    int n_sub;
    double *x_start = slice_gpu_vector_to_host(start_x_full, mask_col, original_problem->num_variables, &n_sub);
    int m_sub;
    double *y_start = slice_gpu_vector_to_host(start_y_full, mask_row, original_problem->num_constraints, &m_sub);

    lp_problem_t *final_sub_lp = construct_sub_lp(
        original_problem, mask_row, mask_col, complete_state, status_col_gpu);
    
    set_start_values(final_sub_lp, x_start, y_start);

    pdhg_parameters_t *final_params = new pdhg_parameters_t;
    set_default_parameters(final_params);
    final_params->verbose = true;
    final_params->termination_criteria.eps_optimal_relative = tol;
    final_params->termination_criteria.eps_feasible_relative = tol;
    final_params->init_primal_weight = init_primal_weight;
    final_params->init_primal_weight_integral = init_primal_weight_integral;
    final_params->termination_criteria.time_sec_limit = time_limit - (stats->phase0_time + stats->adaptive_time_sum);
        
    if (obtain_global_optimal) {
        final_params->termination_criteria.iteration_limit = 0; // No iterations needed
    }
    cupdlpx_result_t *final_result = optimize(final_params, final_sub_lp);

    if (!obtain_global_optimal) {
        cupdlpx_copy_solution_stats(main_result, final_result);
    }
    stats->final_iter = final_result->total_count;
    stats->final_time = final_result->cumulative_time_sec;

    // Update Main Result Stats
    main_result->cumulative_time_sec = stats->phase0_time + stats->adaptive_time_sum + stats->final_time;
    main_result->total_count = stats->phase0_iter + stats->adaptive_iter_sum + stats->final_iter;

    // ------------------------------------------------------------------
    // 4. SOLUTION TRANSFER (Reduced -> Full)
    // ------------------------------------------------------------------
    // We must map the reduced solution in final_result back to the main_result.
    
    // Step A: reduced CPU -> full GPU state
    scatter_host_to_gpu_masked(
        complete_state->pdhg_primal_solution,
        final_result->primal_solution,
        mask_col,
        original_problem->num_variables);
    
    CUDA_CHECK(cudaMemset(complete_state->pdhg_dual_solution, 0, original_problem->num_constraints * sizeof(double)));
    scatter_host_to_gpu_masked(
        complete_state->pdhg_dual_solution,
        final_result->dual_solution,
        mask_row,
        original_problem->num_constraints);

    // Step B: full GPU state -> full CPU (main_result)
    CUDA_CHECK(cudaMemcpy(
        main_result->primal_solution,
        complete_state->pdhg_primal_solution,
        original_problem->num_variables * sizeof(double),
        cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaMemcpy(
        main_result->dual_solution,
        complete_state->pdhg_dual_solution,
        original_problem->num_constraints * sizeof(double),
        cudaMemcpyDeviceToHost));

    // ------------------------------------------------------------------
    // 5. GLOBAL CLEANUP
    // ------------------------------------------------------------------
    // Free Final Phase Objects
    free(x_start);
    free(y_start);
    delete final_params;
    lp_problem_free(final_sub_lp);     // Placeholder
    cupdlpx_result_free(final_result); // Placeholder

    // Free Helper Objects
    delete params_for_sublp;
    free(stats);

    // Free Arrays
    free(mask_col);
    free(mask_row);

    // Free GPU Memory
    cudaFree(start_x_full);
    cudaFree(start_y_full);
    cudaFree(status_col_gpu);
    
    // Free Solver State (Assuming this function cleans internal GPU pointers)
    pdhg_solver_state_free(complete_state); // Placeholder

    return main_result;
}

void generate_initial_mask(
    const lp_problem_t *original_problem,
    const pdhg_parameters_t *params,
    double threshold,
    bool verbose,
    // Outputs
    bool **out_mask_col,
    bool **out_mask_row,
    cupdlpx_result_t **out_phase0_result)
{
    if (verbose)
    {
        printf(">>> Phase 0: Running Initial Identification (Statistics based)...\n");
    }

    // 1. Deep Copy Options & Modify for Phase 0
    pdhg_parameters_t opt_init = *params;

    // Ensure we capture active times statistics (if your solver requires a flag)
    // opt_init.record_active_times = true; // Example flag if needed

    // 2. Run Solver (Phase 0)
    // Returns result with solution and stats on Host
    cupdlpx_result_t *solver_res = optimize(&opt_init, original_problem);

    if (!solver_res)
    {
        fprintf(stderr, "Error: Phase 0 optimization failed (returned NULL).\n");
        return;
    }
    else
    {
        if (verbose)
        {
            printf("   Phase 0 completed. Solver returned results.\n");
        }
    }

    // 3. Get Statistics
    int n = solver_res->num_variables;
    int m = solver_res->num_constraints;
    int total_iter = solver_res->total_count;

    // 4. Determine Effective Threshold
    double effective_threshold = threshold;

    // Safety check: if iterations are too low, statistics are unreliable
    if (total_iter < 10000)
    {
        if (verbose)
        {
            printf("   Warning: Iterations too low (%d), skipping removal.\n", total_iter);
        }
        // Set threshold high enough that nothing is removed
        effective_threshold = 100000.0;
    }

    // 5. Generate Masks
    // Allocate Host boolean arrays
    bool *mask_col = (bool *)calloc(n, sizeof(bool));
    bool *mask_row = (bool *)calloc(m, sizeof(bool));

    int rm_col_count = 0;
    int rm_row_count = 0;
    double limit = effective_threshold * (double)(solver_res->update_acitve_times + 1);

    // Columns (Variables) - using primal_active_times
    if (solver_res->primal_active_times)
    {
        for (int i = 0; i < n; ++i)
        {
            // If active_time > limit, we remove it (mask = false)
            // Otherwise we keep it (mask = true)
            bool remove = (double)solver_res->primal_active_times[i] > limit;
            mask_col[i] = !remove;
            if (remove)
                rm_col_count++;
        }
    }
    else
    {
        // Fallback if stats missing: keep all
        for (int i = 0; i < n; ++i)
            mask_col[i] = true;
        if (verbose)
            printf("   Warning: primal_active_times is NULL.\n");
    }

    // Rows (Constraints) - using dual_active_times (inactive times)
    if (solver_res->dual_active_times)
    {
        for (int i = 0; i < m; ++i)
        {
            // Note: Julia code used "constrs_inactive_times".
            bool remove = (double)solver_res->dual_active_times[i] > limit;
            // remove = false;
            mask_row[i] = !remove;
            if (remove)
                rm_row_count++;
        }
    }
    else
    {
        // Fallback
        for (int i = 0; i < m; ++i)
            mask_row[i] = true;
    }

    if (verbose)
    {
        printf("   Stats: Total Iter = %d\n", total_iter);
        printf("   Removed Cols (Fixed): %d / %d\n", rm_col_count, n);
        printf("   Removed Rows (Slack): %d / %d\n", rm_row_count, m);
    }

    // 6. Set Outputs
    *out_mask_col = mask_col;
    *out_mask_row = mask_row;
    *out_phase0_result = solver_res;
}

lp_problem_t *construct_sub_lp(
    const lp_problem_t *complete_lp,
    const bool *mask_row, // Host mask
    const bool *mask_col, // Host mask
    const pdhg_solver_state_t *state,
    int *col_status) // GPU pointer for status
{
    int n = complete_lp->num_variables;
    int m = complete_lp->num_constraints;

    // ---------------------------------------------------------
    // 1. GPU: Update Status and Prepare x_fixed_only
    // ---------------------------------------------------------

    // Allocate and copy mask to GPU
    // Note: Using bool* to match host type, assuming 1 byte per bool
    bool *mask_gpu_col;
    CUDA_CHECK(cudaMalloc((void **)&mask_gpu_col, n * sizeof(bool)));
    CUDA_CHECK(cudaMemcpy(mask_gpu_col, mask_col, n * sizeof(bool), cudaMemcpyHostToDevice));

    // Run the user-provided kernel to snap active variables
    update_status_snap_nearest_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
        col_status,
        state->pdhg_primal_solution,
        mask_gpu_col, // Casting if kernel expects int*, but typically bool* preferred
        state->variable_lower_bound,
        state->variable_upper_bound,
        n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaGetLastError()); // Check launch
    CUDA_CHECK(cudaDeviceSynchronize()); // Check execution <--- ADD THIS
    // Create x_fixed_only (Copy of primal solution)
    double *x_fixed_only;
    CUDA_CHECK(cudaMalloc((void **)&x_fixed_only, n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(x_fixed_only, state->pdhg_primal_solution, n * sizeof(double), cudaMemcpyDeviceToDevice));

    // Zero out active variables in x_fixed_only
    // Julia: x_fixed_only[mask_col_gpu] .= 0.0
    zero_out_masked_vars_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
        x_fixed_only,
        mask_gpu_col,
        n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaGetLastError()); // Check launch
    CUDA_CHECK(cudaDeviceSynchronize()); // Check execution <--- ADD THIS
    // ---------------------------------------------------------
    // 2. GPU: Compute Offset = A * x_fixed_only
    // ---------------------------------------------------------

    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_primal_sol, x_fixed_only));
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_primal_prod, state->primal_product));

    double *offset_gpu = state->primal_product; // Reuse primal_product buffer for offset

    CUSPARSE_CHECK(cusparseSpMV(
        state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &HOST_ONE, state->matA, state->vec_primal_sol, &HOST_ZERO, state->vec_primal_prod,
        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->primal_spmv_buffer));

    // ---------------------------------------------------------
    // 3. Data Transfer: Offset to Host
    // ---------------------------------------------------------

    double *offset_cpu = (double *)malloc(m * sizeof(double));
    CUDA_CHECK(cudaMemcpy(offset_cpu, offset_gpu, m * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(x_fixed_only));
    CUDA_CHECK(cudaFree(mask_gpu_col)); // Done with GPU mask

    // ---------------------------------------------------------
    // 4. CPU: Slice and Construct Sub-LP
    // ---------------------------------------------------------

    lp_problem_t *sub_lp = (lp_problem_t *)calloc(1, sizeof(lp_problem_t));

    // Count new dimensions
    int n_sub = 0;
    for (int i = 0; i < n; i++)
        if (mask_col[i])
            n_sub++;

    int m_sub = 0;
    for (int i = 0; i < m; i++)
        if (mask_row[i])
            m_sub++;

    sub_lp->num_variables = n_sub;
    sub_lp->num_constraints = m_sub;

    // Allocate Sub-LP arrays
    sub_lp->objective_vector = (double *)malloc(n_sub * sizeof(double));
    sub_lp->variable_lower_bound = (double *)malloc(n_sub * sizeof(double));
    sub_lp->variable_upper_bound = (double *)malloc(n_sub * sizeof(double));

    sub_lp->constraint_lower_bound = (double *)malloc(m_sub * sizeof(double));
    sub_lp->constraint_upper_bound = (double *)malloc(m_sub * sizeof(double));

    // Fill Column-based data (Variables)
    int idx_sub = 0;
    for (int i = 0; i < n; i++)
    {
        if (mask_col[i])
        {
            sub_lp->objective_vector[idx_sub] = complete_lp->objective_vector[i];
            sub_lp->variable_lower_bound[idx_sub] = complete_lp->variable_lower_bound[i];
            sub_lp->variable_upper_bound[idx_sub] = complete_lp->variable_upper_bound[i];
            idx_sub++;
        }
    }

    // Fill Row-based data (Constraints) - Applying OFFSET here
    idx_sub = 0;
    for (int i = 0; i < m; i++)
    {
        if (mask_row[i])
        {
            // Julia: AL_sub = cpu_qp.AL[mask_row] .- offset_sub
            sub_lp->constraint_lower_bound[idx_sub] = complete_lp->constraint_lower_bound[i] - offset_cpu[i];
            sub_lp->constraint_upper_bound[idx_sub] = complete_lp->constraint_upper_bound[i] - offset_cpu[i];
            idx_sub++;
        }
    }

    matrix_desc_t A_desc;
    A_desc.m = m;
    A_desc.n = n;

    // Assuming your enum for CSR is defined (e.g., MATRIX_FORMAT_CSR).
    // If you use raw integers (1, 2, 3), replace this accordingly.
    A_desc.fmt = matrix_csr;
    A_desc.zero_tolerance = 0.0; // Set appropriate tolerance if needed

    // Populate the Union specifically for CSR
    A_desc.data.csr.nnz = complete_lp->constraint_matrix_num_nonzeros;
    A_desc.data.csr.row_ptr = complete_lp->constraint_matrix_row_pointers;
    A_desc.data.csr.col_ind = complete_lp->constraint_matrix_col_indices;
    A_desc.data.csr.vals = complete_lp->constraint_matrix_values;

    matrix_desc_t A_sub_desc = slice_matrix(&A_desc, mask_row, mask_col);

    // 2. Transfer ownership of the malloc'd arrays to the LP struct
    if (A_sub_desc.fmt == 1)
    { // CSR
        // We simply copy the POINTERS. The arrays themselves are not copied again.
        sub_lp->constraint_matrix_row_pointers = (int *)A_sub_desc.data.csr.row_ptr;
        sub_lp->constraint_matrix_col_indices = (int *)A_sub_desc.data.csr.col_ind;
        sub_lp->constraint_matrix_values = (double *)A_sub_desc.data.csr.vals;
        sub_lp->constraint_matrix_num_nonzeros = A_sub_desc.data.csr.nnz;
    }
    // Clean up Host temp
    free(offset_cpu);

    return sub_lp;
}

__global__ void check_row_violations_kernel(
    const double *Ax,
    const double *AL,
    const double *AU,
    const bool *mask_row,
    const double tol_row,
    int *violated_rows,
    int *count,
    int m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m)
    {
        // Julia: (!mask_row) & ...
        if (!mask_row[i])
        {
            double val = Ax[i];
            bool vio = false;

            if (val < AL[i] - tol_row)
            {
                vio = true;
            }
            else if (val > AU[i] + tol_row)
            {
                vio = true;
            }

            if (vio)
            {
                // Atomic append to list
                int pos = atomicAdd(count, 1);
                violated_rows[pos] = i;
            }
        }
    }
}

// Kernel to check column violations (Reduced Costs)
// Logic: Compute grad = c - A'y on the fly, then check status.
__global__ void check_col_violations_kernel(
    const double *c,
    const double *Aty, // A'y (dual product)
    const int *col_status,
    const bool *mask_col,
    const double tol_col,
    int *violated_cols,
    int *count,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        // Gradient = c - A'y
        if (mask_col[i]) {
            return; // Skip active variables
        }
        double grad = c[i] - Aty[i];
        int status = col_status[i];
        bool vio = false;
        
        if (status == VAR_FIXED_LB && grad < -tol_col) {
            vio = true;
        } else if (status == VAR_FIXED_UB && grad > tol_col) {
            vio = true;
        } else if (status == VAR_FIXED_MID && fabs(grad) > tol_col) {
            vio = true;
        }

        if (vio)
        {
            int pos = atomicAdd(count, 1);
            violated_cols[pos] = i;
        }
    }
}

// --- Main Function ---

// Inputs: state, masks, tolerance
// Outputs: Populates arrays of indices and counts (allocated by caller or inside)
void check_violations_gpu(
    const pdhg_solver_state_t *state,
    const bool *mask_row_host, 
    const bool *mask_col_host,
    const int *col_status_gpu,// Already on GPU
    double tol_row_basic,
    double tol_col_basic,
    // Output pointers (Host side)
    int **out_violated_rows, int *out_num_rows,
    int **out_violated_cols, int *out_num_cols)
{
    int n = state->num_variables;
    int m = state->num_constraints;

    // ---------------------------------------------------------
    // 1. Setup GPU Resources
    // ---------------------------------------------------------

    // Allocate and copy Mask Row to GPU
    bool *d_mask_row;
    CUDA_CHECK(cudaMalloc((void**)&d_mask_row, m * sizeof(bool)));
    CUDA_CHECK(cudaMemcpy(d_mask_row, mask_row_host, m * sizeof(bool), cudaMemcpyHostToDevice));
    bool *d_mask_col;
    CUDA_CHECK(cudaMalloc((void**)&d_mask_col, n * sizeof(bool))); // Use 'n'
    CUDA_CHECK(cudaMemcpy(d_mask_col, mask_col_host, n * sizeof(bool), cudaMemcpyHostToDevice)); // Copy 'mask_col_host'


    // Allocate counters on GPU (initialized to 0)
    int *d_count_row, *d_count_col;
    CUDA_CHECK(cudaMalloc((void**)&d_count_row, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_count_col, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_count_row, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_count_col, 0, sizeof(int)));

    // Allocate Output Buffers on GPU (Worst case size M and N)
    int *d_violated_rows, *d_violated_cols;
    CUDA_CHECK(cudaMalloc((void**)&d_violated_rows, m * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_violated_cols, n * sizeof(int)));

    // ---------------------------------------------------------
    // 2. Compute Ax and A'y (Sparse Matrix Vector Mult)
    // ---------------------------------------------------------
    
    // We reuse the buffers in 'state' to avoid new allocations.
    // 2a. Ax (Primal Product)
    //     Ax = A * x_full
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_primal_sol, state->pdhg_primal_solution));
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_primal_prod, state->primal_product));
    
    CUSPARSE_CHECK(cusparseSpMV(
        state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &HOST_ONE, state->matA, state->vec_primal_sol, &HOST_ZERO, state->vec_primal_prod,
        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->primal_spmv_buffer));

    // 2b. A'y (Dual Product)
    //     A'y = A^T * y_full
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_dual_sol, state->pdhg_dual_solution));
    CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_dual_prod, state->dual_product));
    
    // Note: Assuming state->matA is the matrix. 
    // If you have explicit matAt (transpose), use that with NON_TRANSPOSE.
    // Otherwise use matA with TRANSPOSE.
    CUSPARSE_CHECK(cusparseSpMV(
        state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &HOST_ONE, state->matAt, state->vec_dual_sol, &HOST_ZERO, state->vec_dual_prod,
        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->dual_spmv_buffer));

    // ---------------------------------------------------------
    // 3. Compute Tolerances (Norms)
    // ---------------------------------------------------------
    // tol_row = max(norm(AL), norm(AU)) * tol
    // tol_col = norm(c) * tol
    double tol_row = (1.0 + state->constraint_bound_norm) * tol_row_basic;
    double tol_col = (1.0 + state->objective_vector_norm) * tol_col_basic;
    // ---------------------------------------------------------
    // 4. Run Check Kernels
    // ---------------------------------------------------------

    // Row Check
    check_row_violations_kernel<<<state->num_blocks_primal, THREADS_PER_BLOCK>>>(
        state->primal_product,       
        state->constraint_lower_bound,
        state->constraint_upper_bound,
        d_mask_row,
        tol_row,
        d_violated_rows,
        d_count_row,
        m
    );
    CUDA_CHECK(cudaGetLastError());

    // Column Check
    check_col_violations_kernel<<<state->num_blocks_dual, THREADS_PER_BLOCK>>>(
        state->objective_vector,    
        state->dual_product,       
        col_status_gpu,
        d_mask_col,
        tol_col,
        d_violated_cols,
        d_count_col,
        n
    );
    CUDA_CHECK(cudaGetLastError());

    // ---------------------------------------------------------
    // 5. Retrieve Results to Host
    // ---------------------------------------------------------
    
    int h_count_row = 0;
    int h_count_col = 0;
    CUDA_CHECK(cudaMemcpy(&h_count_row, d_count_row, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_count_col, d_count_col, sizeof(int), cudaMemcpyDeviceToHost));

    // Allocate Host arrays for result
    int *h_violated_rows = (int*)malloc(h_count_row * sizeof(int));
    int *h_violated_cols = (int*)malloc(h_count_col * sizeof(int));

    if (h_count_row > 0) {
        CUDA_CHECK(cudaMemcpy(h_violated_rows, d_violated_rows, h_count_row * sizeof(int), cudaMemcpyDeviceToHost));
    }
    if (h_count_col > 0) {
        CUDA_CHECK(cudaMemcpy(h_violated_cols, d_violated_cols, h_count_col * sizeof(int), cudaMemcpyDeviceToHost));
    }

    // Set Outputs
    *out_num_rows = h_count_row;
    *out_num_cols = h_count_col;
    *out_violated_rows = h_violated_rows;
    *out_violated_cols = h_violated_cols;

    // ---------------------------------------------------------
    // 6. Cleanup
    // ---------------------------------------------------------
    CUDA_CHECK(cudaFree(d_mask_row));
    CUDA_CHECK(cudaFree(d_count_row));
    CUDA_CHECK(cudaFree(d_count_col));
    CUDA_CHECK(cudaFree(d_violated_rows));
    CUDA_CHECK(cudaFree(d_violated_cols));
}

double* slice_gpu_vector_to_host(
    const double* d_vector, 
    const bool* h_mask, 
    int n, 
    int* out_len)
{
    // 1. Calculate the size of the new sub-vector
    int new_len = count_true(h_mask, n);
    
    // Update output length
    if (out_len) *out_len = new_len;

    if (new_len == 0) {
        return NULL;
    }

    // 2. Allocate Result Array (CPU)
    double* h_result = (double*)malloc(new_len * sizeof(double));

    // 3. Allocate Temporary Buffer for full vector (CPU)
    double* h_temp_full = (double*)malloc(n * sizeof(double));
    if (!h_temp_full) {
        fprintf(stderr, "Error: Failed to allocate temp buffer for slicing.\n");
        free(h_result);
        return NULL;
    }

    // 4. Transfer full data from GPU to Temp CPU buffer
    //    (One large transfer is usually faster than many small/indexed transfers)
    cudaError_t err = cudaMemcpy(h_temp_full, d_vector, n * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: cudaMemcpy failed in slice_gpu_vector_to_host: %s\n", cudaGetErrorString(err));
        free(h_result);
        free(h_temp_full);
        return NULL;
    }

    // 5. Filter data on CPU
    int current_idx = 0;
    for (int i = 0; i < n; i++) {
        if (h_mask[i]) {
            h_result[current_idx] = h_temp_full[i];
            current_idx++;
        }
    }

    // 6. Cleanup
    free(h_temp_full);

    return h_result;
}

__global__ void scatter_kernel(
    double *full_vec,
    const int *indices,
    const double *sub_vec,
    int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
    {
        full_vec[indices[i]] = sub_vec[i];
    }
}

void scatter_host_to_gpu_masked(
    double *d_full_vec,       // Destination (GPU)
    const double *h_sub_vec,  // Source (Host)
    const bool *h_mask,       // Mask (Host)
    int full_dim)
{
    // 1. Calculate size and build Index Map on CPU
    int sub_count = 0;
    for (int i = 0; i < full_dim; i++) {
        if (h_mask[i]) sub_count++;
    }

    if (sub_count == 0) return;

    // Allocate temp Host index array
    int *h_indices = (int *)malloc(sub_count * sizeof(int));
    
    // Fill indices: mapping packed sub_vec[k] -> full_vec[i]
    int k = 0;
    for (int i = 0; i < full_dim; i++) {
        if (h_mask[i]) {
            h_indices[k++] = i;
        }
    }

    // 2. Allocate temp Device buffers
    int *d_indices;
    double *d_sub_vec;
    CUDA_CHECK(cudaMalloc((void**)&d_indices, sub_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_sub_vec, sub_count * sizeof(double)));

    // 3. Copy Data and Indices to GPU
    CUDA_CHECK(cudaMemcpy(d_indices, h_indices, sub_count * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sub_vec, h_sub_vec, sub_count * sizeof(double), cudaMemcpyHostToDevice));

    // 4. Launch Kernel
    int threads = 256;
    int blocks = (sub_count + threads - 1) / threads;
    
    scatter_kernel<<<blocks, threads>>>(d_full_vec, d_indices, d_sub_vec, sub_count);
    CUDA_CHECK(cudaGetLastError());

    // 5. Cleanup
    free(h_indices);
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_sub_vec));
}