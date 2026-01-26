#include "iterative_refinement.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "utils.h"
#include "preconditioner.h"
#include "solver.h"
#include "cupdlpx_types.h"
double round_to_pow2(double v) {
    if (v == 0.0) return 1.0;
    return pow(2.0, ceil(log2(fabs(v))));
}

// --- Helper: Construct Sub-LP (Same as before) ---
lp_problem_t *construct_refinement_lp_with_slacks(
    const lp_problem_t *orig,
    const double *y_k,
    double delta_P,
    double delta_D,
    const double *res_c,
    const double *shift_l,
    const double *shift_u,
    const double *shift_L,
    const double *shift_U
) {
    int n = orig->num_variables;
    int m = orig->num_constraints;
    int n_sub = n + m; 

    lp_problem_t *sub_lp = (lp_problem_t *)calloc(1, sizeof(lp_problem_t));
    sub_lp->num_variables = n_sub;
    sub_lp->num_constraints = m;

    // 1. Objective
    sub_lp->objective_vector = (double *)malloc(n_sub * sizeof(double));
    for (int i = 0; i < n; i++) sub_lp->objective_vector[i] = res_c[i] * delta_D;
    for (int i = 0; i < m; i++) sub_lp->objective_vector[n + i] = y_k[i] * delta_D;

    // 2. Variable Bounds
    sub_lp->variable_lower_bound = (double *)malloc(n_sub * sizeof(double));
    sub_lp->variable_upper_bound = (double *)malloc(n_sub * sizeof(double));

    for (int i = 0; i < n; i++) {
        sub_lp->variable_lower_bound[i] = shift_l[i] * delta_P;
        sub_lp->variable_upper_bound[i] = shift_u[i] * delta_P;
    }
    for (int i = 0; i < m; i++) {
        sub_lp->variable_lower_bound[n + i] = shift_L[i] * delta_P;
        sub_lp->variable_upper_bound[n + i] = shift_U[i] * delta_P;
    }

    // 3. Matrix [A, -I]
    int new_nnz = orig->constraint_matrix_num_nonzeros + m;
    sub_lp->constraint_matrix_num_nonzeros = new_nnz;
    sub_lp->constraint_matrix_row_pointers = (int *)malloc((m + 1) * sizeof(int));
    sub_lp->constraint_matrix_col_indices = (int *)malloc(new_nnz * sizeof(int));
    sub_lp->constraint_matrix_values = (double *)malloc(new_nnz * sizeof(double));

    int current_nnz = 0;
    sub_lp->constraint_matrix_row_pointers[0] = 0;

    for (int i = 0; i < m; i++) {
        int start = orig->constraint_matrix_row_pointers[i];
        int end = orig->constraint_matrix_row_pointers[i + 1];
        for (int k = start; k < end; k++) {
            sub_lp->constraint_matrix_col_indices[current_nnz] = orig->constraint_matrix_col_indices[k];
            sub_lp->constraint_matrix_values[current_nnz] = orig->constraint_matrix_values[k];
            current_nnz++;
        }
        // Slack -I
        sub_lp->constraint_matrix_col_indices[current_nnz] = n + i;
        sub_lp->constraint_matrix_values[current_nnz] = -1.0;
        current_nnz++;
        sub_lp->constraint_matrix_row_pointers[i + 1] = current_nnz;
    }

    sub_lp->constraint_lower_bound = (double *)calloc(m, sizeof(double));
    sub_lp->constraint_upper_bound = (double *)calloc(m, sizeof(double));

    return sub_lp;
}

// --- Main IR Function ---
cupdlpx_result_t *optimize_iterative_refinement(
    const pdhg_parameters_t *params,
    const lp_problem_t *original_problem,
    const int max_refine_steps,
    const double final_tol,
    bool verbose,
    bool solver_verbose) 
{
    int n = original_problem->num_variables;
    int m = original_problem->num_constraints;

    // Create Non-Rescaling Params
    pdhg_parameters_t *non_rescale_params = new pdhg_parameters_t;
    set_default_parameters(non_rescale_params);
    non_rescale_params->l_inf_ruiz_iterations = 0;
    non_rescale_params->has_pock_chambolle_alpha = false;
    non_rescale_params->bound_objective_rescaling = false;
    print_initial_info(params, original_problem);

    // Initialize Solver State
    rescale_info_t *rescale_info = rescale_problem(non_rescale_params, original_problem);
    pdhg_solver_state_t *check_state = initialize_solver_state(original_problem, rescale_info);
    
    // [CLEANUP] Free rescale info immediately after use
    rescale_info_free(rescale_info); 
    // [CLEANUP] Free the temp params
    delete non_rescale_params; 
    // ------------------------------------------------------------------
    // 2. INITIAL SOLVE (Standard, with scaling allowed)
    // ------------------------------------------------------------------
    if (verbose) printf(">>> IR: Running Initial Solve...\n");
    
    pdhg_parameters_t initial_params = *params;
    // We can solve somewhat loosely initially
    // initial_params.termination_criteria.eps_optimal_relative = 1e-6; 
    initial_params.verbose = solver_verbose;
    cupdlpx_result_t *main_result = optimize(&initial_params, original_problem);
    double primal_scale = main_result->absolute_primal_residual / main_result->relative_primal_residual;
    double dual_scale = main_result->absolute_dual_residual / main_result->relative_dual_residual;
    printf("    Initial Solve Completed: Primal Scale: %.4e, Dual Scale: %.4e\n", 
           primal_scale, dual_scale);
    
    if (!main_result) {
        pdhg_solver_state_free(check_state);
        return NULL;
    }

    // ------------------------------------------------------------------
    // 3. REFINE LOOP
    // ------------------------------------------------------------------
    double *h_Ax  = (double*)malloc(m * sizeof(double));
    double *h_Aty = (double*)malloc(n * sizeof(double));
    
    double *shift_l = (double*)malloc(n * sizeof(double));
    double *shift_u = (double*)malloc(n * sizeof(double));
    double *shift_L = (double*)malloc(m * sizeof(double));
    double *shift_U = (double*)malloc(m * sizeof(double));
    double *res_c   = (double*)malloc(n * sizeof(double));

    double delta_P = 1.0;
    double delta_D = 1.0;
    double alpha = 10.0;
    int iter = 0.0;
    while (true) {
        if (max_refine_steps <= 0) break; 
        CUDA_CHECK(cudaMemcpy(check_state->pdhg_primal_solution, main_result->primal_solution, 
                              n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(check_state->pdhg_dual_solution, main_result->dual_solution, 
                              m * sizeof(double), cudaMemcpyHostToDevice));

        CUSPARSE_CHECK(cusparseDnVecSetValues(check_state->vec_primal_sol, check_state->pdhg_primal_solution));
        CUSPARSE_CHECK(cusparseDnVecSetValues(check_state->vec_primal_prod, check_state->primal_product));

        CUSPARSE_CHECK(cusparseSpMV(
            check_state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &HOST_ONE, check_state->matA, check_state->vec_primal_sol, &HOST_ZERO, check_state->vec_primal_prod,
            CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, check_state->primal_spmv_buffer));

        CUSPARSE_CHECK(cusparseDnVecSetValues(check_state->vec_dual_sol, check_state->pdhg_dual_solution));
        CUSPARSE_CHECK(cusparseDnVecSetValues(check_state->vec_dual_prod, check_state->dual_product));

        CUSPARSE_CHECK(cusparseSpMV(
            check_state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &HOST_ONE, check_state->matAt, check_state->vec_dual_sol, &HOST_ZERO, check_state->vec_dual_prod,
            CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, check_state->dual_spmv_buffer));

        CUDA_CHECK(cudaMemcpy(h_Ax, check_state->primal_product, m * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_Aty, check_state->dual_product, n * sizeof(double), cudaMemcpyDeviceToHost));
        
        double max_p_vio = 0.0;
        double max_d_vio = 0.0;
        double l2_p_vio = 0.0;
        double l2_d_vio = 0.0;
        double bound_tol = 1e-6; 

        for(int j=0; j<n; j++) {
            double x_val = main_result->primal_solution[j];
            double lb = original_problem->variable_lower_bound[j];
            double ub = original_problem->variable_upper_bound[j];

            shift_l[j] = lb - x_val;
            shift_u[j] = ub - x_val;
        }

        for(int i=0; i<m; i++) {
            double ax_val = h_Ax[i];
            double lb = original_problem->constraint_lower_bound[i];
            double ub = original_problem->constraint_upper_bound[i];

            shift_L[i] = lb - ax_val;
            shift_U[i] = ub - ax_val;

            // Primal Violation: Ax < L or Ax > U
            if (ax_val < lb) max_p_vio = fmax(max_p_vio, lb - ax_val);
            if (ax_val > ub) max_p_vio = fmax(max_p_vio, ax_val - ub);
            if (ax_val < lb) l2_p_vio += (lb - ax_val) * (lb - ax_val);
            if (ax_val > ub) l2_p_vio += (ax_val - ub) * (ax_val - ub);
        }

        for(int j=0; j<n; j++) {
            double rc = original_problem->objective_vector[j] - h_Aty[j];
            res_c[j] = rc; 

            double x_val = main_result->primal_solution[j];
            double lb = original_problem->variable_lower_bound[j];
            double ub = original_problem->variable_upper_bound[j];
            
            double vio = 0.0;

            bool at_lb = (x_val <= lb + bound_tol);
            bool at_ub = (x_val >= ub - bound_tol);

            if (at_lb && at_ub) {
                vio = 0.0; 
            }
            else if (at_lb) {
                if (rc < 0) vio = -rc;
            }
            else if (at_ub) {
                if (rc > 0) vio = rc;
            }
            else {
                vio = fabs(rc);
            }

            max_d_vio = fmax(max_d_vio, vio);
            l2_d_vio += vio * vio;
        }
        double l2_p_vio_norm = sqrt(l2_p_vio);
        double l2_d_vio_norm = sqrt(l2_d_vio);
        if (verbose) {
            
            printf(">>> IR Step %d: Max Primal Inf: %.4e, L2 Primal Inf: %.4e, Max Dual Inf: %.4e, L2 Dual Inf: %.4e\n", 
                   iter, max_p_vio, l2_p_vio_norm, max_d_vio, l2_d_vio_norm);
            printf("    Relative L2 Residuals: Primal: %.4e, Dual: %.4e\n", 
                   l2_p_vio_norm / primal_scale, l2_d_vio_norm / dual_scale);
        }
        main_result->absolute_primal_residual = l2_p_vio_norm;
        main_result->absolute_dual_residual = l2_d_vio_norm;
        main_result->relative_primal_residual = l2_p_vio_norm / primal_scale;
        main_result->relative_dual_residual = l2_d_vio_norm / dual_scale;
        if (iter > max_refine_steps) {
            if (verbose) printf(">>> IR Reached Maximum Refinement Steps.\n");
            break;
        }
        if (l2_p_vio_norm / primal_scale < final_tol && l2_d_vio_norm / dual_scale < final_tol) {
            if (verbose) printf(">>> IR Converged.\n");
            break;
        }
        double target_P = (max_p_vio > 1e-12) ? (1.0 / max_p_vio) : 1e12;
        double target_D = (max_d_vio > 1e-12) ? (1.0 / max_d_vio) : 1e12;

        double alpha_growth = 10.0;
        double next_P = delta_P * alpha_growth;
        double next_D = delta_D * alpha_growth;

        delta_P = fmin(target_P, next_P);
        delta_D = fmin(target_D, next_D);

        delta_P = fmax(delta_P, 1.0);
        delta_D = fmax(delta_D, 1.0);

        double max_scale_limit = 1e6; 
        
        delta_P = fmin(delta_P, max_scale_limit);
        delta_D = fmin(delta_D, max_scale_limit);

        delta_P = round_to_pow2(delta_P);
        delta_D = round_to_pow2(delta_D);

        if (verbose) {
            printf("    Scaling: dP=%.1e, dD=%.1e (Max Vio: P=%.1e, D=%.1e)\n", 
                   delta_P, delta_D, max_p_vio, max_d_vio);
        }

        lp_problem_t *sub_lp = construct_refinement_lp_with_slacks(
            original_problem, 
            main_result->dual_solution, 
            delta_P, delta_D,
            res_c, shift_l, shift_u, shift_L, shift_U
        );

        double current_tol = 1e-6; 
        
        if (delta_D < 1e4 || delta_P < 1e4) {
            current_tol = 1e-6; 
        }

        double b_norm = 0.0;
        for (int i = 0; i < m; i++) {
            double lb = sub_lp->constraint_lower_bound[i];
            double ub = sub_lp->constraint_upper_bound[i];
            
            b_norm += (lb * lb) + (ub * ub);
        }
        b_norm = sqrt(b_norm);
        double c_norm = 0.0;
        for (int j = 0; j < n; j++) {
            double c = sub_lp->objective_vector[j];
            c_norm += c * c;
        }
        c_norm = sqrt(c_norm);
        pdhg_parameters_t sub_params = initial_params;
        sub_params.termination_criteria.eps_optimal_relative = 1e-3 ;
        sub_params.termination_criteria.eps_feasible_relative = 0.01 / (1.0 + c_norm + b_norm);
        sub_params.termination_criteria.time_sec_limit = 30.0; 
        sub_params.restart_params.artificial_restart_threshold = 0.4;

        cupdlpx_result_t *sub_res = optimize(&sub_params, sub_lp);
        iter += 1;
        bool step_valid = true;
        
        if (!sub_res) {
            step_valid = false;
        } else {
            if (fabs(sub_res->primal_objective_value) > 1e15 || 
                fabs(sub_res->dual_objective_value) > 1e15) {
                if (verbose) printf("    >>> WARNING: Sub-problem diverged (Obj > 1e15). Discarding step.\n");
                step_valid = false;
            }
        }

        if (step_valid) {
            for (int j = 0; j < n; j++) main_result->primal_solution[j] += sub_res->primal_solution[j] / delta_P;
            for (int i = 0; i < m; i++) main_result->dual_solution[i] += sub_res->dual_solution[i] / delta_D;
            
            main_result->total_count += sub_res->total_count;
            main_result->cumulative_time_sec += sub_res->cumulative_time_sec;
        } else {
            if (sub_res) cupdlpx_result_free(sub_res);
            free(sub_lp->objective_vector); 
            free(sub_lp);
            break; 
        }

        // Cleanup
        if (sub_res) cupdlpx_result_free(sub_res);
        free(sub_lp->objective_vector);
        free(sub_lp->variable_lower_bound);
        free(sub_lp->variable_upper_bound);
        free(sub_lp->constraint_lower_bound);
        free(sub_lp->constraint_upper_bound);
        free(sub_lp->constraint_matrix_row_pointers);
        free(sub_lp->constraint_matrix_col_indices);
        free(sub_lp->constraint_matrix_values);
        free(sub_lp);
    }

    if (verbose) printf(">>> IR: Running Final Solve...\n");
    pdhg_parameters_t final_params = *params;
    final_params.verbose = true;
    lp_problem_t final_problem = *original_problem;
    set_start_values(
        &final_problem,
        main_result->primal_solution,
        main_result->dual_solution);
    final_params.verbose = true;
    final_params.termination_criteria.eps_optimal_relative = 1e-8;
    final_params.termination_criteria.eps_feasible_relative = 1e-8;
    final_params.restart_params.artificial_restart_threshold = 0.6;
    free(h_Ax); free(h_Aty);
    free(shift_l); free(shift_u); free(shift_L); free(shift_U); free(res_c);
    pdhg_solver_state_free(check_state);

    return main_result;
}