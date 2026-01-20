#include "permute.h"
#include <math.h>
#include <random>

int cmp_tuples(const void *a, const void *b) {
    return ((permute_tuple_t *)a)->new_col - ((permute_tuple_t *)b)->new_col;
}

void col_permute_in_place(
    int m,                             
    int *Ap, int *Aj, double *Ax,       
    const int *old_col_to_new          
) {
    int max_row_nnz = 0;
    for(int i=0; i<m; i++) {
        int len = Ap[i+1] - Ap[i];
        if(len > max_row_nnz) max_row_nnz = len;
    }
    
    permute_tuple_t *buffer = (permute_tuple_t *)malloc(max_row_nnz * sizeof(permute_tuple_t));

    for (int r = 0; r < m; r++) {
        int start = Ap[r];
        int end = Ap[r+1];
        int len = end - start;

        if (len == 0) continue;

        for (int k = 0; k < len; k++) {
            int current_idx = start + k;
            int old_col = Aj[current_idx];
            
            buffer[k].new_col = old_col_to_new[old_col];
            buffer[k].val = Ax[current_idx];
        }

        if (len > 1) {
            qsort(buffer, len, sizeof(permute_tuple_t), cmp_tuples);
        }

        // 3. 写回 (Write back)
        for (int k = 0; k < len; k++) {
            int current_idx = start + k;
            Aj[current_idx] = buffer[k].new_col;
            Ax[current_idx] = buffer[k].val;
        }
    }

    free(buffer);
}

void permute_rows_structural(lp_problem_t *qp, const int *row_perm) {
    int m = qp->num_constraints;
    int nnz = qp->constraint_matrix_num_nonzeros;
    
    int *new_Ap = (int *)malloc((m + 1) * sizeof(int));
    int *new_Aj = (int *)malloc(nnz * sizeof(int));
    double *new_Ax = (double *)malloc(nnz * sizeof(double));

    new_Ap[0] = 0;
    int current_nz = 0;

    for (int i = 0; i < m; i++) {
        int old_row_idx = row_perm[i]; 
        
        int start = qp->constraint_matrix_row_pointers[old_row_idx];
        int len = qp->constraint_matrix_row_pointers[old_row_idx + 1] - start;
        
        if (len > 0) {
            memcpy(&new_Aj[current_nz], &qp->constraint_matrix_col_indices[start], len * sizeof(int));
            memcpy(&new_Ax[current_nz], &qp->constraint_matrix_values[start], len * sizeof(double));
            current_nz += len;
        }
        
        new_Ap[i + 1] = current_nz;
    }

    free(qp->constraint_matrix_row_pointers);
    free(qp->constraint_matrix_col_indices);
    free(qp->constraint_matrix_values);

    qp->constraint_matrix_row_pointers = new_Ap;
    qp->constraint_matrix_col_indices = new_Aj;
    qp->constraint_matrix_values = new_Ax;
}

void permute_double_array(double *arr, int n, const int *perm) {
    if (!arr) return;
    double *tmp = (double *)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) tmp[i] = arr[perm[i]];
    memcpy(arr, tmp, n * sizeof(double));
    free(tmp);
}

void compute_inv_perm(int n, const int *perm, int *inv_perm) {
    for (int i = 0; i < n; i++) inv_perm[perm[i]] = i;
}

void permute_problem(lp_problem_t *qp, int *row_perm, int *col_perm) {
    int m = qp->num_constraints;
    int n = qp->num_variables;

    permute_double_array(qp->objective_vector, n, col_perm);
    permute_double_array(qp->variable_lower_bound, n, col_perm);
    permute_double_array(qp->variable_upper_bound, n, col_perm);
    permute_double_array(qp->primal_start, n, col_perm);

    permute_double_array(qp->constraint_lower_bound, m, row_perm);
    permute_double_array(qp->constraint_upper_bound, m, row_perm);
    permute_double_array(qp->dual_start, m, row_perm);

    permute_rows_structural(qp, row_perm);

    int *inv_col_perm = (int *)malloc(n * sizeof(int));
    compute_inv_perm(n, col_perm, inv_col_perm);

    col_permute_in_place(
        m, 
        qp->constraint_matrix_row_pointers, 
        qp->constraint_matrix_col_indices, 
        qp->constraint_matrix_values, 
        inv_col_perm
    );

    free(inv_col_perm);
}

void generate_random_permutation(int n, int *perm) {
    for (int i = 0; i < n; i++) perm[i] = i;
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int t = perm[i];
        perm[i] = perm[j];
        perm[j] = t;
    }
}

void randomly_permute_problem(lp_problem_t *qp, int **out_row_perm, int **out_col_perm) {
    int m = qp->num_constraints;
    int n = qp->num_variables;

    int *row_perm = (int *)malloc(m * sizeof(int));
    int *col_perm = (int *)malloc(n * sizeof(int));

    generate_random_permutation(m, row_perm);
    generate_random_permutation(n, col_perm);

    permute_problem(qp, row_perm, col_perm);

    *out_row_perm = row_perm;
    *out_col_perm = col_perm;
}