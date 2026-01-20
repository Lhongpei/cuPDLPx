#include "cupdlpx_types.h"
#include "internal_types.h"
#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(cmd) do {                         \
  cudaError_t e = cmd;                               \
  if( e != cudaSuccess ) {                           \
    printf("Cuda failure %s:%d '%s'\n",              \
        __FILE__, __LINE__, cudaGetErrorString(e));  \
    exit(EXIT_FAILURE);                              \
  }                                                  \
} while(0)

#define NCCL_CHECK(cmd) do {                         \
  ncclResult_t r = cmd;                              \
  if (r != ncclSuccess) {                            \
    printf("NCCL failure %s:%d '%s'\n",              \
        __FILE__, __LINE__, ncclGetErrorString(r));  \
    exit(EXIT_FAILURE);                              \
  }                                                  \
} while(0)

ncclComm_t init_nccl(MPI_Comm mpi_comm) {
    ncclUniqueId id;
    ncclComm_t nccl_comm;
    int rank, nranks;

    MPI_Comm_rank(mpi_comm, &rank);
    MPI_Comm_size(mpi_comm, &nranks);

    if (rank == 0) {
        NCCL_CHECK(ncclGetUniqueId(&id));
    }

    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, mpi_comm);
    NCCL_CHECK(ncclCommInitRank(&nccl_comm, nranks, id, rank));

    return nccl_comm;
}

grid_context_t initialize_parallel_context(int P_row, int P_col) {
    grid_context_t grid;
    int initialized;
    
    MPI_Initialized(&initialized);
    if (!initialized) {
        MPI_Init(NULL, NULL);
    }

    grid.comm_global = MPI_COMM_WORLD;
    MPI_Comm_rank(grid.comm_global, &grid.rank_global);
    
    grid.dims[0] = P_row;
    grid.dims[1] = P_col;

    int num_devices;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    int local_device_id = grid.rank_global % num_devices;
    CUDA_CHECK(cudaSetDevice(local_device_id));

    int my_row = grid.rank_global / P_col;
    int my_col = grid.rank_global % P_col;
    
    grid.coords[0] = my_row;
    grid.coords[1] = my_col;

    MPI_Comm_split(grid.comm_global, my_row, grid.rank_global, &grid.comm_row);
    MPI_Comm_split(grid.comm_global, my_col, grid.rank_global, &grid.comm_col);
    grid.nccl_row = init_nccl(grid.comm_row);
    grid.nccl_col = init_nccl(grid.comm_col);

    return grid;
}

int* get_balanced_cuts(const int* weights, int total_dim, int num_partitions) {
    int* cuts = (int*)malloc((num_partitions + 1) * sizeof(int));
    cuts[0] = 0; 
    cuts[num_partitions] = total_dim;

    if (num_partitions == 1) return cuts;

    long long total_weight = 0;
    for (int i = 0; i < total_dim; i++) total_weight += weights[i];

    double target_per_part = (double)total_weight / num_partitions;
    long long current_cumulative = 0;
    int partition_idx = 1;

    for (int i = 0; i < total_dim; i++) {
        current_cumulative += weights[i];

        if (current_cumulative >= partition_idx * target_per_part) {
            cuts[partition_idx] = i + 1;
            partition_idx++;
            if (partition_idx >= num_partitions) break;
        }
    }

    while (partition_idx < num_partitions) {
        cuts[partition_idx] = total_dim;
        partition_idx++;
    }

    return cuts;
}

void csr_extract_submatrix(
    int n_total, int m_total, 
    const int* A_row_ptr, const int* A_col_ind, const double* A_val,
    int row_start, int row_end,
    int col_start, int col_end,
    int** sub_row_ptr, int** sub_col_ind, double** sub_val, int* sub_nnz
) {
    int m_sub = row_end - row_start;
    
    int nnz_count = 0;
    for (int i = row_start; i < row_end; i++) {
        for (int jj = A_row_ptr[i]; jj < A_row_ptr[i+1]; jj++) {
            int col = A_col_ind[jj];
            if (col >= col_start && col < col_end) {
                nnz_count++;
            }
        }
    }
    *sub_nnz = nnz_count;

    *sub_row_ptr = (int*)malloc((m_sub + 1) * sizeof(int));
    *sub_col_ind = (int*)malloc(nnz_count * sizeof(int));
    *sub_val = (double*)malloc(nnz_count * sizeof(double));

    (*sub_row_ptr)[0] = 0;
    int current_nnz = 0;
    for (int i = row_start; i < row_end; i++) {
        for (int jj = A_row_ptr[i]; jj < A_row_ptr[i+1]; jj++) {
            int col = A_col_ind[jj];
            if (col >= col_start && col < col_end) {
                (*sub_col_ind)[current_nnz] = col - col_start; 
                (*sub_val)[current_nnz] = A_val[jj];
                current_nnz++;
            }
        }
        (*sub_row_ptr)[i - row_start + 1] = current_nnz;
    }
}

double* copy_slice(const double* src, int start, int count) {
    if (count <= 0) return NULL;
    double* dst = (double*)malloc(count * sizeof(double));
    memcpy(dst, src + start, count * sizeof(double));
    return dst;
}

lp_problem_t* partition_lp_problem(
    const lp_problem_t* global_lp, 
    grid_context_t* grid, 
    partition_method_t method,
    int* out_n_start,  
    int* out_m_start 
) {
    lp_problem_t* loc = (lp_problem_t*)calloc(1, sizeof(lp_problem_t));

    int my_row_idx = grid->coords[0];
    int my_col_idx = grid->coords[1];
    int P_rows = grid->dims[0];
    int P_cols = grid->dims[1];

    int n_total = global_lp->num_variables;
    int m_total = global_lp->num_constraints;

    int n_start, n_end, m_start, m_end;

    if (method == NNZ_BALANCE_PARTITION) {
        int* col_weights = (int*)calloc(n_total, sizeof(int));
        for (int i = 0; i < global_lp->constraint_matrix_num_nonzeros; i++) {
            int c = global_lp->constraint_matrix_col_indices[i];
            if(c < n_total) col_weights[c]++;
        }
        int* col_cuts = get_balanced_cuts(col_weights, n_total, P_cols);
        n_start = col_cuts[my_col_idx];
        n_end   = col_cuts[my_col_idx + 1];
        free(col_weights); free(col_cuts);

        int* row_weights = (int*)malloc(m_total * sizeof(int));
        for (int i = 0; i < m_total; i++) {
            row_weights[i] = global_lp->constraint_matrix_row_pointers[i+1] 
                           - global_lp->constraint_matrix_row_pointers[i];
        }
        int* row_cuts = get_balanced_cuts(row_weights, m_total, P_rows);
        m_start = row_cuts[my_row_idx];
        m_end   = row_cuts[my_row_idx + 1];
        free(row_weights); free(row_cuts);

    } else {
        int n_chunk = n_total / P_cols;
        n_start = my_col_idx * n_chunk;
        n_end   = (my_col_idx == P_cols - 1) ? n_total : (my_col_idx + 1) * n_chunk;

        int m_chunk = m_total / P_rows;
        m_start = my_row_idx * m_chunk;
        m_end   = (my_row_idx == P_rows - 1) ? m_total : (my_row_idx + 1) * m_chunk;
    }

    if (out_n_start) *out_n_start = n_start;
    if (out_m_start) *out_m_start = m_start;

    loc->num_variables = n_end - n_start;
    loc->num_constraints = m_end - m_start;

    csr_extract_submatrix(
        n_total, m_total,
        global_lp->constraint_matrix_row_pointers,
        global_lp->constraint_matrix_col_indices,
        global_lp->constraint_matrix_values,
        m_start, m_end, n_start, n_end,
        &loc->constraint_matrix_row_pointers,
        &loc->constraint_matrix_col_indices,
        &loc->constraint_matrix_values,
        &loc->constraint_matrix_num_nonzeros
    );

    loc->objective_vector     = copy_slice(global_lp->objective_vector, n_start, loc->num_variables);
    loc->variable_lower_bound = copy_slice(global_lp->variable_lower_bound, n_start, loc->num_variables);
    loc->variable_upper_bound = copy_slice(global_lp->variable_upper_bound, n_start, loc->num_variables);
    
    loc->constraint_lower_bound = copy_slice(global_lp->constraint_lower_bound, m_start, loc->num_constraints);
    loc->constraint_upper_bound = copy_slice(global_lp->constraint_upper_bound, m_start, loc->num_constraints);

    if (global_lp->primal_start) {
        loc->primal_start = copy_slice(global_lp->primal_start, n_start, loc->num_variables);
    }
    if (global_lp->dual_start) {
        loc->dual_start = copy_slice(global_lp->dual_start, m_start, loc->num_constraints);
    }
    
    loc->objective_constant = global_lp->objective_constant; 
    return loc;
}

rescale_info_t* partition_rescale_info(
    rescale_info_t* global_info, 
    grid_context_t* grid, 
    partition_method_t method,
    int* out_n_start, 
    int* out_m_start
) {
    rescale_info_t* loc_info = (rescale_info_t*)calloc(1, sizeof(rescale_info_t));
    
    int n_start, m_start;
    loc_info->scaled_problem = partition_lp_problem(
        global_info->scaled_problem, 
        grid, 
        method, 
        &n_start, 
        &m_start
    );

    lp_problem_t* loc_lp = loc_info->scaled_problem;

    loc_info->var_rescale = copy_slice(
        global_info->var_rescale, 
        n_start, 
        loc_lp->num_variables
    );

    loc_info->con_rescale = copy_slice(
        global_info->con_rescale, 
        m_start, 
        loc_lp->num_constraints
    );

    if (out_n_start) *out_n_start = n_start;
    if (out_m_start) *out_m_start = m_start;

    loc_info->con_bound_rescale  = global_info->con_bound_rescale;
    loc_info->obj_vec_rescale    = global_info->obj_vec_rescale;
    loc_info->rescaling_time_sec = global_info->rescaling_time_sec;

    return loc_info;
}