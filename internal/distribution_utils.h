#pragma once
#include "internal_types.h"
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>  
#include <nccl.h>
#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Initializes the MPI and NCCL contexts for a 2D processor grid.
 * * @param P_row Number of processor rows
 * @param P_col Number of processor columns
 * @return GridContext The initialized context structure
 */
grid_context_t initialize_parallel_context(int P_row, int P_col);
rescale_info_t* partition_rescale_info(
    rescale_info_t* global_info, 
    grid_context_t* grid, 
    partition_method_t method,
    int* out_n_start, 
    int* out_m_start
);
lp_problem_t* partition_lp_problem(
    const lp_problem_t* global_lp, 
    grid_context_t* grid, 
    partition_method_t method,
    int* out_n_start,  
    int* out_m_start 
);
#ifdef __cplusplus
}
#endif