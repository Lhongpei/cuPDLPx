#pragma once

#include "cupdlpx_types.h"
#include "cupdlpx.h"
#include "internal_types.h"
#ifdef __cplusplus
extern "C"
{
#endif

    // create an sub-lp problem from mask
    void proj_scheme_primal_feasibility_polishing(
        pdhg_solver_state_t *state,
        const pdhg_parameters_t *params);
    void proj_scheme_dual_feasibility_polishing(
        pdhg_solver_state_t *state, 
        const pdhg_parameters_t *params
    );
#ifdef __cplusplus
} // extern "C"
#endif