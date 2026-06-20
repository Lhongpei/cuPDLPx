#pragma once

#include "internal_types.h"

#ifdef __cplusplus
extern "C"
{
#endif
    typedef struct {
        bool success;
        int iterations;
        double time_sec;
        termination_reason_t termination_reason;
    } feas_polish_result_t;
    void feasibility_polish(const pdhg_parameters_t *params, pdhg_solver_state_t *state);
    void proj_scheme_feasibility_polish(const pdhg_parameters_t *params, pdhg_solver_state_t *state);
    void norm_scheme_feasibility_polish(const pdhg_parameters_t *params, pdhg_solver_state_t *state);
    void proj_bb_scheme_feasibility_polish(const pdhg_parameters_t *params, pdhg_solver_state_t *state);
    void proj_restart_scheme_feasibility_polish(const pdhg_parameters_t *params, pdhg_solver_state_t *state);
    void proj_obj_scheme_feasibility_polish(const pdhg_parameters_t *params, pdhg_solver_state_t *state);
    void proj_bt_scheme_feasibility_polish(const pdhg_parameters_t *params, pdhg_solver_state_t *state);

#ifdef __cplusplus
}
#endif
