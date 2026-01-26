/*
Copyright 2025 Haihao Lu

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "cupdlpx.h"
#include "mps_parser.h"
#include "solver.h"
#include "utils.h"
#include "active_identify.h"
#include "iterative_refinement.h"
#include <getopt.h>
#include <libgen.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char *termination_reason_tToString(termination_reason_t reason)
{
    switch (reason)
    {
    case TERMINATION_REASON_OPTIMAL:
        return "OPTIMAL";
    case TERMINATION_REASON_PRIMAL_INFEASIBLE:
        return "PRIMAL_INFEASIBLE";
    case TERMINATION_REASON_DUAL_INFEASIBLE:
        return "DUAL_INFEASIBLE";
    case TERMINATION_REASON_TIME_LIMIT:
        return "TIME_LIMIT";
    case TERMINATION_REASON_ITERATION_LIMIT:
        return "ITERATION_LIMIT";
    case TERMINATION_REASON_UNSPECIFIED:
    case TERMINATION_REASON_FEAS_POLISH_SUCCESS:
        return "FEAS_POLISH_SUCCESS";
    default:
        return "UNSPECIFIED";
    }
}

const char *active_set_termination_reason_tToString(active_set_termination_reason_t reason)
{
    switch (reason)
    {
    case ACTIVE_SET_GLOBAL_OPTIMAL:
        return "GLOBAL_OPTIMAL";
    case ACTIVE_SET_FALLBACK_ORIGINAL:
        return "FALLBACK_ORIGINAL";
    case ACTIVE_SET_NO_START_ACTIVE_SET:
        return "NO_START";
    case ACTIVE_SET_IN_PROGRESS:
        return "IN_PROGRESS";
    default:
        return "UNSPECIFIED";
    }
}

char *get_output_path(const char *output_dir, const char *instance_name,
                      const char *suffix)
{
    size_t path_len =
        strlen(output_dir) + strlen(instance_name) + strlen(suffix) + 2;
    char *full_path = safe_malloc(path_len * sizeof(char));
    snprintf(full_path, path_len, "%s/%s%s", output_dir, instance_name, suffix);
    return full_path;
}

char *extract_instance_name(const char *filename)
{
    char *filename_copy = strdup(filename);
    if (filename_copy == NULL)
    {
        perror("Memory allocation failed");
        return NULL;
    }

    char *base = basename(filename_copy);
    char *dot = strchr(base, '.');
    if (dot)
    {
        *dot = '\0';
    }

    char *instance_name = strdup(base);
    free(filename_copy);
    return instance_name;
}

void save_solution(const double *data, int size, const char *output_dir,
                   const char *instance_name, const char *suffix)
{
    char *file_path = get_output_path(output_dir, instance_name, suffix);
    if (file_path == NULL)
    {
        return;
    }

    FILE *outfile = fopen(file_path, "w");
    if (outfile == NULL)
    {
        perror("Error opening solution file");
        free(file_path);
        return;
    }

    for (int i = 0; i < size; ++i)
    {
        fprintf(outfile, "%.10g\n", data[i]);
    }

    fclose(outfile);
    free(file_path);
}

void save_solver_summary(const cupdlpx_result_t *result, const char *output_dir,
                         const char *instance_name)
{
    char *file_path = get_output_path(output_dir, instance_name, "_summary.txt");
    if (file_path == NULL)
    {
        return;
    }

    FILE *outfile = fopen(file_path, "w");
    if (outfile == NULL)
    {
        perror("Error opening summary file");
        free(file_path);
        return;
    }
    fprintf(outfile, "Termination Reason: %s\n",
            termination_reason_tToString(result->termination_reason));
    fprintf(outfile, "Runtime (sec): %e\n", result->cumulative_time_sec);
    fprintf(outfile, "Iterations Count: %d\n", result->total_count);
    fprintf(outfile, "Primal Objective Value: %e\n",
            result->primal_objective_value);
    fprintf(outfile, "Dual Objective Value: %e\n", result->dual_objective_value);
    fprintf(outfile, "Absolute Primal Residual: %e\n",
            result->absolute_primal_residual);
    fprintf(outfile, "Relative Primal Residual: %e\n",
            result->relative_primal_residual);
    fprintf(outfile, "Absolute Dual Residual: %e\n",
            result->absolute_dual_residual);
    fprintf(outfile, "Relative Dual Residual: %e\n",
            result->relative_dual_residual);
    fprintf(outfile, "Absolute Objective Gap: %e\n", result->objective_gap);
    fprintf(outfile, "Relative Objective Gap: %e\n",
            result->relative_objective_gap);
    if(result->feasibility_polishing_time > 0.0){
        fprintf(outfile, "Feasibility Polishing Time (sec): %e\n", result->feasibility_polishing_time);
        fprintf(outfile, "Feasibility Polishing Iteration Count: %d\n", result->feasibility_iteration);
    }
    if (result->has_adaptive_active_set) {
        fprintf(outfile, "Adaptive Active Set Iterations: %d\n", result->adaptive_iteration);
        fprintf(outfile, "Active Set Termination Reason: %s\n", active_set_termination_reason_tToString(result->active_set_termination_reason));
    }
    fclose(outfile);
    free(file_path);
}

void print_usage(const char *prog_name)
{
    fprintf(stderr, "Usage: %s [OPTIONS] <mps_file> <output_dir>\n\n", prog_name);

    fprintf(stderr, "Arguments:\n");
    fprintf(stderr, "  <mps_file>               Path to the input problem in MPS "
                    "format (.mps or .mps.gz).\n");
    fprintf(stderr, "  <output_dir>             Directory where output files "
                    "will be saved. It will contain:\n");
    fprintf(stderr, "                             - <basename>_summary.txt\n");
    fprintf(stderr,
            "                             - <basename>_primal_solution.txt\n");
    fprintf(stderr,
            "                             - <basename>_dual_solution.txt\n\n");

    fprintf(stderr, "Options:\n");
    fprintf(stderr,
            "  -h, --help                          Display this help message.\n");
    fprintf(stderr, "  -v, --verbose                       "
                    "Enable verbose logging (default: false).\n");
    fprintf(stderr, "      --time_limit <seconds>          "
                    "Time limit in seconds (default: 3600.0).\n");
    fprintf(
        stderr,
        "      --iter_limit <iterations>       Iteration limit (default: %d).\n",
        INT32_MAX);
    fprintf(stderr, "      --eps_opt <tolerance>           "
                    "Relative optimality tolerance (default: 1e-4).\n");
    fprintf(stderr, "      --eps_feas <tolerance>          "
                    "Relative feasibility tolerance (default: 1e-4).\n");
    fprintf(stderr, "      --eps_infeas_detect <tolerance> "
                    "Infeasibility detection tolerance (default: 1e-10).\n");
    fprintf(stderr, "      --l_inf_ruiz_iter <int>         "
                    "Iterations for L-inf Ruiz rescaling (default: 10).\n");
    fprintf(stderr, "      --no_pock_chambolle             "
                    "Disable Pock-Chambolle rescaling (default: enabled).\n");
    fprintf(stderr, "      --pock_chambolle_alpha <float>  "
                    "Value for Pock-Chambolle alpha (default: 1.0).\n");
    fprintf(stderr, "      --no_bound_obj_rescaling        "
                    "Disable bound objective rescaling (default: enabled).\n");
    fprintf(stderr, "      --eval_freq <int>               "
                    "Termination evaluation frequency (default: 200).\n");
    fprintf(stderr, "      --sv_max_iter <int>             "
                    "Max iterations for singular value estimation (default: 5000).\n");
    fprintf(stderr, "      --sv_tol <float>                "
                    "Tolerance for singular value estimation (default: 1e-4).\n");
    fprintf(stderr, "  -f  --feasibility_polishing         "
                    "Enable feasibility use feasibility polishing (default: false).\n");
    fprintf(stderr, "      --eps_feas_polish <tolerance>   Relative feasibility "
                    "polish tolerance (default: 1e-6).\n");
}

int main(int argc, char *argv[])
{
    pdhg_parameters_t params;
    set_default_parameters(&params);

    static struct option long_options[] = {
        {"help", no_argument, 0, 'h'},
        {"verbose", no_argument, 0, 'v'},
        {"time_limit", required_argument, 0, 1001},
        {"iter_limit", required_argument, 0, 1002},
        {"eps_opt", required_argument, 0, 1003},
        {"eps_feas", required_argument, 0, 1004},
        {"eps_infeas_detect", required_argument, 0, 1005},
        {"eps_feas_polish", required_argument, 0, 1006},
        {"feasibility_polishing", no_argument, 0, 'f'},
        {"l_inf_ruiz_iter", required_argument, 0, 1007},
        {"pock_chambolle_alpha", required_argument, 0, 1008},
        {"no_pock_chambolle", no_argument, 0, 1009},
        {"no_bound_obj_rescaling", no_argument, 0, 1010},
        {"sv_max_iter", required_argument, 0, 1011},
        {"sv_tol", required_argument, 0, 1012},
        {"eval_freq", required_argument, 0, 1013},
        {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "hvf", long_options, NULL)) != -1)
    {
        switch (opt)
        {
        case 'h':
            print_usage(argv[0]);
            return 0;
        case 'v':
            params.verbose = true;
            break;
        case 1001: // --time_limit
            params.termination_criteria.time_sec_limit = atof(optarg);
            break;
        case 1002: // --iter_limit
            params.termination_criteria.iteration_limit = atoi(optarg);
            break;
        case 1003: // --eps_optimal
            params.termination_criteria.eps_optimal_relative = atof(optarg);
            break;
        case 1004: // --eps_feas
            params.termination_criteria.eps_feasible_relative = atof(optarg);
            break;
        case 1005: // --eps_infeas_detect
            params.termination_criteria.eps_infeasible = atof(optarg);
            break;
        case 1006: // --eps_feas_polish_relative
            params.termination_criteria.eps_feas_polish_relative = atof(optarg);
            break;
        case 'f':  // --feasibility_polishing
            params.feasibility_polishing = true;
            break;
        case 1007: // --l_inf_ruiz_iter
            params.l_inf_ruiz_iterations = atoi(optarg);
            break;
        case 1008: // --pock_chambolle_alpha
            params.pock_chambolle_alpha = atof(optarg);
            break;
        case 1009: // --no_pock_chambolle
            params.has_pock_chambolle_alpha = false;
            break;
        case 1010: // --no_bound_obj_rescaling
            params.bound_objective_rescaling = false;
            break;
        case 1011: // --sv_max_iter
            params.sv_max_iter = atoi(optarg);
            break;
        case 1012: // --sv_tol
            params.sv_tol = atof(optarg);
            break;
        case 1013: // --eval_freq
            params.termination_evaluation_frequency = atoi(optarg);
            break;
        case '?': // Unknown option
            return 1;
        }
    }

    // if (argc - optind != 2)
    // {
    //     fprintf(
    //         stderr,
    //         "Error: You must specify an input file and an output directory.\n\n");
    //     print_usage(argv[0]);
    //     return 1;
    // }
    params.verbose = true;
    const char *filename = argv[optind];
    const char *output_dir = argv[optind + 1];

    // filename = "/data1/MIPLIB2017/selected/lp/neos-5041822-cockle.mps.gz"; 
    // filename = "/data1/MIPLIB2017/selected/lp/app1-2.mps.gz"; 
    // filename = "/data1/MIPLIB2017/selected/lp/physiciansched5-3.mps.gz"; 
    filename = "/data1/MIPLIB2017/selected/lp/physiciansched6-1.mps.gz";

    // physiciansched5-3
    // output_dir = "/home/sevan/cuPDLPx/test/active_test/";

    char *instance_name = extract_instance_name(filename);
    if (instance_name == NULL)
    {
        return 1;
    }

    lp_problem_t *problem = read_mps_file(filename);

    if (problem == NULL)
    {
        fprintf(stderr, "Failed to read or parse the file.\n");
        free(instance_name);
        return 1;
    }

    cupdlpx_result_t *result = optimize(&params, problem);
    // cupdlpx_result_t *result = optimize_with_adaptive_active_identify(&params, problem, 20, 0.5, 1e-8, 1000.00, true);
    // cupdlpx_result_t *optimize_two_stage(
    // const pdhg_parameters_t *params,
    // const lp_problem_t *original_problem,
    // const double coarse_tol, 
    // const double fine_tol, 
    // bool verbose)
    // params.restart_params.artificial_restart_threshold = 1000.0;
    // cupdlpx_result_t *result = optimize_two_stage(&params, problem, 1e-4, 1e-8, 1000.00, true, true, true);
    // cupdlpx_result_t *result = optimize_iterative_refinement(&params, problem, 2, 1e-8, true, true);
    if (result == NULL)
    {
        fprintf(stderr, "Solver failed.\n");
    }
    else
    {
        save_solver_summary(result, output_dir, instance_name);
        save_solution(result->primal_solution, problem->num_variables, output_dir,
                      instance_name, "_primal_solution.txt");
        save_solution(result->dual_solution, problem->num_constraints, output_dir,
                      instance_name, "_dual_solution.txt");
        cupdlpx_result_free(result);
    }

    lp_problem_free(problem);
    free(instance_name);

    return 0;
}