#ifndef ACADOS_SOLVER_pendulum_ode_model_H_
#define ACADOS_SOLVER_pendulum_ode_model_H_

#include "acados/utils/types.h"

#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

// 一些宏定义量，一般只会用到一部分
#define PENDULUM_ODE_MODEL_NX     4
#define PENDULUM_ODE_MODEL_NZ     0
#define PENDULUM_ODE_MODEL_NU     1
#define PENDULUM_ODE_MODEL_NP     0
#define PENDULUM_ODE_MODEL_NBX    0
#define PENDULUM_ODE_MODEL_NBX0   4
#define PENDULUM_ODE_MODEL_NBU    1
#define PENDULUM_ODE_MODEL_NSBX   0
#define PENDULUM_ODE_MODEL_NSBU   0
#define PENDULUM_ODE_MODEL_NSH    0
#define PENDULUM_ODE_MODEL_NSG    0
#define PENDULUM_ODE_MODEL_NSPHI  0
#define PENDULUM_ODE_MODEL_NSHN   0
#define PENDULUM_ODE_MODEL_NSGN   0
#define PENDULUM_ODE_MODEL_NSPHIN 0
#define PENDULUM_ODE_MODEL_NSBXN  0
#define PENDULUM_ODE_MODEL_NS     0
#define PENDULUM_ODE_MODEL_NSN    0
#define PENDULUM_ODE_MODEL_NG     0
#define PENDULUM_ODE_MODEL_NBXN   0
#define PENDULUM_ODE_MODEL_NGN    0
#define PENDULUM_ODE_MODEL_NY0    5
#define PENDULUM_ODE_MODEL_NY     5
#define PENDULUM_ODE_MODEL_NYN    4
#define PENDULUM_ODE_MODEL_N      20
#define PENDULUM_ODE_MODEL_NH     0
#define PENDULUM_ODE_MODEL_NPHI   0
#define PENDULUM_ODE_MODEL_NHN    0
#define PENDULUM_ODE_MODEL_NPHIN  0
#define PENDULUM_ODE_MODEL_NR     0

#ifdef __cplusplus
extern "C" {
#endif

// ** capsule for solver data **
typedef struct pendulum_ode_model_solver_capsule
{
    // acados objects
    ocp_nlp_in *nlp_in;
    ocp_nlp_out *nlp_out;
    ocp_nlp_out *sens_out;
    ocp_nlp_solver *nlp_solver;
    void *nlp_opts;
    ocp_nlp_plan_t *nlp_solver_plan;
    ocp_nlp_config *nlp_config;
    ocp_nlp_dims *nlp_dims;

    // number of expected runtime parameters
    unsigned int nlp_np;

    /* external functions */
    // dynamics

    external_function_param_casadi *forw_vde_casadi;
    external_function_param_casadi *expl_ode_fun;




    // cost






    // constraints




} pendulum_ode_model_solver_capsule;

ACADOS_SYMBOL_EXPORT pendulum_ode_model_solver_capsule * pendulum_ode_model_acados_create_capsule(void);
ACADOS_SYMBOL_EXPORT int pendulum_ode_model_acados_free_capsule(pendulum_ode_model_solver_capsule *capsule);

ACADOS_SYMBOL_EXPORT int pendulum_ode_model_acados_create(pendulum_ode_model_solver_capsule * capsule);

ACADOS_SYMBOL_EXPORT int pendulum_ode_model_acados_reset(pendulum_ode_model_solver_capsule* capsule, int reset_qp_solver_mem);

/**
 * Generic version of pendulum_ode_model_acados_create which allows to use a different number of shooting intervals than
 * the number used for code generation. If new_time_steps=NULL and n_time_steps matches the number used for code
 * generation, the time-steps from code generation is used.
 */
ACADOS_SYMBOL_EXPORT int pendulum_ode_model_acados_create_with_discretization(pendulum_ode_model_solver_capsule * capsule, int n_time_steps, double* new_time_steps);
/**
 * Update the time step vector. Number N must be identical to the currently set number of shooting nodes in the
 * nlp_solver_plan. Returns 0 if no error occurred and a otherwise a value other than 0.
 */
ACADOS_SYMBOL_EXPORT int pendulum_ode_model_acados_update_time_steps(pendulum_ode_model_solver_capsule * capsule, int N, double* new_time_steps);
/**
 * This function is used for updating an already initialized solver with a different number of qp_cond_N.
 */
ACADOS_SYMBOL_EXPORT int pendulum_ode_model_acados_update_qp_solver_cond_N(pendulum_ode_model_solver_capsule * capsule, int qp_solver_cond_N);
ACADOS_SYMBOL_EXPORT int pendulum_ode_model_acados_update_params(pendulum_ode_model_solver_capsule * capsule, int stage, double *value, int np);
ACADOS_SYMBOL_EXPORT int pendulum_ode_model_acados_update_params_sparse(pendulum_ode_model_solver_capsule * capsule, int stage, int *idx, double *p, int n_update);

ACADOS_SYMBOL_EXPORT int pendulum_ode_model_acados_solve(pendulum_ode_model_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT int pendulum_ode_model_acados_free(pendulum_ode_model_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void pendulum_ode_model_acados_print_stats(pendulum_ode_model_solver_capsule * capsule);
                     
ACADOS_SYMBOL_EXPORT ocp_nlp_in *pendulum_ode_model_acados_get_nlp_in(pendulum_ode_model_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *pendulum_ode_model_acados_get_nlp_out(pendulum_ode_model_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *pendulum_ode_model_acados_get_sens_out(pendulum_ode_model_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_solver *pendulum_ode_model_acados_get_nlp_solver(pendulum_ode_model_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_config *pendulum_ode_model_acados_get_nlp_config(pendulum_ode_model_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void *pendulum_ode_model_acados_get_nlp_opts(pendulum_ode_model_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_dims *pendulum_ode_model_acados_get_nlp_dims(pendulum_ode_model_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_plan_t *pendulum_ode_model_acados_get_nlp_plan(pendulum_ode_model_solver_capsule * capsule);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_SOLVER_pendulum_ode_model_H_
