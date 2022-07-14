
// C 标准库
#include <stdio.h>
#include <stdlib.h>
// ACADOS 相关库
#include "acados/utils/print.h"
#include "acados/utils/math.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"
// 生成的 模型头文件
#include "acados_solver_pendulum_ode_model.h"
// 宏定义常量 相关定义在 acados_solver_pendulum_ode_model.h
#define NX     PENDULUM_ODE_MODEL_NX
#define NZ     PENDULUM_ODE_MODEL_NZ
#define NU     PENDULUM_ODE_MODEL_NU
#define NP     PENDULUM_ODE_MODEL_NP
#define NBX    PENDULUM_ODE_MODEL_NBX
#define NBX0   PENDULUM_ODE_MODEL_NBX0
#define NBU    PENDULUM_ODE_MODEL_NBU
#define NSBX   PENDULUM_ODE_MODEL_NSBX
#define NSBU   PENDULUM_ODE_MODEL_NSBU
#define NSH    PENDULUM_ODE_MODEL_NSH
#define NSG    PENDULUM_ODE_MODEL_NSG
#define NSPHI  PENDULUM_ODE_MODEL_NSPHI
#define NSHN   PENDULUM_ODE_MODEL_NSHN
#define NSGN   PENDULUM_ODE_MODEL_NSGN
#define NSPHIN PENDULUM_ODE_MODEL_NSPHIN
#define NSBXN  PENDULUM_ODE_MODEL_NSBXN
#define NS     PENDULUM_ODE_MODEL_NS
#define NSN    PENDULUM_ODE_MODEL_NSN
#define NG     PENDULUM_ODE_MODEL_NG
#define NBXN   PENDULUM_ODE_MODEL_NBXN
#define NGN    PENDULUM_ODE_MODEL_NGN
#define NY0    PENDULUM_ODE_MODEL_NY0
#define NY     PENDULUM_ODE_MODEL_NY
#define NYN    PENDULUM_ODE_MODEL_NYN
#define NH     PENDULUM_ODE_MODEL_NH
#define NPHI   PENDULUM_ODE_MODEL_NPHI
#define NHN    PENDULUM_ODE_MODEL_NHN
#define NPHIN  PENDULUM_ODE_MODEL_NPHIN
#define NR     PENDULUM_ODE_MODEL_NR

// 主函数
int main()
{
    pendulum_ode_model_solver_capsule *acados_ocp_capsule = pendulum_ode_model_acados_create_capsule();
        /* 构造一个 acados_ocp_capsule ，可以理解为 定义一个 OCP
        在python中 ，类似 ocp = AcadosOcp()
        pendulum_ode_model_acados_create_capsule() 定义在 acados_solver_pendulum_ode_model.c
        申请了一块内存及其指针
        OCP 的内容 pendulum_ode_model_solver_capsule 定义在 acados_solver_pendulum_ode_model.h
        结构体包括了 solver data: acados 对象、期望运行时间、动态模型、成本、约束
    */
    int N = PENDULUM_ODE_MODEL_N;//shooting nodes 数、断点数、射击节点数 ， 与 python 中的 N 一致
    double* new_time_steps = NULL; // 有机会在不生成新代码的情况下更改 C 中的拍摄间隔数
    int status = pendulum_ode_model_acados_create_with_discretization(acados_ocp_capsule, N, new_time_steps);
        /* 创建了钟摆的常微分方程离散模型
        pendulum_ode_model_acados_create_with_discretization
    */
    if (status)
    {
        printf("pendulum_ode_model_acados_create() returned status %d. Exiting.\n", status);
        exit(1);
    }

    // 上一步创建了钟摆的常微分方程离散模型，这一步读取 ocp_nlp_xxx 相关参数
    ocp_nlp_config *nlp_config = pendulum_ode_model_acados_get_nlp_config(acados_ocp_capsule);
    ocp_nlp_dims *nlp_dims = pendulum_ode_model_acados_get_nlp_dims(acados_ocp_capsule);
    ocp_nlp_in *nlp_in = pendulum_ode_model_acados_get_nlp_in(acados_ocp_capsule);
    ocp_nlp_out *nlp_out = pendulum_ode_model_acados_get_nlp_out(acados_ocp_capsule);
    ocp_nlp_solver *nlp_solver = pendulum_ode_model_acados_get_nlp_solver(acados_ocp_capsule);
    void *nlp_opts = pendulum_ode_model_acados_get_nlp_opts(acados_ocp_capsule);

    // 初始条件
    // idxbx（边界索引）的初始值
    int idxbx0[NBX0];
    idxbx0[0] = 0;
    idxbx0[1] = 1;
    idxbx0[2] = 2;
    idxbx0[3] = 3;
    // x 和 u 的下边界的初始值
    double lbx0[NBX0];
    double ubx0[NBX0];
    lbx0[0] = 0;
    ubx0[0] = 0;
    lbx0[1] = 3.141592653589793;
    ubx0[1] = 3.141592653589793;
    lbx0[2] = 0;
    ubx0[2] = 0;
    lbx0[3] = 0;
    ubx0[3] = 0;

    // 类似与python的 acados_ocp_solver.set(0, "lbx", xcurrent)，即 .set()
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "idxbx", idxbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "lbx", lbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "ubx", ubx0);

    // 状态量初始化
    double x_init[NX];
    x_init[0] = 0.0;
    x_init[1] = 0.0;
    x_init[2] = 0.0;
    x_init[3] = 0.0;

    // 控制量初始化
    double u0[NU];
    u0[0] = 0.0;

    // 准备一些指标
    int NTIMINGS = 1; // 类似与python中的 Tf
    double min_time = 1e12;
    double kkt_norm_inf;
    double elapsed_time;
    int sqp_iter;

    // 记录状态量x和控制量u轨迹的数组
    double xtraj[NX * (N+1)];
    double utraj[NU * N];


    // 循环求解OCP
    int rti_phase = 0;

    for (int ii = 0; ii < NTIMINGS; ii++)
    {
        // initialize solution
        for (int i = 0; i < N; i++)
        {
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "x", x_init);
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "u", u0);
        }
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, N, "x", x_init);
        ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "rti_phase", &rti_phase);
        status = pendulum_ode_model_acados_solve(acados_ocp_capsule);
        // 构造求解器，类似于python中的status = acados_ocp_solver.solve()
        ocp_nlp_get(nlp_config, nlp_solver, "time_tot", &elapsed_time);
        min_time = MIN(elapsed_time, min_time);
    }

    /* 打印求解结果和统计数据 */
    for (int ii = 0; ii <= nlp_dims->N; ii++)
        ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, ii, "x", &xtraj[ii*NX]);
    for (int ii = 0; ii < nlp_dims->N; ii++)
        ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, ii, "u", &utraj[ii*NU]);

    printf("\n--- 状态量变化过程 ---\n");
    d_print_exp_tran_mat( NX, N+1, xtraj, NX);
    printf("\n--- 控制量执行过程 ---\n");
    d_print_exp_tran_mat( NU, N, utraj, NU );
    // ocp_nlp_out_print(nlp_solver->dims, nlp_out);

    printf("\nsolved ocp %d times, solution printed above\n\n", NTIMINGS);

    if (status == ACADOS_SUCCESS) //
    {
        printf("pendulum_ode_model_acados_solve(): SUCCESS!\n");
    }
    else
    {
        printf("pendulum_ode_model_acados_solve() failed with status %d.\n", status);
    }

    // 读取解决方案
    ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, 0, "kkt_norm_inf", &kkt_norm_inf);
    ocp_nlp_get(nlp_config, nlp_solver, "sqp_iter", &sqp_iter);

    // 打印一些迭代次数之类的量
    pendulum_ode_model_acados_print_stats(acados_ocp_capsule);

    printf("\nSolver info:\n");
    printf(" SQP iterations %2d\n minimum time for %d solve %f [ms]\n KKT %e\n",
           sqp_iter, NTIMINGS, min_time*1000, kkt_norm_inf);

    // 析构求求解器
    status = pendulum_ode_model_acados_free(acados_ocp_capsule);
    if (status) {
        printf("pendulum_ode_model_acados_free() returned status %d. \n", status);
    }
    // 析构OCP
    status = pendulum_ode_model_acados_free_capsule(acados_ocp_capsule);
    if (status) {
        printf("pendulum_ode_model_acados_free_capsule() returned status %d. \n", status);
    }

    return status;
}
