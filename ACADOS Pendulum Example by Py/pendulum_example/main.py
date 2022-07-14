from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pendulum_model import export_pendulum_ode_model
from utils import plot_pendulum, plot_pendulum_track
import numpy as np
import scipy.linalg

# 创建一个OCP对象
ocp = AcadosOcp()
# 设置模型
model = export_pendulum_ode_model()
ocp.model = model

Tf = 1 # 预测时域
nx = model.x.size()[0] # 状态量个数
nu = model.u.size()[0] # 控制量个数
ny = nx + nu # Lagrange term 的残差数
ny_e = nx # Mayer term 的残差数
N = 20 # shooting nodes 数、断点数、射击节点数

# 设置维度
ocp.dims.N = N
# NOTE: 除 N 之外的所有维度现在都可以被 Python 自动检测，所有其他维度将被检测值覆盖
# 设置成本模型
ocp.cost.cost_type = 'LINEAR_LS' # 中间 shooting nodes 的成本类型(1到N-1)
ocp.cost.cost_type_e = 'LINEAR_LS' # 最终 shooting nodes 的成本类型(N)
Q = 2*np.diag([1e3, 1e3, 1e-2, 1e-2])   # 惩罚矩阵
R = 2*np.diag([1e-2])                   # 惩罚矩阵
ocp.cost.W = scipy.linalg.block_diag(Q, R) # 中间射击节点（1到N-1）的权重矩阵。(1到N-1)
ocp.cost.W_e = Q # 最终 shootting node 的权重矩阵。(N)
ocp.cost.Vx = np.zeros((ny, nx))
ocp.cost.Vx[:nx,:nx] = np.eye(nx) # x 中间射击节点处的矩阵系数（1 到 N-1）
Vu = np.zeros((ny, nu))
Vu[4,0] = 1.0
ocp.cost.Vu = Vu # u 中间射击节点处的矩阵系数（1 到 N-1）
ocp.cost.Vx_e = np.eye(nx) # x 最终射击节点处的矩阵系数（N）
ocp.cost.yref  = np.zeros((ny, ))# 中间射击节点处的参考（1 到 N-1）
ocp.cost.yref_e = np.zeros((ny_e, )) # 最终射击节点处的参考（N）
ocp.cost.yref = np.array([0.8, 0, 0, 0, 0])
ocp.cost.yref_e = np.array((0.8, 0, 0, 0))
# 设置约束
Fmax = 80 # u1_max
x0 = np.array([0.0, np.pi, 0.0, 0.0])
ocp.constraints.constr_type = 'BGH'     # 射击节点的约束类型(0到N-1)
ocp.constraints.lbu = np.array([-Fmax]) # 射击节点处 u 的下限（0到N-1）
ocp.constraints.ubu = np.array([+Fmax]) # 射击节点处 u 的上限（0到N-1）
ocp.constraints.x0 = x0                 # 状态初始值
ocp.constraints.idxbu = np.array([0])   # 在射击节点（0到N-1）处u上的边界索引
ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
ocp.solver_options.integrator_type = 'ERK'
ocp.solver_options.nlp_solver_type = 'SQP' 
ocp.solver_options.qp_solver_cond_N = N
# 设置预测时域
ocp.solver_options.tf = Tf
# 得到求解器和仿真器并得到对应的配置文件
acados_ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp_' + model.name + '.json')
acados_integrator = AcadosSimSolver(ocp, json_file = 'acados_ocp_' + model.name + '.json')
Nsim = 100
simX = np.ndarray((Nsim+1, nx))
simU = np.ndarray((Nsim, nu))
xcurrent = x0
simX[0,:] = xcurrent
# 闭环
for i in range(Nsim):
    # 设置边界
    acados_ocp_solver.set(0, "lbx", xcurrent)
    acados_ocp_solver.set(0, "ubx", xcurrent)
    # 求解器求解
    status = acados_ocp_solver.solve()
    if status != 0:
        raise Exception('acados acados_ocp_solver returned status {}. Exiting.'.format(status))
    # 读取控制量
    simU[i,:] = acados_ocp_solver.get(0, "u")

    # 设置仿真时的状态量和控制量
    acados_integrator.set("x", xcurrent)
    acados_integrator.set("u", simU[i,:])
    # 控制器求解
    status = acados_integrator.solve()
    if status != 0:
        raise Exception('acados integrator returned status {}. Exiting.'.format(status))

    # 更新状态
    xcurrent = acados_integrator.get("x")
    simX[i+1,:] = xcurrent
# 输出结果
print("  滑块水平位移      连杆摆动角度     滑块水平速度   连杆摆动角速度 {}".format(simX))
print("  推动滑块的水平力 {}".format(simU))
plot_pendulum(np.linspace(0, Tf/N*Nsim, Nsim+1), Fmax, simU, simX)
plot_pendulum_track(simX, l = 0.8, Nsim = 100)

