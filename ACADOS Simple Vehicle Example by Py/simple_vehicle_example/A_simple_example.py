#!/usr/bin/env python
# coding=UTF-8
'''
Author: Yue Yilin
Date: 2022-07-05 09:28:48
LastEditors: Yue Yilin
LastEditTime: 2022-07-11 00:00:00
Note: --
'''

import os
import sys

from matplotlib.pyplot import plot
sys.path.insert(0, 'common')
# sys.path.append('./common')
# import shutil
# import errno
# import timeit
import numpy as np
import scipy.linalg
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from simple_vehicle_model import export_simple_vehicle_model
from utils import Draw_MPC_point_stabilization_v1


# def safe_mkdir_recursive(directory, overwrite=False):
#     if not os.path.exists(directory):
#         try:
#             os.makedirs(directory)
#         except OSError as exc:
#             if exc.errno == errno.EEXIST and os.path.isdir(directory):
#                 pass
#             else:
#                 raise
#     else:
#         if overwrite:
#             try:
#                 shutil.rmtree(directory)
#             except:
#                 print('Error while removing directory {}'.format(directory))


# create ocp object to formulate the OCP
# 创建一个OCP
# https://docs.acados.org/python_interface/index.html#acados_template.acados_ocp.AcadosOcp
ocp = AcadosOcp()

# Ensure current working directory is current folder
# os.chdir(os.path.dirname(os.path.realpath(__file__)))
# acados_models_dir = './acados_models'
# safe_mkdir_recursive(os.path.join(os.getcwd(), acados_models_dir))
# acados_source_path = os.environ['ACADOS_SOURCE_DIR']
# sys.path.insert(0, acados_source_path)
## 设置ACADOS系统引用以及库的路径（因为ACADOS最后将以C的形式运行，所以必须设置正确）
# ocp.acados_include_path = acados_source_path + '/include'
# ocp.acados_lib_path = acados_source_path + '/lib'

# set model
# 设定ocp.model
model = export_simple_vehicle_model()
ocp.model = model

# 预测时域
Tf = 10
# 状态量个数
nx = model.x.size()[0]
# 控制量个数
nu = model.u.size()[0]
# ny数为x与u之和，原因详见系统CasADi例子以及ACADOS构建优化问题PDF介绍
ny = nx + nu
ny_e = nx
# 仿真时常
# T = 20
# number of discretization steps
N = 100

# n_params = len(model.p)
# 额外参数，本例中没有
# n_params = 0

# set dimensions
ocp.dims.N = N
# NOTE: all dimensions but N are now detected automatically in the Python
#  interface, all other dimensions will be overwritten by the detection.
# 设定ocp的各项属性
# ocp.dims.np = n_params
# ocp.parameter_values = np.zeros(n_params)

# set cost module
## cost类型为线性
ocp.cost.cost_type = 'LINEAR_LS'
ocp.cost.cost_type_e = 'LINEAR_LS'
# Q = 2*np.diag([1e1, 1e1, 1e-2])
# R = 2*np.diag([1e-1, 1e-2])
Q = np.array([[1., 0.0, 0.0], [0.0, 5., 0.0], [0.0, 0.0, .2]])
R = np.array([[.5, 0.0], [0.0, .05]])
ocp.cost.W = scipy.linalg.block_diag(Q, R)
ocp.cost.W_e = Q
## 这里V类矩阵的定义需要参考ACADOS构建里面的解释，实际上就是定义一些映射关系
ocp.cost.Vx = np.zeros((ny, nx))
ocp.cost.Vx[:nx,:nx] = np.eye(nx)

# ocp.cost.Vu = np.zeros((ny, nu))
# ocp.cost.Vu[-nu:, -nu:] = np.eye(nu)
Vu = np.zeros((ny, nu))
Vu[3,0] = 1.0
ocp.cost.Vu = Vu

ocp.cost.Vx_e = np.eye(nx)

ocp.cost.yref  = np.zeros((ny, ))
ocp.cost.yref_e = np.zeros((ny_e, ))
# 一些状态的值，在实际仿真中可以重新给定，所里这里就定义一些空值
# x_ref = np.zeros(nx)
# u_ref = np.zeros(nu)
# ocp.cost.yref = np.concatenate((x_ref, u_ref))
# ocp.cost.yref_e = x_ref


# set constraints
# 约束条件值
# https://docs.acados.org/python_interface/index.html#acados_template.acados_ocp.AcadosOcpConstraints
v_max = 2.6
v_min = 0.0
omega_max = np.pi/4.0
omega_min = -np.pi/4.0
x_min = -20.
x_max = 20.
y_min = -20.
y_max = 20.
x0 = np.zeros(nx)
# 约束条件设置
ocp.constraints.constr_type = 'BGH'
ocp.constraints.x0 = x0
ocp.constraints.lbu = np.array([v_min, omega_min])
ocp.constraints.ubu = np.array([v_max, omega_max])
## 这里是为了定义之前约束条件影响的index，它不需要像CasADi那样定义np.inf这种没有实际意义的约束。
ocp.constraints.idxbu = np.array([0, 1])
# ocp.constraints.lbx = np.array([x_min, y_min])
# ocp.constraints.ubx = np.array([x_max, y_max])
# ocp.constraints.idxbx = np.array([0, 1])

# solver options
# 这些选项建议查手册
# https://docs.acados.org/python_interface/index.html#acados_template.acados_ocp.AcadosOcpConstraints
ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
# explicit Runge-Kutta integrator
ocp.solver_options.integrator_type = 'ERK'
# ocp.solver_options.print_level = 0
ocp.solver_options.nlp_solver_type = 'SQP'
# ocp.solver_options.sim_method_num_stages = 4
# ocp.solver_options.sim_method_num_steps = 3
# ocp.solver_options.nlp_solver_step_length = 0.05
# ocp.solver_options.nlp_solver_max_iter = 200
# ocp.solver_options.tol = 1e-4
ocp.solver_options.qp_solver_cond_N = N
# set prediction horizon
ocp.solver_options.tf = Tf

############################## generate c code ##############################
# 最后一步就是生成配置文件以及相应的C代码，ACADOS可以将仿真器也自动生成
# 也就是我们不需要自己模拟系统的变化状态
## 配置文件
# json_file = os.path.join('./'+model.name+'_acados_ocp.json')
# acados_ocp_solver = AcadosOcpSolver(ocp, json_file=json_file)
# acados_integrator = AcadosSimSolver(ocp, json_file=json_file)
## 求解器
## https://docs.acados.org/python_interface/index.html#acados_template.acados_ocp_solver.AcadosOcpSolver
acados_ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp_' + model.name + '.json')
## 仿真模拟器（仅仿真使用）
## https://docs.acados.org/python_interface/index.html#acados_template.acados_sim_solver.AcadosSimSolver
acados_integrator = AcadosSimSolver(ocp, json_file = 'acados_ocp_' + model.name + '.json')


################################# start sim #################################

# Nsim = int(T * N / Tf)
Nsim = N
# 存储仿真结果
simX = np.ndarray((Nsim + 1, nx))
simU = np.ndarray((Nsim, nu))
# 初始状态
xcurrent = x0
# 结束状态
xend = np.array([4., 4., 0])
# 仿真状态
simX[0,:] = xcurrent

# acados_ocp_solver.set(Nsim, 'yref', xend)
# xendbetween = np.concatenate((xend, np.zeros(2)))
# for i in range(Nsim):
#     acados_ocp_solver.set(i, 'yref', xendbetween)


# closed loop
# 闭环仿真
for i in range(Nsim):

    # update reference
    # 设置j时刻的期望状态
    # for j in range(N):
    #     yref = np.array([4, 4, 0, 0, 0])
    #     acados_ocp_solver.set(j, "yref", yref)
    # # 设置N时刻的期望状态
    # yref_N = np.array([4, 3, 0])
    # acados_ocp_solver.set(N, "yref", yref_N)

    # solve ocp
    # 设置当前循环x0 (stage 0)
    acados_ocp_solver.set(0, "lbx", xcurrent)
    acados_ocp_solver.set(0, "ubx", xcurrent)
    # 求解
    status = acados_ocp_solver.solve()

    # 检查求解结果
    # if status != 0:
        # raise Exception('acados acados_ocp_solver returned status {}. Exiting.'.format(status))

    # 得到下个时刻最优控制
    simU[i,:] = acados_ocp_solver.get(0, "u")
    # simX[i,:] = acados_ocp_solver.get(0, "x")
    # xcurrent = simX[i,:]

    # simulate system
    acados_integrator.set("x", xcurrent)
    acados_integrator.set("u", simU[i,:])

    # 以下纯粹为了仿真
    # 仿真器获得当前位置和控制指令
    status = acados_integrator.solve()
    if status != 0:
        raise Exception('acados integrator returned status {}. Exiting.'.format(status))

    # update state
    # 更新状态
    # print("xcurrent is {}".format(xcurrent))
    xcurrent = acados_integrator.get("x")
    simX[i+1,:] = xcurrent
    
print("simX is {}".format(simX))
# print("xcurrent is {}".format(simU))
# np.plt.plot(np.linspace(0, Tf/N*Nsim, Nsim+1),simX)
Draw_MPC_point_stabilization_v1(rob_diam=0.5, init_state=x0, target_state=xend, robot_states=simX, export_fig=False)