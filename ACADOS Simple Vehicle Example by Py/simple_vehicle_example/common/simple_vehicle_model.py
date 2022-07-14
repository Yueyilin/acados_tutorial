#!/usr/bin/env python
# coding=UTF-8
'''
Author: Yue Yilin
Date: 2022-07-05 09:28:48
LastEditors: Yue Yilin
LastEditTime: 2022-07-10 00:00:00
Note: --
'''

# import numpy as np
# import casadi as ca
# 导入Python接口
from re import U
from acados_template import AcadosModel
# 导入CasADi库
from casadi import SX, vertcat, sin, cos, Function

# 定义车辆模型
def export_simple_vehicle_model():

    # 定义模型名
    model_name = 'simple_vehicle'

    # constants
    # 常数定义，此模型没有

    # CasADi Model
    # CasADi 模型定义，用的CasADi接口
    # 设定状态量x和控制量u
    # 状态x
    x1      = SX.sym('x1')
    x2      = SX.sym('x2')
    theta   = SX.sym('theta')
    # 创建状态向量x
    x       = vertcat(x1, x2, theta)
    # 控制u
    v       = SX.sym('v')
    omega   = SX.sym('omega')
    # 创建控制向量u
    u       = vertcat(v, omega)

    # xdot 系统微分方程
    x1_dot      = SX.sym('x1_dot')
    x2_dot      = SX.sym('x2_dot')
    theta_dot   = SX.sym('theta_dot')
    # 系统微分方程
    xdot = vertcat(x1_dot, x2_dot, theta_dot)

    # algebraic variables
    # z = vertcat([])

    # parameters
    # p = vertcat([])
    p = []

    # 动态模型
    # 显式定义
    f_expl = vertcat(v * cos(theta),
                     v * sin(theta),
                     omega)
    # 隐式定义
    f_impl = xdot - f_expl

    # 创建ACADOS模型
    model = AcadosModel()
    # 模型各个部分定义，从CasADi的表达式映射到ACADOS的模型中
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    # model.z = z
    model.p = p
    model.name = model_name

    return model

