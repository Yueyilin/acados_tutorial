from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos, Function

def export_pendulum_ode_model():
    model_name = 'pendulum_ode_model'
    # 常量
    M = 1. # 滑块质量 [kg]
    m = 0.1 # 重锤质量 [kg]
    g = 9.81 # 重力加速度 [m/s^2]
    l = 0.8 # 连杆长度 [m]
    # 设定状态量和控制量
    # 设定状态量
    x1      = SX.sym('x1')              # 滑块水平位移
    theta   = SX.sym('theta')           # 连杆摆动角度
    v1      = SX.sym('v1')              # 滑块水平速度
    dtheta  = SX.sym('dtheta')          # 连杆摆动角速度
    x = vertcat(x1, theta, v1, dtheta)  # 合并状态量
    # 设定控制量
    F = SX.sym('F') # 推动滑块的力
    u = vertcat(F)  # 合并控制量
    # 设定状态量倒数
    x1_dot      = SX.sym('x1_dot')      # 滑块水平位移导数
    theta_dot   = SX.sym('theta_dot')   # 连杆摆动角度导数
    v1_dot      = SX.sym('v1_dot')      # 滑块水平速度导数
    dtheta_dot  = SX.sym('dtheta_dot')  # 连杆摆动角加速度
    xdot = vertcat(x1_dot, theta_dot, v1_dot, dtheta_dot) # 合并状态量倒数
    # z: 描述了 DAE 的代数变量的CasADi 变量； 默认值：空
    # z = None
    # p: 描述了 DAE 的代数变量的CasADi 变量； 默认值：空
    p = []
    # 动态方程
    cos_theta = cos(theta)
    sin_theta = sin(theta)
    denominator = M + m - m*cos_theta*cos_theta
    # f_expl = vertcat( x1_dot,
    #                   theta_dot,
    #                   v1_dot,
    #                   dtheta_dot)
    # 显式表达式
    f_expl = vertcat(v1,
                     dtheta,
                     (m*l*sin_theta*dtheta*dtheta + m*g*cos_theta*sin_theta+F)/denominator,
                     (-m*l*cos_theta*sin_theta*dtheta*dtheta + F*cos_theta+(M+m)*g*sin_theta)/(l*denominator)
                     )
    # 隐式表达式
    f_impl = xdot - f_expl
    # 创建ACADOS模型
    model = AcadosModel()
    # 将CasADi模型（参数）映射到ACADOS模型
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    # model.z = z
    model.p = p
    model.name = model_name

    return model
