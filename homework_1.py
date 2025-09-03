import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 1. 定义系统动态方程
def f(t, x):
    """
    计算系统在状态 x = [x1, x2] 下的速度向量。
    注意：solve_ivp 要求函数第一个参数是时间 t。
    """
    x1, x2 = x
    x1_dot = x2 - 2
    x2_dot = -x1 - x2 + (1/3) * (x1 + 1)**3 + 1
    return [x1_dot, x2_dot]

# 2. 定义相图的域
x1_min, x1_max = -1 - 4, -1 + 4
x2_min, x2_max = 2 - 4, 2 + 4

# 3. 创建网格
grid_density = 30
x1_vals = np.linspace(x1_min, x1_max, grid_density)
x2_vals = np.linspace(x2_min, x2_max, grid_density)
X1, X2 = np.meshgrid(x1_vals, x2_vals)

# 4. 计算每个网格点的速度向量
U = np.zeros(X1.shape)
V = np.zeros(X2.shape)
for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        x = np.array([X1[i, j], X2[i, j]])
        velocity = f(0, x)
        U[i, j] = velocity[0]
        V[i, j] = velocity[1]

# 5. 向量标准化
M = np.sqrt(U**2 + V**2)
M[M == 0] = 1
U = U / M
V = V / M

# 6. 绘制相图
plt.figure(figsize=(8, 8))
plt.title('Phase Portrait with Trajectories')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.quiver(X1, X2, U, V, color='gray')

# 7. 绘制平衡点
stable_eq_point = np.array([-1, 2])
unstable_eq_point_1 = np.array([np.sqrt(3) - 1, 2])
unstable_eq_point_2 = np.array([-np.sqrt(3) - 1, 2])
plt.plot(stable_eq_point[0], stable_eq_point[1], 'ro', markersize=10, label='Stable Equilibrium')
plt.plot(unstable_eq_point_1[0], unstable_eq_point_1[1], 'b*', markersize=10, label='Unstable Equilibrium')
plt.plot(unstable_eq_point_2[0], unstable_eq_point_2[1], 'b*', markersize=10)

# 8. 模拟并绘制两条轨迹
t_span = [0, 50]  # 模拟时间从 0 到 50
t_eval = np.linspace(t_span[0], t_span[1], 500) # 评估点，使轨迹更平滑

# 初始条件：一个在吸引区域内部，一个在外部
y0_in = [-0.5, 2.5]
y0_out = [-3, 0]

# 求解内部轨迹
sol_in = solve_ivp(f, t_span, y0_in, t_eval=t_eval)
plt.plot(sol_in.y[0], sol_in.y[1], 'g-', linewidth=2, label='Trajectory (inside)')

# 求解外部轨迹
sol_out = solve_ivp(f, t_span, y0_out, t_eval=t_eval)
plt.plot(sol_out.y[0], sol_out.y[1], 'm--', linewidth=2, label='Trajectory (outside)')

plt.legend()
plt.grid(True)
plt.axis('equal')
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)

# 9. 保存图片到文件
plt.savefig('phase_portrait_with_trajectories.png')