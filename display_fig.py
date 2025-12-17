import numpy as np
import matplotlib.pyplot as plt

# ===================== 读 CSV 数据 =====================

data = np.loadtxt('simulation_result_HOCBF.csv', delimiter=',', skiprows=1)

t   = data[:, 0]
x1  = data[:, 1]
x2  = data[:, 2]
v1  = data[:, 3]
v2  = data[:, 4]
u1  = data[:, 5]
u2  = data[:, 6]
ud1 = data[:, 7]
ud2 = data[:, 8]
h   = data[:, 9]
hc  = data[:, 10]

# ===================== 场景参数（要和 C++ 一致） =====================

# 机器人半尺寸（矩形），对应 par.w = Vec2(0.2, 0.3)
agent_half = np.array([0.2, 0.3])

# 矩形障碍（中心 + half-size，都是水平放置 psi=0）
rect_obs = [
    (np.array([4.0, 6.5]), np.array([1.0, 0.2])),  # 和 C++ 第一块一致
    (np.array([2.0, 7.0]), np.array([1.0, 0.2])),  # 第二块
]

# 6 边形障碍（CCW 顶点，和 C++ polyVerts 一致）
poly_vertices = np.array([
    [2.0, 1.0],
    [4.0, 1.0],
    [5.0, 3.0],
    [3.5, 4.5],
    [2.0, 5.5],
    [1.5, 3.0],
])

goal = np.array([7.0, 7.0])
start = np.array([x1[0], x2[0]])

# ===================== 一些几何小工具 =====================

def rect_polygon(center, half_size):
    """给定中心和 half-size，生成一个闭合矩形顶点序列（轴对齐）"""
    cx, cy = center
    wx, wy = half_size
    pts = np.array([
        [cx - wx, cy - wy],
        [cx + wx, cy - wy],
        [cx + wx, cy + wy],
        [cx - wx, cy + wy],
        [cx - wx, cy - wy],  # 闭合
    ])
    return pts

def inflate_polygon_by_rect(vertices_ccw, rect_half):
    """
    用 half-size = rect_half 的轴对齐矩形，对一个凸多边形做 Minkowski 膨胀。
    做法和 C++ CBF 一致：
      每条边的半空间 n^T x >= n^T p + r
      其中 r = |n_x|*w_x + |n_y|*w_y
    最后求相邻 offset 直线的交点，得到膨胀后的多边形顶点。
    假设 vertices_ccw 是凸多边形，按 CCW 排列。
    """
    verts = np.asarray(vertices_ccw)
    M = verts.shape[0]

    n_list = []
    d_list = []

    wx, wy = rect_half

    # 为每条边构建：法向 n, 偏移 d = n^T p + r
    for j in range(M):
        vj = verts[j]
        vk = verts[(j + 1) % M]
        e = vk - vj
        elen = np.linalg.norm(e)
        if elen < 1e-9:
            continue
        # CCW 顶点 -> 外法向 n = (e_y, -e_x) / |e|
        n = np.array([e[1], -e[0]]) / elen
        r = abs(n[0]) * wx + abs(n[1]) * wy
        d = n.dot(vj) + r
        n_list.append(n)
        d_list.append(d)

    n_list = np.asarray(n_list)
    d_list = np.asarray(d_list)
    K = n_list.shape[0]

    inflated = []
    for i in range(K):
        n1 = n_list[i]
        d1 = d_list[i]
        n2 = n_list[(i + 1) % K]
        d2 = d_list[(i + 1) % K]
        A = np.vstack([n1, n2])
        b = np.array([d1, d2])
        # 求相邻两条 offset 直线的交点
        x = np.linalg.solve(A, b)
        inflated.append(x)

    inflated = np.asarray(inflated)
    inflated_closed = np.vstack([inflated, inflated[0]])  # 闭合
    return inflated_closed

# 预先算好膨胀多边形
inflated_poly = inflate_polygon_by_rect(poly_vertices, agent_half)

# ===================== 开始画图 =====================

plt.rcParams['font.size'] = 12

fig, axes = plt.subplots(3, 1, figsize=(8, 12))
ax_traj, ax_h, ax_u = axes

# ---------- 1. 轨迹 + 障碍物 + 膨胀 + 机器人形状 ----------

ax_traj.set_title('Trajectory with Robot Shape and Inflated Obstacles')

# 起点和终点
ax_traj.plot(start[0], start[1], 'bo', label='Start')
ax_traj.plot(goal[0],  goal[1],  'go', label='Goal')

# 轨迹（机器人质心）
ax_traj.plot(x1, x2, color='purple', linewidth=2, label='Trajectory')

# 矩形障碍（原始 + 膨胀）
first_orig_rect = True
first_infl_rect = True

for center, half in rect_obs:
    # 原始障碍矩形
    poly = rect_polygon(center, half)
    ax_traj.plot(
        poly[:, 0], poly[:, 1],
        'k-', linewidth=2,
        label='Rect obstacle' if first_orig_rect else None
    )
    first_orig_rect = False

    # 膨胀后障碍 = half + agent_half
    poly_inf = rect_polygon(center, half + agent_half)
    ax_traj.plot(
        poly_inf[:, 0], poly_inf[:, 1],
        'r--', linewidth=1.5,
        label='Inflated rect (obs + robot)' if first_infl_rect else None
    )
    first_infl_rect = False

# 多边形障碍：原始
poly_closed = np.vstack([poly_vertices, poly_vertices[0]])
ax_traj.plot(
    poly_closed[:, 0], poly_closed[:, 1],
    'k-', linewidth=2, label='Polygon obstacle'
)

# 多边形障碍：膨胀后
ax_traj.plot(
    inflated_poly[:, 0], inflated_poly[:, 1],
    'r--', linewidth=1.5, label='Inflated polygon'
)

# 在若干时刻画出机器人矩形（真实 shape）
num_boxes = 48
idxs = np.linspace(0, len(t) - 1, num_boxes, dtype=int)

for i, idx in enumerate(idxs):
    cx = x1[idx]
    cy = x2[idx]
    robot_poly = rect_polygon(np.array([cx, cy]), agent_half)
    ax_traj.plot(
        robot_poly[:, 0], robot_poly[:, 1],
        color='blue', alpha=0.4,
        linewidth=1.5,
        label='Robot shape' if i == 0 else None
    )

ax_traj.set_xlabel(r'$x_1$ (m)')
ax_traj.set_ylabel(r'$x_2$ (m)')
ax_traj.set_xlim(0, 8)
ax_traj.set_ylim(0, 8)
ax_traj.set_aspect('equal', 'box')
ax_traj.grid(True)
# ax_traj.legend(loc='upper left')

# ---------- 2. CBF 值随时间（平滑 h 和非平滑 hc） ----------

ax_h.set_title('Barrier functions $h(t)$ and $h_c(t)$')
ax_h.axhline(0.0, color='k', linewidth=1, linestyle='--')

ax_h.plot(t, h,  'tab:red', linewidth=2, label=r'smooth $h$')
ax_h.plot(t, hc, 'tab:gray', linewidth=2, label=r'nonsmooth $h_c$')

ax_h.set_xlabel('t (s)')
ax_h.set_ylabel('Barrier value')
ax_h.grid(True)
ax_h.legend(loc='best')

# ---------- 3. 控制输入（不用红绿，避免覆盖感） ----------

ax_u.set_title('Control inputs $u_1,u_2$ and desired $u_{d}$')

# 通道 1：蓝色，虚线 = 期望，实线 = 实际
ax_u.plot(
    t, ud1,
    color='tab:blue', linestyle='--', linewidth=1.2,
    alpha=0.8, label=r'$u_{1,d}$'
)
ax_u.plot(
    t, u1,
    color='tab:blue', linestyle='-', linewidth=2.0,
    alpha=0.8, label=r'$u_1$'
)

# 通道 2：橙色，虚线 = 期望，实线 = 实际
ax_u.plot(
    t, ud2,
    color='tab:orange', linestyle='--', linewidth=1.2,
    alpha=0.8, label=r'$u_{2,d}$'
)
ax_u.plot(
    t, u2,
    color='tab:orange', linestyle='-', linewidth=2.0,
    alpha=0.8, label=r'$u_2$'
)

ax_u.set_xlabel('t (s)')
ax_u.set_ylabel('u (m/s$^2$)')
ax_u.grid(True)
ax_u.legend(loc='best')

plt.tight_layout()
plt.show()

