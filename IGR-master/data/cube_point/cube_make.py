import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# --- 1. 参数设置 ---
num_points = 2000  # 点云数量
half_size = 0.8    # 正方体半边长（中心在原点，范围 [-0.8, 0.8]）
output_dir = "~/bi_she/IGR-master/data"
output_file = os.path.join(output_dir, "cube_point_cloud.npy")

# 展开用户目录
output_dir = os.path.expanduser(output_dir)
output_file = os.path.expanduser(output_file)

os.makedirs(os.path.dirname(output_file), exist_ok=True)

# --- 2. 生成正方体表面点云（均匀采样） ---
# 正方体有6个面，每个面分配 num_points // 6 个点
points_per_face = num_points // 6
point_cloud = []

# 6个面的法向量和固定坐标轴
# 面1: x = +half_size (右面)
u = np.random.uniform(-half_size, half_size, points_per_face)
v = np.random.uniform(-half_size, half_size, points_per_face)
point_cloud.append(np.stack([np.full_like(u, half_size), u, v], axis=1))

# 面2: x = -half_size (左面)
u = np.random.uniform(-half_size, half_size, points_per_face)
v = np.random.uniform(-half_size, half_size, points_per_face)
point_cloud.append(np.stack([np.full_like(u, -half_size), u, v], axis=1))

# 面3: y = +half_size (前面)
u = np.random.uniform(-half_size, half_size, points_per_face)
v = np.random.uniform(-half_size, half_size, points_per_face)
point_cloud.append(np.stack([u, np.full_like(u, half_size), v], axis=1))

# 面4: y = -half_size (后面)
u = np.random.uniform(-half_size, half_size, points_per_face)
v = np.random.uniform(-half_size, half_size, points_per_face)
point_cloud.append(np.stack([u, np.full_like(u, -half_size), v], axis=1))

# 面5: z = +half_size (上面)
u = np.random.uniform(-half_size, half_size, points_per_face)
v = np.random.uniform(-half_size, half_size, points_per_face)
point_cloud.append(np.stack([u, v, np.full_like(u, half_size)], axis=1))

# 面6: z = -half_size (下面)
u = np.random.uniform(-half_size, half_size, points_per_face)
v = np.random.uniform(-half_size, half_size, points_per_face)
point_cloud.append(np.stack([u, v, np.full_like(u, -half_size)], axis=1))

point_cloud = np.vstack(point_cloud)

# 如果点数不够，随机补充一些
if len(point_cloud) < num_points:
    extra = num_points - len(point_cloud)
    # 随机选一个面补充
    face_idx = np.random.randint(0, 6, extra)
    u = np.random.uniform(-half_size, half_size, extra)
    v = np.random.uniform(-half_size, half_size, extra)
    extra_points = []
    for i in range(extra):
        fi = face_idx[i]
        if fi == 0:
            extra_points.append([half_size, u[i], v[i]])
        elif fi == 1:
            extra_points.append([-half_size, u[i], v[i]])
        elif fi == 2:
            extra_points.append([u[i], half_size, v[i]])
        elif fi == 3:
            extra_points.append([u[i], -half_size, v[i]])
        elif fi == 4:
            extra_points.append([u[i], v[i], half_size])
        else:
            extra_points.append([u[i], v[i], -half_size])
    point_cloud = np.vstack([point_cloud, np.array(extra_points)])

# --- 3. 保存为 .npy 文件 ---
np.save(output_file, point_cloud)
print(f"✅ 正方体点云已生成并保存至: {output_file}")
print(f"   点云形状: {point_cloud.shape}")

# --- 4. 可视化（可选） ---
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=1, c='r')
ax.set_title(f"Generated Cube Point Cloud ({len(point_cloud)} points)")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

plt.show()