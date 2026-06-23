import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# --- 1. 参数设置 ---
num_points = 2000  # 点云数量（和你之前尝试的 --num_points 对应）
radius = 1.0       # 球体半径
output_dir = "~/bi_she/IGR-master/data"  # 保存路径（先确保这个文件夹存在）
output_file = os.path.join(output_dir, "sphere_point_cloud.npy")

# 展开用户目录（处理 ~ 符号）
output_dir = os.path.expanduser(output_dir)
output_file = os.path.expanduser(output_file)

# 如果目录不存在，创建它
os.makedirs(output_dir, exist_ok=True)

# --- 2. 生成球体点云（均匀采样） ---
# 使用球坐标系均匀采样，避免极点聚集
theta = np.random.uniform(0, 2 * np.pi, num_points)  # 方位角 [0, 2π]
u = np.random.uniform(0, 1, num_points)               # 用于均匀采样极角
phi = np.arccos(1 - 2 * u)                             # 极角 [0, π]

# 转换为笛卡尔坐标 (x, y, z)
x = radius * np.sin(phi) * np.cos(theta)
y = radius * np.sin(phi) * np.sin(theta)
z = radius * np.cos(phi)

# 组合成 N x 3 的点云数组
point_cloud = np.stack([x, y, z], axis=1)

# --- 3. 保存为 .npy 文件 ---
np.save(output_file, point_cloud)
print(f"✅ 球体点云已生成并保存至: {output_file}")
print(f"   点云形状: {point_cloud.shape}")

# --- 4. 可视化（可选，看看效果） ---
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=1, c='b')
ax.set_title(f"Generated Sphere Point Cloud ({num_points} points)")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()