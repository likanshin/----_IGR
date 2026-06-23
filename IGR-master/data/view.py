import numpy as np
import matplotlib.pyplot as plt

data = np.load("/home/rikanshin/bi_she/IGR-master/data/sphere_point/sphere_point_cloud.npy")
print(f"点云形状: {data.shape}")

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=1, c='b')
ax.set_title(f"Sphere Point Cloud ({len(data)} points)")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()