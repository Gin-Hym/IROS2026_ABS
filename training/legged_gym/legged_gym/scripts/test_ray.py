import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_rays():
    # 1. 你的配置参数
    theta_start = -np.pi/2
    theta_end = 3*np.pi/2 + 0.0001
    theta_step = np.pi/10
    
    # 垂直方向 (注意：这里是你配置里的 0.0 到 0.5)
    theta_up_start = 0.0
    theta_up_end = np.pi/4
    theta_up_step = np.pi/20




    # 2. 生成数据 (模拟 PyTorch 的 meshgrid)
    azimuths = np.arange(theta_start, theta_end, theta_step)
    elevations = np.arange(theta_up_start, theta_up_end, theta_up_step)
    
    # 打印生成的维度信息
    print(f"水平角度数量 (Azimuth): {len(azimuths)}")
    print(f"垂直角度数量 (Elevation): {len(elevations)}")
    print(f"总射线数量: {len(azimuths) * len(elevations)}")

    # 生成网格
    grid_az, grid_el = np.meshgrid(azimuths, elevations)
    
    # 展平
    flat_az = grid_az.flatten()
    flat_el = grid_el.flatten()

    # 3. 转换为 3D 坐标 (方向向量)
    # x = cos(el) * cos(az)
    # y = cos(el) * sin(az)
    # z = sin(el)
    # 假设射线长度为 1 (只看方向)
    r = 1.0
    xs = r * np.cos(flat_el) * np.cos(flat_az)
    ys = r * np.cos(flat_el) * np.sin(flat_az)
    zs = r * np.sin(flat_el)

    # 4. 绘图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 画出射线端点
    # 使用颜色区分高度 (Elevation)
    sc = ax.scatter(xs, ys, zs, c=flat_el, cmap='viridis', s=20)
    
    # 画出射线连线 (从原点到端点)
    # 为了不让图太乱，只画一部分或者画淡一点
    for i in range(len(xs)):
        ax.plot([0, xs[i]], [0, ys[i]], [0, zs[i]], color='gray', alpha=0.1)

    # 画出原点 (机器人位置)
    ax.scatter([0], [0], [0], color='red', s=100, label='Robot Center')

    # 设置坐标轴范围
    limit = 1.0
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    
    ax.set_xlabel('X (Forward)')
    ax.set_ylabel('Y (Side)')
    ax.set_zlabel('Z (Up)')
    ax.set_title(f'Lidar Rays Distribution\nTotal Rays: {len(xs)} (Horizontal: 360 deg, Vertical: 0-28 deg)')
    
    # 添加颜色条说明垂直角度
    cbar = plt.colorbar(sc, ax=ax, pad=0.1)
    cbar.set_label('Elevation Angle (rad)')

    plt.legend()
    plt.show()

if __name__ == "__main__":
    visualize_rays()