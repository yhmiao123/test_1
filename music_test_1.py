import numpy as np
import matplotlib.pyplot as plt

# 参数设置
f_c = 3e9  # 中心频率，3 GHz
lambda_c = 3e8 / f_c  # 波长
d = 0.5 * lambda_c  # 振元间距，0.5个波长
N_x = 64  # x轴阵列元素数（列数）
N_y = 64  # y轴阵列元素数（行数）
N_elements = N_x * N_y  # 阵列总元素数
N_subcarriers = 1024  # 子载波数目（1024）
N_tau = 200  # 网格数量，用于扫描不同的角度（假设我们有200个不同的角度点）

# 构建阵列流形（空间导向向量）
def steering_vector(azimuth, elevation, N_x, N_y, d, lambda_c):
    k = 2 * np.pi / lambda_c  # 波数
    a = np.zeros(N_elements, dtype=complex)  # 阵列流形初始化
    
    for m in range(N_x):
        for n in range(N_y):
            # 计算每个阵列元素的x和y位置
            x_pos = m * d
            y_pos = n * d
            # 计算每个阵列元素对应的相位
            phase = np.exp(1j * k * (x_pos * np.sin(elevation) * np.cos(azimuth) +
                                     y_pos * np.sin(elevation) * np.sin(azimuth)))
            a[m * N_y + n] = phase
    return a

# 读取Sionna计算的CFR矩阵 H_rt
H_rt = np.load('H_rt.npy')  # 加载CFR矩阵，假设已经存储为Numpy文件

# MUSIC算法进行角度估计
def music_algorithm(H_rt, N_x, N_y, lambda_c, d, N_tau):
    P_music_azimuth = np.empty(N_tau, dtype=float)  # 存储方位角的P_music值
    P_music_elevation = np.empty(N_tau, dtype=float)  # 存储俯仰角的P_music值
    azimuth_grid = np.linspace(-np.pi/2, np.pi/2, N_tau)  # 方位角的网格，从-90°到90°
    elevation_grid = np.linspace(-np.pi/4, np.pi/4, N_tau)  # 俯仰角的网格，从-45°到45°
    
    for k, (azimuth, elevation) in enumerate(zip(azimuth_grid, elevation_grid)):
        # 计算阵列流形
        a_tau = steering_vector(azimuth, elevation, N_x, N_y, d, lambda_c)
        
        # 计算协方差矩阵（可以根据需要处理H_rt）
        R = np.cov(H_rt)  # 协方差矩阵，取决于H_rt的维度
        
        # 计算噪声子空间
        eigvals, eigvecs = np.linalg.eig(R)
        noise_subspace = eigvecs[:, np.argsort(eigvals)[:-1]]  # 去除最大的特征值，保留噪声子空间
        
        # 计算P_music值
        y = np.dot(noise_subspace.T.conj(), a_tau)  # 计算y = EnH * a_tau
        denom = np.abs(np.vdot(y, y))  # 计算分母
        P_music_azimuth[k] = 1.0 / max(denom, 1e-20)  # 避免除零
        P_music_elevation[k] = 1.0 / max(denom, 1e-20)  # 避免除零
    
    return P_music_azimuth, P_music_elevation, azimuth_grid, elevation_grid

# 运行MUSIC算法
P_music_azimuth, P_music_elevation, azimuth_grid, elevation_grid = music_algorithm(H_rt, N_x, N_y, lambda_c, d, N_tau)

# 绘制方位角的MUSIC谱
plt.figure()
plt.plot(azimuth_grid, 10 * np.log10(P_music_azimuth))
plt.title('MUSIC Spectrum for Azimuth')
plt.xlabel('Azimuth Angle (radians)')
plt.ylabel('Power (dB)')
plt.grid(True)
plt.show()

# 绘制俯仰角的MUSIC谱
plt.figure()
plt.plot(elevation_grid, 10 * np.log10(P_music_elevation))
plt.title('MUSIC Spectrum for Elevation')
plt.xlabel('Elevation Angle (radians)')
plt.ylabel('Power (dB)')
plt.grid(True)
plt.show()

# 绘制联合谱（方位角 vs 俯仰角）
P_music_joint = np.zeros((N_tau, N_tau))  # 联合谱矩阵

for i, azimuth in enumerate(azimuth_grid):
    for j, elevation in enumerate(elevation_grid):
        # 计算阵列流形
        a_tau = steering_vector(azimuth, elevation, N_x, N_y, d, lambda_c)
        
        # 计算协方差矩阵（可以根据需要处理H_rt）
        R = np.cov(H_rt)  # 协方差矩阵，取决于H_rt的维度
        
        # 计算噪声子空间
        eigvals, eigvecs = np.linalg.eig(R)
        noise_subspace = eigvecs[:, np.argsort(eigvals)[:-1]]  # 去除最大的特征值，保留噪声子空间
        
        # 计算P_music值
        y = np.dot(noise_subspace.T.conj(), a_tau)  # 计算y = EnH * a_tau
        denom = np.abs(np.vdot(y, y))  # 计算分母
        P_music_joint[i, j] = 1.0 / max(denom, 1e-20)  # 避免除零

# 绘制联合谱
plt.figure()
plt.imshow(10 * np.log10(P_music_joint), aspect='auto', cmap='jet', origin='lower')
plt.title('Joint MUSIC Spectrum (Azimuth vs Elevation)')
plt.xlabel('Azimuth Angle Index')
plt.ylabel('Elevation Angle Index')
plt.colorbar(label='Power (dB)')
plt.show()
