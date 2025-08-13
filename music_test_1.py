import numpy as np
import matplotlib.pyplot as plt

# 参数设置
f_c = 3e9  # 中心频率，3 GHz
lambda_c = 3e8 / f_c  # 波长
k = 2 * np.pi / lambda_c  # 波数
d = 0.5 * lambda_c  # 振元间距，0.5个波长
N_x = 64  # x轴阵列元素数（列数）
N_y = 64  # y轴阵列元素数（行数）
N_elements = N_x * N_y  # 阵列总元素数
N_subcarriers = 1024  # 子载波数目（1024）

# 设置角度网格
azimuth_range = np.linspace(-np.pi, np.pi, 360)  # 方位角范围 -π 到 π，即 -180° 到 180°
elevation_range = np.linspace(0, np.pi, 181)  # 俯仰角范围 0 到 π，即 0° 到 180°

# 构建阵列流形（空间导向向量）
def steering_vector(azimuth, elevation, N_x, N_y, d, lambda_c):
    # 阵列的振元位置坐标
    y_positions = np.arange(N_y) * d  # y方向
    z_positions = np.arange(N_x) * d  # z方向
    a = np.zeros(N_elements, dtype=complex)  # 阵列流形初始化

    # 计算每个阵列元件的相位
    for m in range(N_x):
        for n in range(N_y):
            y_n = y_positions[n]
            z_m = z_positions[m]
            
            # 计算每个振元的相位变化
            phase = np.exp(1j * k * (np.sin(elevation) * np.cos(azimuth) * y_n +
                                     np.sin(elevation) * np.sin(azimuth) * z_m))  # 计算相位变化
            a[m * N_y + n] = phase
    return a

# 读取Sionna计算的CFR矩阵 H_rt
H_rt = np.load('H_rt.npy')  # 加载CFR矩阵，假设已经存储为Numpy文件

# MUSIC算法进行角度估计
def music_algorithm(H_rt, N_x, N_y, lambda_c, d, azimuth_range, elevation_range):
    P_music_azimuth = np.empty(len(azimuth_range), dtype=float)  # 存储方位角的P_music值
    P_music_elevation = np.empty(len(elevation_range), dtype=float)  # 存储俯仰角的P_music值
    P_music_joint = np.zeros((len(azimuth_range), len(elevation_range)))  # 联合谱矩阵

    # 计算协方差矩阵和特征值分解（循环外）
    R = np.cov(H_rt)  # 协方差矩阵，取决于H_rt的维度
    eigvals, eigvecs = np.linalg.eig(R)
    sorted_indices = np.argsort(eigvals)
    noise_subspace = eigvecs[:, sorted_indices[:-1]]  # 去除最大特征值，保留噪声子空间
    
    for i, azimuth in enumerate(azimuth_range):
        for j, elevation in enumerate(elevation_range):
            # 计算阵列流形
            a_tau = steering_vector(azimuth, elevation, N_x, N_y, d, lambda_c)
            
            # 计算P_music值
            y = np.dot(noise_subspace.T.conj(), a_tau)  # 计算y = EnH * a_tau
            denom = np.abs(np.vdot(y, y))  # 计算分母
            P_music_joint[i, j] = 1.0 / max(denom, 1e-20)  # 避免除零

        # 计算方位角的P_music
        a_tau_azimuth = steering_vector(azimuth, 0, N_x, N_y, d, lambda_c)  # 只考虑方位角
        y_azimuth = np.dot(noise_subspace.T.conj(), a_tau_azimuth)
        denom_azimuth = np.abs(np.vdot(y_azimuth, y_azimuth))
        P_music_azimuth[i] = 1.0 / max(denom_azimuth, 1e-20)  # 避免除零

    for j, elevation in enumerate(elevation_range):
        # 计算俯仰角的P_music
        a_tau_elevation = steering_vector(0, elevation, N_x, N_y, d, lambda_c)  # 只考虑俯仰角
        y_elevation = np.dot(noise_subspace.T.conj(), a_tau_elevation)
        denom_elevation = np.abs(np.vdot(y_elevation, y_elevation))
        P_music_elevation[j] = 1.0 / max(denom_elevation, 1e-20)  # 避免除零

    return P_music_azimuth, P_music_elevation, P_music_joint, azimuth_range, elevation_range

# 运行MUSIC算法
P_music_azimuth, P_music_elevation, P_music_joint, azimuth_range, elevation_range = music_algorithm(H_rt, N_x, N_y, lambda_c, d, azimuth_range, elevation_range)

# 绘制方位角的MUSIC谱
plt.figure()
plt.plot(np.degrees(azimuth_range), 10 * np.log10(P_music_azimuth))
plt.title('MUSIC Spectrum for Azimuth')
plt.xlabel('Azimuth Angle (degrees)')
plt.ylabel('Power (dB)')
plt.grid(True)
plt.show()

# 绘制俯仰角的MUSIC谱
plt.figure()
plt.plot(np.degrees(elevation_range), 10 * np.log10(P_music_elevation))
plt.title('MUSIC Spectrum for Elevation')
plt.xlabel('Elevation Angle (degrees)')
plt.ylabel('Power (dB)')
plt.grid(True)
plt.show()

# 绘制联合谱（方位角 vs 俯仰角）
plt.figure()
plt.imshow(10 * np.log10(P_music_joint), aspect='auto', cmap='jet', origin='lower', extent=[np.degrees(azimuth_range[0]), np.degrees(azimuth_range[-1]), np.degrees(elevation_range[0]), np.degrees(elevation_range[-1])])
plt.title('Joint MUSIC Spectrum (Azimuth vs Elevation)')
plt.xlabel('Azimuth Angle (degrees)')
plt.ylabel('Elevation Angle (degrees)')
plt.colorbar(label='Power (dB)')
plt.show()
