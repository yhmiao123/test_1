import numpy as np
import matplotlib.pyplot as plt

# 参数设置
f_c = 3e9  # 中心频率，3 GHz
lambda_c = 3e8 / f_c  # 波长
d = 0.5 * lambda_c  # 振元间距，0.5个波长
N_x = 64  # x轴阵列元素数（列数）
N_y = 64  # y轴阵列元素数（行数）
N_elements = N_x * N_y  # 阵列总元素数
N_tau = 200  # 子载波数目（取值）


# 构建阵列流形（空间导向向量）
def steering_vector(f, tau, N_x, N_y, d, lambda_c):
    # 计算波矢量
    k = 2 * np.pi * f / 3e8  # 波数
    a = np.zeros(N_elements, dtype=complex)  # 阵列导向向量初始化

    # 计算每个阵元的相位
    for m in range(N_x):
        for n in range(N_y):
            x_pos = m * d
            y_pos = n * d
            # 计算每个阵元的相位因子
            phase = np.exp(1j * k * (x_pos * np.sin(tau[0]) * np.cos(tau[1]) + y_pos * np.sin(tau[0]) * np.sin(tau[1])))
            a[m * N_y + n] = phase
    return a


# 读取Sionna计算的CFR矩阵 H_rt
H_rt = np.load('H_rt.npy')  # 直接加载CFR矩阵，假设保存为Numpy文件


# 根据给定的频率网格和时间窗，计算每个τ的P_music值
def music_algorithm(H_rt, N_x, N_y, lambda_c, d, N_tau):
    P_music = np.empty(N_tau, dtype=float)  # 存储每个τ的P_music值
    tau_grid = np.linspace(-np.pi / 2, np.pi / 2, N_tau)  # 假设τ的搜索范围是-90°到90°

    for k, tau in enumerate(tau_grid):
        # 计算阵列流形
        a_tau = steering_vector(f_c, [tau, 0], N_x, N_y, d, lambda_c)  # 只考虑方位角（水平角）

        # 计算协方差矩阵
        R = np.cov(H_rt)  # 可以根据需要对H_rt进行预处理

        # 计算噪声子空间
        eigvals, eigvecs = np.linalg.eig(R)
        noise_subspace = eigvecs[:, np.argsort(eigvals)[:-1]]  # 去掉最大的特征值，保留噪声子空间

        # 计算P_music值
        y = np.dot(noise_subspace.T.conj(), a_tau)  # 计算y = EnH * a_tau
        denom = np.abs(np.vdot(y, y))  # 计算分母
        P_music[k] = 1.0 / max(denom, 1e-20)  # 避免除零

    return P_music, tau_grid


# 运行MUSIC算法
P_music, tau_grid = music_algorithm(H_rt, N_x, N_y, lambda_c, d, N_tau)

# 绘制MUSIC谱
plt.figure()
plt.plot(tau_grid, 10 * np.log10(P_music))
plt.title('MUSIC Spectrum')
plt.xlabel('Angle (radians)')
plt.ylabel('Power (dB)')
plt.grid(True)
plt.show()

