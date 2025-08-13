import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd

# 1. 配置参数
M = 8  # 阵列天线数量
d = 0.5  # 天线间距（以波长为单位）
signal_count = 3  # 假设有3个信号
SNR = 20  # 信噪比（dB）
freq = 1e9  # 信号频率（1 GHz）
c = 3e8  # 光速

# 2. 信号的到达角度 (俯仰角和水平角)
angles_azimuth = np.array([-30, 45, 120])  # 水平角 (单位：度)
angles_elevation = np.array([10, 20, 30])  # 俯仰角 (单位：度)

# 3. 构建阵列响应矩阵（方向向量）
def array_response(M, d, angles_azimuth, angles_elevation, freq, c):
    k = 2 * np.pi * freq / c  # 波数
    a = np.zeros((M, len(angles_azimuth)), dtype=complex)
    for idx, (theta_azimuth, theta_elevation) in enumerate(zip(angles_azimuth, angles_elevation)):
        # 角度转换为弧度
        theta_azimuth_rad = np.deg2rad(theta_azimuth)
        theta_elevation_rad = np.deg2rad(theta_elevation)
        
        # 计算阵列响应
        for m in range(M):
            # 天线间距
            d_m = m * d
            # 水平和俯仰角对方向余弦的影响
            a[m, idx] = np.exp(1j * k * d_m * (np.sin(theta_azimuth_rad) * np.cos(theta_elevation_rad)))
    return a

# 4. 生成信号（加噪声）
def generate_signals(A, signal_count, SNR, M):
    # 假设信号为随机复数信号
    signal = (np.random.randn(signal_count, 1) + 1j * np.random.randn(signal_count, 1))
    # 构造接收信号
    noisy_signal = A @ signal.T  # 接收信号矩阵
    noise = np.random.randn(M, 1) + 1j * np.random.randn(M, 1)
    noisy_signal += 10 ** (-SNR / 20) * noise  # 添加噪声
    return noisy_signal

# 5. MUSIC算法进行角度估计
def music_algorithm(R, num_sources, M, angles_range):
    # 计算协方差矩阵
    R = R @ R.T.conj() / M
    # SVD分解
    _, _, vh = svd(R)
    noise_space = vh[num_sources:].T
    # MUSIC谱估计
    spectrum = np.zeros(len(angles_range))
    for idx, angle in enumerate(angles_range):
        steering_vector = np.exp(1j * 2 * np.pi * np.arange(M) * np.sin(np.deg2rad(angle)) / (d))  # 方向向量
        spectrum[idx] = 1 / np.linalg.norm(noise_space.conj().T @ steering_vector) ** 2
    return spectrum

# 6. 模拟接收信号并执行MUSIC算法
A = array_response(M, d, angles_azimuth, angles_elevation, freq, c)  # 构造阵列响应矩阵
noisy_signal = generate_signals(A, signal_count, SNR, M)  # 生成带噪信号
R = np.cov(noisy_signal)  # 计算接收信号的协方差矩阵

# 7. MUSIC角度估计
angles_range = np.linspace(-180, 180, 360)  # 水平角范围
spectrum = music_algorithm(R, signal_count, M, angles_range)  # 计算MUSIC谱

# 8. 绘制MUSIC谱
plt.figure(figsize=(8, 6))
plt.plot(angles_range, 10 * np.log10(np.abs(spectrum)))
plt.title('MUSIC Spectrum')
plt.xlabel('Azimuth Angle (degrees)')
plt.ylabel('Spectrum (dB)')
plt.grid(True)
plt.axvline(x=angles_azimuth[0], color='r', linestyle='--', label=f'True Angle: {angles_azimuth[0]}°')
plt.axvline(x=angles_azimuth[1], color='g', linestyle='--', label=f'True Angle: {angles_azimuth[1]}°')
plt.axvline(x=angles_azimuth[2], color='b', linestyle='--', label=f'True Angle: {angles_azimuth[2]}°')
plt.legend()
plt.show()

# 9. 输出估计角度
estimated_angles = angles_range[np.argmax(spectrum, axis=0)]
print("Estimated Azimuth Angles:", estimated_angles)
