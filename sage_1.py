import numpy as np
import matplotlib.pyplot as plt

# 系统参数设定
frequency = 2.4e9  # 频率 (2.4 GHz)
bandwidth = 20e6  # 带宽 (20 MHz)
subcarriers = 128  # 子载波个数
antenna_array = 4  # 天线阵列尺寸 (4x4阵列)

# 计算时延分辨率
time_resolution = 1 / bandwidth  # 时延分辨率 (单位：秒)

# 多径参数假设
K = 20  # 多径条数
# 时延分布
delay_means = np.array([75e-9, 85e-9, 180e-9])  # 各簇的时延集中位置 (单位：秒)
delay_spread = 20e-9  # 时延扩展 (单位：秒)
delays = np.random.normal(loc=delay_means, scale=delay_spread, size=K)

# 角度分布（模拟LOS、反射路径的角度）
angles_AoD = np.random.uniform(low=0, high=2*np.pi, size=K)  # 水平离开角（0到2π之间）
angles_AoA = np.random.uniform(low=0, high=2*np.pi, size=K)  # 到达角（0到2π之间）
angles_EoD = np.random.uniform(low=0, high=np.pi/2, size=K)  # 俯仰离开角（0到π/2之间）
angles_EoA = np.random.uniform(low=0, high=np.pi/2, size=K)  # 俯仰到达角（0到π/2之间）

# 幅值和相位
amplitudes = np.random.rand(K)  # 随机生成幅值
phases = np.random.uniform(low=0, high=2*np.pi, size=K)  # 随机生成相位

# 构建信道频率响应H(f)（考虑4x4阵列，阵列沿y-o-z面放置）
def generate_cfr(frequency, delays, angles_AoD, angles_AoA, angles_EoD, angles_EoA, amplitudes, phases, bandwidth, subcarriers, antenna_array):
    # 生成子载波频率
    f = np.linspace(frequency - bandwidth/2, frequency + bandwidth/2, subcarriers)
    
    H = np.zeros((subcarriers, antenna_array, antenna_array), dtype=complex)
    
    # 假设阵列间距为半波长
    d = 0.5  # 半波长的阵列间距
    
    # 计算每个路径的信道响应
    for i in range(K):
        # 时延对应的频率响应
        delay = delays[i]
        amplitude = amplitudes[i]
        phase = phases[i]
        
        # 计算每个子载波的相位因子 e^{j 2π f τ_i}
        phase_factor = np.exp(1j * 2 * np.pi * f * delay)
        
        # 水平和俯仰角度
        theta_AoD = angles_AoD[i]  # 水平离开角
        phi_AoD = angles_EoD[i]    # 俯仰离开角
        theta_AoA = angles_AoA[i]  # 水平到达角
        phi_AoA = angles_EoA[i]    # 俯仰到达角
        
        # 计算发射天线阵列和接收天线阵列的响应（考虑阵列的y-z平面布置）
        a_AoD = np.zeros(antenna_array**2, dtype=complex)  # 发射天线阵列响应
        a_AoA = np.zeros(antenna_array**2, dtype=complex)  # 接收天线阵列响应
        
        # 天线阵列元素位置（假设阵列元素沿y轴和z轴均匀排列）
        for m in range(antenna_array):
            for n in range(antenna_array):
                # 发射天线阵列的响应
                a_AoD[m * antenna_array + n] = np.exp(1j * 2 * np.pi * d * (m * np.sin(theta_AoD) * np.cos(phi_AoD) + n * np.sin(theta_AoD) * np.sin(phi_AoD)))
                # 接收天线阵列的响应
                a_AoA[m * antenna_array + n] = np.exp(1j * 2 * np.pi * d * (m * np.sin(theta_AoA) * np.cos(phi_AoA) + n * np.sin(theta_AoA) * np.sin(phi_AoA)))
        
        # 信道频率响应 H(f) = 幅值 * 相位因子 * 阵列响应
        H += amplitude * phase_factor[:, None, None] * np.outer(a_AoD, a_AoA).reshape((subcarriers, antenna_array, antenna_array))
    
    return H

# 生成信道频率响应（CFR）
H = generate_cfr(frequency, delays, angles_AoD, angles_AoA, angles_EoD, angles_EoA, amplitudes, phases, bandwidth, subcarriers, antenna_array)

# 可视化信道频率响应的幅度（显示一个通道的幅度）
plt.imshow(np.abs(H[:, 0, 0]), aspect='auto', cmap='jet', origin='lower')
plt.title('Channel Frequency Response (Magnitude)')
plt.xlabel('Subcarriers')
plt.ylabel('Frequency')
plt.colorbar()
plt.show()

# 这里，H的维度为 (子载波数, 发射天线数, 接收天线数)，适用于SAGE算法的输入

# 计算信道响应的复数部分
def compute_response(f, delays, angles_AoD, angles_AoA, angles_EoD, angles_EoA, amplitudes, phases, antenna_array):
    H = np.zeros((len(f), antenna_array, antenna_array), dtype=complex)

    d = 0.5  # 半波长的阵列间距
    for i in range(len(delays)):
        delay = delays[i]
        amplitude = amplitudes[i]
        phase = phases[i]
        phase_factor = np.exp(1j * 2 * np.pi * f * delay)
        
        theta_AoD = angles_AoD[i]
        phi_AoD = angles_EoD[i]
        theta_AoA = angles_AoA[i]
        phi_AoA = angles_EoA[i]

        # 发射天线阵列和接收天线阵列的响应
        a_AoD = np.zeros(antenna_array**2, dtype=complex)
        a_AoA = np.zeros(antenna_array**2, dtype=complex)

        # 计算阵列响应
        for m in range(antenna_array):
            for n in range(antenna_array):
                a_AoD[m * antenna_array + n] = np.exp(1j * 2 * np.pi * d * (m * np.sin(theta_AoD) * np.cos(phi_AoD) + n * np.sin(theta_AoD) * np.sin(phi_AoD)))
                a_AoA[m * antenna_array + n] = np.exp(1j * 2 * np.pi * d * (m * np.sin(theta_AoA) * np.cos(phi_AoA) + n * np.sin(theta_AoA) * np.sin(phi_AoA)))
        
        # 更新信道响应
        H += amplitude * phase_factor[:, None, None] * np.outer(a_AoD, a_AoA).reshape((len(f), antenna_array, antenna_array))

    return H

# SAGE算法实现
def sage_algorithm(H, f, antenna_array, K, max_iter=50, tol=1e-6):
    # 初始化多径参数
    delays_est = np.linspace(70e-9, 200e-9, K)  # 假设时延估计
    angles_AoD_est = np.random.uniform(0, 2*np.pi, K)
    angles_AoA_est = np.random.uniform(0, 2*np.pi, K)
    angles_EoD_est = np.random.uniform(0, np.pi/2, K)
    angles_EoA_est = np.random.uniform(0, np.pi/2, K)
    amplitudes_est = np.random.rand(K)
    phases_est = np.random.uniform(0, 2*np.pi, K)
    
    prev_H = np.zeros_like(H)
    
    for iteration in range(max_iter):
        # E步：计算每条路径的责任
        responsibility = np.zeros((K, H.shape[0]))  # K条路径与每个子载波的责任值
        
        for k in range(K):
            # 使用当前估计的多径参数来计算每条路径的响应
            H_k = compute_response(f, [delays_est[k]], [angles_AoD_est[k]], [angles_AoA_est[k]], [angles_EoD_est[k]], [angles_EoA_est[k]], [amplitudes_est[k]], [phases_est[k]], antenna_array)
            
            # 计算责任值
            responsibility[k, :] = np.abs(np.sum(H_k * np.conj(H), axis=(1, 2)))  # 计算每条路径的贡献度
            
        # 归一化责任
        responsibility /= responsibility.sum(axis=0, keepdims=True)
        
        # M步：更新参数
        for k in range(K):
            # 更新时延估计
            delays_est[k] = np.sum(responsibility[k, :] * f) / np.sum(responsibility[k, :])  # 估计时延（加权平均）
            
            # 更新角度、幅值和相位
            # 估计发射和接收角度
            angles_AoD_est[k] = np.sum(responsibility[k, :] * angles_AoD_est) / np.sum(responsibility[k, :])
            angles_AoA_est[k] = np.sum(responsibility[k, :] * angles_AoA_est) / np.sum(responsibility[k, :])
            angles_EoD_est[k] = np.sum(responsibility[k, :] * angles_EoD_est) / np.sum(responsibility[k, :])
            angles_EoA_est[k] = np.sum(responsibility[k, :] * angles_EoA_est) / np.sum(responsibility[k, :])
            
            # 更新幅值和相位
            amplitudes_est[k] = np.sum(responsibility[k, :] * np.abs(H)) / np.sum(responsibility[k, :])
            phases_est[k] = np.sum(responsibility[k, :] * np.angle(H)) / np.sum(responsibility[k, :])
        
        # 检查收敛性
        if np.linalg.norm(H - prev_H) < tol:
            print(f'Converged after {iteration + 1} iterations.')
            break
        
        prev_H = H.copy()
    
    return delays_est, angles_AoD_est, angles_AoA_est, angles_EoD_est, angles_EoA_est, amplitudes_est, phases_est

# 运行SAGE算法
delays_est, angles_AoD_est, angles_AoA_est, angles_EoD_est, angles_EoA_est, amplitudes_est, phases_est = sage_algorithm(H, f, antenna_array, K)

# 打印估计的多径参数
print("Estimated delays (s):", delays_est)
print("Estimated angles (AoD, AoA, EoD, EoA):", list(zip(angles_AoD_est, angles_AoA_est, angles_EoD_est, angles_EoA_est)))
print("Estimated amplitudes:", amplitudes_est)
print("Estimated phases:", phases_est)
