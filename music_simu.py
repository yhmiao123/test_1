import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh

np.random.seed(1)

# ===================== 参数配置 =====================
c = 3e8
f = 3.5e9                     # 载频 (Hz)
lam = c / f
dx = dy = lam / 2             # 阵元间距 = λ/2

Mx, My = 8, 8                 # 平面阵列维度 (共 M = Mx*My 个阵元)
M = Mx * My

K = 3                         # 信号个数
SNR_dB = 10                   # 每阵元输入SNR(dB)
T = 200                       # 快拍数（快拍越大，估计越稳）

# 真实来向 (方位φ, 俯仰θ)，单位度
true_az = np.array([-40.0, 20.0, 110.0])     # φ ∈ [-180, 180]
true_el = np.array([35.0, 70.0, 50.0])       # θ ∈ [0, 180]

# 搜索网格（可适当加密/稀疏以平衡速度）
az_grid = np.linspace(-180, 180, 361)        # 每1°
el_grid = np.linspace(0, 180, 181)           # 每1°

# ===================== 工具函数 =====================
def deg2rad(x):
    return x * np.pi / 180.0

def steering_vector_upa(az_deg, el_deg, Mx, My, dx, dy, lam):
    """
    为UPA生成一个(K=1)的导向向量: a ∈ C^{M}
    az: 方位φ（度），el: 俯仰θ（度）
    """
    az = deg2rad(az_deg)
    el = deg2rad(el_deg)
    # 方向余弦
    u = np.sin(el) * np.cos(az)
    v = np.sin(el) * np.sin(az)

    m_idx = np.arange(Mx).reshape(-1, 1)      # 列向量
    n_idx = np.arange(My).reshape(1, -1)      # 行向量

    # 相位项: 2π/λ * (m*dx*u + n*dy*v)
    phase = 2j * np.pi / lam * (m_idx * dx * u + n_idx * dy * v)
    A2D = np.exp(phase)                       # Mx × My
    a = A2D.reshape(-1, order='F')            # 列优先展平 -> M×1
    return a / np.sqrt(a.size)                # 归一化，避免数值尺度膨胀

def build_steering_matrix(az_list, el_list):
    """
    A = [a(φ1,θ1),...,a(φK,θK)] ∈ C^{M×K}
    """
    A = np.stack([steering_vector_upa(az, el, Mx, My, dx, dy, lam) for az, el in zip(az_list, el_list)], axis=1)
    return A  # (M,K)

def db(x):
    return 10*np.log10(np.maximum(x, 1e-12))

# 简单的非极大值抑制找峰（2D）
def find_topk_peaks(P, K, nbh=3):
    """
    从2D谱矩阵P中找前K个峰。nbh为抑制邻域大小。
    返回：[(i,j,值), ...]，i为el索引，j为az索引
    """
    Pcopy = P.copy()
    peaks = []
    for _ in range(K):
        idx = np.unravel_index(np.argmax(Pcopy), Pcopy.shape)
        val = Pcopy[idx]
        peaks.append((idx[0], idx[1], val))
        i, j = idx
        i0 = max(0, i - nbh); i1 = min(Pcopy.shape[0], i + nbh + 1)
        j0 = max(0, j - nbh); j1 = min(Pcopy.shape[1], j + nbh + 1)
        Pcopy[i0:i1, j0:j1] = -np.inf
    return peaks

# ===================== 生成数据 =====================
A = build_steering_matrix(true_az, true_el)           # (M,K)

# 生成K×T的复高斯符号（也可改成PSK等）
S = (np.random.randn(K, T) + 1j*np.random.randn(K, T)) / np.sqrt(2)

# 噪声
signal_power = 1.0                 # 归一化信号功率
SNR_lin = 10**(SNR_dB/10)
noise_var = signal_power / SNR_lin
N = (np.sqrt(noise_var/2) * (np.random.randn(M, T) + 1j*np.random.randn(M, T)))

# 接收信号
X = A @ S + N                      # (M,T)

# 协方差矩阵 (M×M Hermitian)
R = (X @ X.conj().T) / T

# ===================== 子空间分解（MUSIC） =====================
# R是厄米特矩阵，用eigh（对称/厄米特）更稳
eigvals, eigvecs = eigh(R)        # 升序
# 噪声子空间：前 M-K 个特征向量
En = eigvecs[:, :M-K]             # (M, M-K)

# ===================== 扫描网格计算谱 =====================
P = np.zeros((len(el_grid), len(az_grid)))   # P[iel, iaz]

# 为了加速，这里分两层循环（10万级点还可接受；如需更快可做矢量化/分块）
for i, el in enumerate(el_grid):
    for j, az in enumerate(az_grid):
        a = steering_vector_upa(az, el, Mx, My, dx, dy, lam)   # (M,)
        denom = np.linalg.norm(En.conj().T @ a)**2             # aᴴ En Enᴴ a
        P[i, j] = 1.0 / max(denom, 1e-12)

# 归一化到最大值0 dB
P_dB = db(P / P.max())

# ===================== 找峰并给出估计 =====================
peaks = find_topk_peaks(P, K=K, nbh=3)
estimates = []
for (iel, iaz, val) in peaks:
    est_el = el_grid[iel]
    est_az = az_grid[iaz]
    estimates.append((est_az, est_el, val))

# 按谱值从高到低排序
estimates.sort(key=lambda x: x[2], reverse=True)

# ===================== 打印结果 =====================
print("真实来向(φ, θ) (deg):")
for az, el in zip(true_az, true_el):
    print(f"  φ={az:7.2f}, θ={el:7.2f}")

print("\nMUSIC估计(φ, θ) (deg):")
for az, el, _ in estimates:
    print(f"  φ={az:7.2f}, θ={el:7.2f}")

# ===================== 绘图 =====================
plt.figure(figsize=(9, 6))
# imshow: 行是el，列是az；为让az从-180到180正向显示，extent设定
plt.imshow(P_dB, origin='lower',
           extent=[az_grid[0], az_grid[-1], el_grid[0], el_grid[-1]],
           aspect='auto')
plt.colorbar(label='Spatial Spectrum (dB)')
plt.xlabel('Azimuth φ (deg)')
plt.ylabel('Elevation θ (deg)')
plt.title('2D-MUSIC Spectrum (UPA)')

# 标注真值（红×）与估计值（白○）
for az, el in zip(true_az, true_el):
    plt.plot(az, el, 'rx', markersize=10, mew=2, label='True DOA' if 'True DOA' not in plt.gca().get_legend_handles_labels()[1] else None)
for az, el, _ in estimates:
    plt.plot(az, el, 'wo', markersize=8, mfc='none', mew=2, label='Estimated DOA' if 'Estimated DOA' not in plt.gca().get_legend_handles_labels()[1] else None)

plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
