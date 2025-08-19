import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.ndimage import maximum_filter, generate_binary_structure

# -------------------------
# 基本参数设置（更新MIMO，8×8阵列）
# -------------------------
f_c = 3e9  # 中心频率
lambda_c = 3e8 / f_c  # 波长
k = 2 * np.pi / lambda_c  # 波数
d = 0.5 * lambda_c  # 振元间距

# MIMO阵列尺寸：8×8
N_x = N_y = 8  # 发送端与接收端均为8×8阵列
M = N_x * N_y  # 每个阵列的元素数

# 子载波数量（快拍数）
N_subcarriers = 200

# -------------------------
# 阵列坐标（8×8阵列，中心化）
# -------------------------
y_idx = np.arange(N_y) - (N_y - 1) / 2
z_idx = np.arange(N_x) - (N_x - 1) / 2
y_pos = y_idx * d
z_pos = z_idx * d
Y, Z = np.meshgrid(y_pos, z_pos)  # 形状 N_x × N_y
Yv = Y.ravel()  # 展平成 M×1
Zv = Z.ravel()

def steering_3d(Xv, Yv, Zv, az, el, k, convention="polar_from_z"):
    """
    Xv, Yv, Zv: 形状 (M,) 的阵元坐标（米）
    az, el: 标量（弧度）
    k: 波数 2π/λ
    convention:
      - "polar_from_z": el=θ∈[0,π] 从 +z 量起的“极角”；az=φ∈(-π,π]
          sx=sinθ cosφ, sy=sinθ sinφ, sz=cosθ  ← 你现在用的这个
      - "elevation_from_xy": el=ψ∈[-π/2,π/2] 相对 x–y 平面的“仰角”；az=φ
          sx=cosψ cosφ, sy=cosψ sinφ, sz=sinψ
    """
    if convention == "polar_from_z":
        sx = np.sin(el) * np.cos(az)
        sy = np.sin(el) * np.sin(az)
        sz = np.cos(el)
    elif convention == "elevation_from_xy":
        sx = np.cos(el) * np.cos(az)
        sy = np.cos(el) * np.sin(az)
        sz = np.sin(el)
    else:
        raise ValueError("unknown convention")

    phase = k * (sx * Xv + sy * Yv + sz * Zv)
    return np.exp(1j * phase)  # shape: (M,)

# -------------------------
# 旋转矩阵（假设阵列有旋转角度）
# -------------------------
def rotation_matrix(azimuth, elevation):
    """返回旋转矩阵，根据方位角和俯仰角旋转"""
    Rz = np.array([
        [np.cos(azimuth), -np.sin(azimuth), 0],
        [np.sin(azimuth), np.cos(azimuth), 0],
        [0, 0, 1]
    ])
    Ry = np.array([
        [np.cos(elevation), 0, np.sin(elevation)],
        [0, 1, 0],
        [-np.sin(elevation), 0, np.cos(elevation)]
    ])
    return np.dot(Rz, Ry)


# -------------------------
# 计算MIMO阵列的导向向量（考虑旋转）
# -------------------------
def steering_vector_mimo(az, el, Yv, Zv, k, rotation_matrix=None):
    # 如果提供旋转矩阵，对阵列坐标进行旋转
    if rotation_matrix is not None:
        coords = np.vstack((Yv, Zv, np.zeros_like(Yv)))  # 3xM 的阵列坐标
        rotated_coords = np.dot(rotation_matrix, coords)
        Yv_rot, Zv_rot = rotated_coords[0, :], rotated_coords[1, :]
    else:
        Yv_rot, Zv_rot = Yv, Zv

    # 计算阵列流形（更新后的坐标）
    sy = np.sin(el) * np.sin(az)
    sz = np.cos(el)
    a = np.exp(1j * k * (sy * Yv_rot + sz * Zv_rot))
    return a


# -------------------------
# 读取CFR（H_rt）的新维度 [64, 64, 200]，即接收端与发射端都有阵列
# -------------------------
def add_awgn_complex(X, snr_db):
    sig_pow = np.mean(np.abs(X) ** 2)
    noise_pow = sig_pow / (10 ** (snr_db / 10.0))
    noise = (np.random.randn(*X.shape) + 1j * np.random.randn(*X.shape)) * np.sqrt(noise_pow / 2.0)
    return X + noise


# -------------------------
# MUSIC算法（估计DOA和AOA）
# -------------------------
def music_algorithm_mimo(H_rt, N_x, N_y, lambda_c, d, azimuth_range, elevation_range, rotation_angle=0):
    P_music_azimuth = np.empty(len(azimuth_range), dtype=float)
    P_music_elevation = np.empty(len(elevation_range), dtype=float)
    P_music_joint = np.zeros((len(azimuth_range), len(elevation_range)))

    # 协方差矩阵计算
    R = np.cov(H_rt)  # 协方差矩阵，取决于H_rt的维度
    eigvals, eigvecs = np.linalg.eig(R)
    sorted_indices = np.argsort(eigvals)
    noise_subspace = eigvecs[:, sorted_indices[:-1]]  # 噪声子空间

    # 旋转矩阵（给阵列加旋转角度）
    rotation_matrix_3d = rotation_matrix(np.deg2rad(rotation_angle), 0)

    # 计算MUSIC谱
    for i, azimuth in enumerate(azimuth_range):
        for j, elevation in enumerate(elevation_range):
            a_tau = steering_vector_mimo(azimuth, elevation, Yv, Zv, k, rotation_matrix_3d)
            y = np.dot(noise_subspace.T.conj(), a_tau)
            denom = np.abs(np.vdot(y, y))
            P_music_joint[i, j] = 1.0 / max(denom, 1e-20)

        # 计算方位角的P_music
        a_tau_azimuth = steering_vector_mimo(azimuth, 0, Yv, Zv, k, rotation_matrix_3d)
        y_azimuth = np.dot(noise_subspace.T.conj(), a_tau_azimuth)
        denom_azimuth = np.abs(np.vdot(y_azimuth, y_azimuth))
        P_music_azimuth[i] = 1.0 / max(denom_azimuth, 1e-20)

    for j, elevation in enumerate(elevation_range):
        a_tau_elevation = steering_vector_mimo(0, elevation, Yv, Zv, k, rotation_matrix_3d)
        y_elevation = np.dot(noise_subspace.T.conj(), a_tau_elevation)
        denom_elevation = np.abs(np.vdot(y_elevation, y_elevation))
        P_music_elevation[j] = 1.0 / max(denom_elevation, 1e-20)

    return P_music_azimuth, P_music_elevation, P_music_joint, azimuth_range, elevation_range


# -------------------------
# 峰值检测函数
# -------------------------
def find_peaks_2d(P, az_grid, el_grid, num_peaks=5, min_separation=3, threshold_rel_db=-10.0):
    P = np.asarray(P)
    P_db = 10 * np.log10(P / np.max(P) + 1e-12)

    # 候选：局部极大 & 超过相对阈值
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(P_db, size=min_separation, mode='nearest') == P_db
    mask = P_db >= threshold_rel_db
    cand = np.argwhere(local_max & mask)

    # 根据谱值排序，做一次贪心的“最小间隔”筛选
    scores = [P_db[i, j] for i, j in cand]
    order = np.argsort(scores)[::-1]
    kept = []
    for idx in order:
        i, j = cand[idx]
        ok = True
        for (ii, jj) in kept:
            if abs(ii - i) < min_separation and abs(jj - j) < min_separation:
                ok = False;
                break
        if ok:
            kept.append((i, j))
        if len(kept) >= num_peaks:
            break

    # 转成角度（度）
    results = []
    for i, j in kept:
        az_deg = np.rad2deg(az_grid[i])
        el_deg = np.rad2deg(el_grid[j])
        results.append({
            "az_deg": float(az_deg),
            "el_deg": float(el_deg),
            "P_dB": float(P_db[i, j])
        })
    return results


# -------------------------
# 主流程
# -------------------------
H_rt = np.load('H_rt.npy')  # 新CFR（[64,64,200]）

# 加噪（根据目标SNR加噪）
target_snr_db = 15.0  # 调整信噪比
H_rt_noisy = add_awgn_complex(H_rt, target_snr_db)

# 设定角度范围
azimuth_range = np.deg2rad(np.linspace(-180, 180, 361))
elevation_range = np.deg2rad(np.linspace(0, 180, 181))

# 运行MIMO MUSIC算法
P_music_azimuth, P_music_elevation, P_music_joint, azimuth_range, elevation_range = music_algorithm_mimo(
    H_rt_noisy, N_x, N_y, lambda_c, d, azimuth_range, elevation_range, rotation_angle=45)

# 峰值检测
peaks = find_peaks_2d(P_music_joint, azimuth_range, elevation_range, num_peaks=6, min_separation=3,
                      threshold_rel_db=-12.0)

# 输出峰值
for p in peaks:
    print(f"检测到的角度: 方位角={p['az_deg']:.2f}° 俯仰角={p['el_deg']:.2f}°，功率={p['P_dB']:.2f} dB")

# 画图
plt.figure()
plt.imshow(10 * np.log10(P_music_joint.T + 1e-12), aspect='auto', cmap='jet', origin='lower',
           extent=[np.rad2deg(azimuth_range[0]), np.rad2deg(azimuth_range[-1]),
                   np.rad2deg(elevation_range[0]), np.rad2deg(elevation_range[-1])])
plt.title('Joint MUSIC Spectrum (Azimuth vs Elevation)')
plt.xlabel('Azimuth (deg)')
plt.ylabel('Elevation (deg)')
plt.colorbar(label='dB')
# 标注峰
for p in peaks:
    plt.plot(p["az_deg"], p["el_deg"], 'wo', markersize=4)
plt.show()
