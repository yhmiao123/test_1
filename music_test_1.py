import numpy as np
from matplotlib import pyplot as plt
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera,\
                      PathSolver, RadioMapSolver, subcarrier_frequencies
# 加载数据
# 该版本是用得天线维度当做多观测
load_path = r'.\Result\env_test1\results_MIMO_44_44_path.npy'

# 加载数据（直接加载为字典）
data = np.load(load_path, allow_pickle=True).item()

# 现在你可以访问里面的任意变量
H_rt = data['H_rt']      # shape: (频点, 天线, 快照)
h_rt = data['h_rt']      # shape: (频点, 天线, 快照)
paths = data['paths']    # 可能是列表，每条路径包含 AOA/AOD/延迟/功率等信息
tau = data['tau']

print("数据已成功加载！")
print("H_rt shape:", H_rt.shape)
print("h_rt shape:", h_rt.shape)


subcarrier_spacing = 300e3   # 子载波间隔

num_subcarriers = 1024      # 子载波数量
frequencies = subcarrier_frequencies(num_subcarriers, subcarrier_spacing)
frequencies = frequencies.numpy()

Fs = frequencies[num_subcarriers-1]-frequencies[0]
Ts = 1/Fs
Ts_ns = Ts * 1e9  # 采样间隔 (10纳秒)
tao_max=max(tau)/1e-9  #ns
t_ns = np.arange(0, tao_max*5, Ts_ns)  # 时间轴 (纳秒)
t = t_ns * 1e-9  # 转换为秒

L=16*16  # 快拍数 越多协方差估计越稳定


#-----------music算法函数------------
def steering_vector(freqs,tau):
    """
    a_n(tao)=exp(-j2pi*f_n*tao)
    """
    return np.exp(-1j*2.0*np.pi*freqs*tau)  # shap(N_f,)

def parabolic_refine(x,i):
    xm1,x0,xp1=x[i-1],x[i],x[i+1]
    denom=(xm1-2*x0+xp1)
    if np.abs(denom)<1e-12:
        return 0.0,float(x0)
    delta=0.5*(xm1-xp1)/denom
    peak_val=x0-0.25*(xm1-xp1)*delta
    return float(delta),float(peak_val)

def add_awgn_complex(X, snr_db):
    sig_pow = np.mean(np.abs(X) ** 2)
    noise_pow = sig_pow / (10 ** (snr_db / 10.0))
    noise = (np.random.randn(*X.shape) + 1j * np.random.randn(*X.shape)) * np.sqrt(noise_pow / 2.0)
    return X + noise


def get_snapshots(H_rt):
    a, b, c = H_rt.shape
    if c == 0:
        raise ValueError("输入数组的最后一个维度 c 必须大于 0")

    # 转置轴顺序：(2, 0, 1) → (c, a, b)，然后 reshape
    snapshots = H_rt.transpose(2, 0, 1).reshape(c, a * b)
    return snapshots

def music_delay_estimation(H_rt,P_true,N_f,num_snapshots,tau_grid):

    target_snr_db = 10.0  # 调整信噪比

    H_rt_noisy = add_awgn_complex(H_rt, target_snr_db)

    snapshots=get_snapshots(H_rt_noisy)  # shap[N,L]

    #计算协方差矩阵
    R = (snapshots @ snapshots.conj().T)/num_snapshots  # shap[N,N]

    # 特征值分解
    eigvals, eigvecs = np.linalg.eig(R)

    #  特征值降序排列
    idx_desc = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx_desc]
    eigvecs = eigvecs[:, idx_desc]

    Es = eigvecs[:, :P_true]  # 信号子空间
    En = eigvecs[:, P_true:]  # 噪声子空间
    EnH = En.conj().T

    P_music = np.empty(len(tau_grid), dtype=float)
    for k, tau in enumerate(tau_grid):
        a_tau = steering_vector(frequencies, tau)
        y = EnH @ a_tau
        denom = np.vdot(y, y).real
        P_music[k] = 1.0 / max(denom, 1e-20)

    # 归一化画图
    P_music /= np.max(P_music)

    try:
        from scipy.signal import find_peaks
        min_sep_time = 0.1 / Fs
        min_sep_sample = max(1, int(min_sep_time / (tau_grid[1] - tau_grid[0])))
        peaks, props = find_peaks(P_music, distance=min_sep_sample, height=0.3)

        if len(peaks) >= P_true:
            top = np.argsort(P_music[peaks])[-P_true:]
            peak_idx = np.sort(peaks[top])
        else:
            # peak_idx = np.argsort(P_music)[-P_true:]
            # peak_idx.sort()
            top = np.argsort(P_music[peaks])[-P_true:]  # 取前 P_true 个，不够就全取
            peak_idx = np.sort(peaks[top])

            # 剩余需要补的数量
            need_more = P_true - len(peak_idx)

            if need_more > 0:
                # 在 P_music 中找出不在 peak_idx 中的最大值位置
                # 先构建一个 mask：排除已经选过的索引
                mask = np.ones(len(P_music), dtype=bool)
                mask[peak_idx] = False  # 排除已选中的索引

                # 在非 peak_idx 的位置中找最大的 need_more 个值
                candidate_indices = np.where(mask)[0]
                candidate_values = P_music[candidate_indices]

                # 找最大的 need_more 个值的索引
                top_candidates = np.argsort(candidate_values)[-need_more:][::-1]  # 从大到小
                new_peaks = candidate_indices[top_candidates]

                # 合并新选的峰值
                peak_idx = np.concatenate([peak_idx, new_peaks])
                peak_idx.sort()  # 最后统一排序
    except Exception:
        cand = np.where((P_music[1:-1]) > P_music[:-2] & (P_music[1:-1] > P_music[2:0]))[0] + 1
        if len(cand) >= P_true:
            peak_idx = cand[np.argsort(P_music[cand])][-P_true:]
            peak_idx.sort()
        else:
            peak_idx = np.argsort(P_music)[-P_true:]
            peak_idx.sort()

    # 二次差值细化峰位置
    tau_hat = []
    for i in peak_idx:
        if 1 <= i <= (len(tau_grid) - 2):
            delta, _ = parabolic_refine(P_music, i)
        else:
            delta = 0.0
        # 网格间隔
        d_tau = tau_grid[1] - tau_grid[0]
        tau_est = tau_grid[i] + delta * d_tau
        tau_hat.append(tau_est)
    tau_hat = np.array(tau_hat)

    return tau_hat, P_music

#-------------主函数进行music估计-----------
P_true=30  #需要估计的时延数目

tau_min=0.0
tau_max=500e-9  # 最大时延230ns 扫描0-500ns的
N_tau=501  # 扫描网格数 扫描间隔为1e-9  1ns
tau_grid=np.linspace(tau_min,tau_max,N_tau)  #shap(N_tau)

tau_hat, P_music = music_delay_estimation(H_rt=H_rt,P_true=P_true,N_f=num_subcarriers,num_snapshots=L,tau_grid=tau_grid)

sorted_tau_ns = np.sort(tau * 1e9)  # 先转为 ns，再排序
print("排序后的实时延 (ns):", sorted_tau_ns)
print(" 估计时延 (ns):", tau_hat*1e9)

plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False


#  画图 1）原始cir图 2）music时延谱  3）进行了峰值提取后的music时延谱
plt.figure(figsize=(10, 6))
plt.subplot(311)
h_rt=h_rt[1,1,:]
plt.stem(tau/1e-9, np.abs(h_rt), basefmt="b-", linefmt="b-", markerfmt="bo")
plt.title("理想的 CIR (无限时延分辨率)")
plt.xlabel("时延 (ns)")
plt.ylabel("幅度")
plt.xlim(-25, 525)  # 强制 x 从 0 开始，上限为最大延迟

plt.subplot(312)
plt.stem(tau_grid*1e9, P_music,  basefmt="b-", linefmt="b-", markerfmt="bo")
plt.grid(True, alpha=0.3)
plt.xlabel('时延 τ (ns)')
plt.ylabel('归一化P(τ)')
plt.title('MUSIC 算法用于多径时延估计')

plt.subplot(313)
plt.plot(tau_grid*1e9, P_music, 'b-', linewidth=1.5, label='MUSIC 时延谱')
# 添加标注
# for t in tau_hat:
#     idx=np.argmin(np.abs(tau_grid-t))
#     plt.plot(t*1e9,P_music[idx],'rx',markersize=10)
#     plt.text(t*1e9,P_music[idx]+0.05,f"{t*1e9:.3f}ns",color='r',ha='center',fontsize=9)

plt.grid(True, alpha=0.3)
plt.xlabel('时延 τ (ns)')
plt.ylabel('归一化P(τ)')
plt.title('MUSIC 算法用于多径时延估计')
plt.tight_layout()
plt.show()

from sklearn.cluster import DBSCAN
import numpy as np
from scipy.signal import find_peaks

def find_delay_estimates_with_clustering(P_music, tau_grid, P_true, Fs):
    # 1. **峰值检测**：先使用 find_peaks 找到显著的峰值
    adaptive_threshold = np.max(P_music) * 0.3
    min_sep_time = 0.1 / Fs
    min_sep_sample = max(1, int(min_sep_time / (tau_grid[1] - tau_grid[0])))
    peaks, _ = find_peaks(P_music, height=adaptive_threshold, distance=min_sep_sample, prominence=0.2)

    # 2. **如果峰值数量大于 P_true**：选择最强的 P_true 个峰值
    if len(peaks) >= P_true:
        sorted_peaks = np.argsort(P_music[peaks])[-P_true:]
        tau_hat = tau_grid[peaks][sorted_peaks]
    else:
        # 3. **如果峰值数量小于 P_true**：使用聚类方法补充
        tau_hat = tau_grid[peaks]  # 先存储已检测到的峰值

        # 聚类补充剩余的时延
        remaining_peaks = np.delete(np.arange(len(P_music)), peaks)  # 获取未检测的峰值索引
        remaining_values = P_music[remaining_peaks]  # 获取未检测的谱值

        # 进行聚类，找到剩余的时延点
        clustering = DBSCAN(eps=1e-9, min_samples=2).fit(remaining_peaks.reshape(-1, 1))
        cluster_centers = []

        # 找到每个簇的中心
        for label in set(clustering.labels_):
            if label != -1:  # 排除噪声点
                cluster_indices = np.where(clustering.labels_ == label)[0]
                cluster_centroid = np.mean(remaining_peaks[cluster_indices])  # 计算每个簇的中心
                cluster_centers.append(cluster_centroid)

        # 4. **去重操作**：确保聚类中心点不与已有峰值重复
        cluster_centers = np.array(cluster_centers)
        valid_centers = []

        for center in cluster_centers:
            distances = np.abs(tau_grid[peaks] - tau_grid[remaining_peaks[center]])
            if np.all(distances > 1):  # 阈值 1ns，避免重复峰值
                valid_centers.append(center)

        # 合并峰值：将补充的聚类中心加入到已检测峰值中
        tau_hat = np.concatenate([tau_hat, tau_grid[valid_centers]])

        # 5. **确保时延数量为 P_true**：排序并返回前 P_true 个时延
        tau_hat = np.sort(tau_hat)[:P_true]

    return tau_hat

# 示例调用
P_true = 30  # 需要估计的时延数目
tau_hat = find_delay_estimates_with_clustering(P_music, tau_grid, P_true, Fs)

# 打印结果
print("估计的时延（ns）:", tau_hat * 1e9)

