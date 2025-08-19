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
        clustering = DBSCAN(eps=1e-9, min_samples=2).fit(remaining_values.reshape(-1, 1))  # 使用谱值进行聚类
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


[0.00961761 0.00919585 0.00864597 0.00818616 0.0080192  0.00810508, 0.00824878 0.00844613 0.00876708 0.00896995 0.00895965 0.00929417, 0.01016228 0.0101804  0.00871938 0.0076763  0.00786408 0.00865903, 0.00881014 0.0084839  0.00856183 0.0089257  0.00923807 0.00966485, 0.00980857 0.00903476 0.00830627 0.00848196 0.00901026 0.00882162, 0.00838869 0.00852834 0.00903877 0.00950051 0.00995368 0.01009646, 0.00959776 0.0092143  0.00945922 0.00976415 0.00959487 0.00942751, 0.00952928 0.00956347 0.00946661 0.00947073 0.00959616 0.00986521, 0.01024033 0.01019023 0.00951439 0.00891154 0.00872781 0.00869182, 0.00866802 0.00883885 0.00916768 0.00941034 0.00955766 0.00967297, 0.00967432 0.00960561 0.0095666  0.00952842 0.00950057 0.00931136, 0.00868696 0.00845717 0.010866   0.02680357 0.0485395  0.01441868, 0.01036059 0.01032596 0.01045027 0.01078253 0.01436767 0.0303598, 0.13120263 0.46949017 0.40373202 0.4277199  0.13844893 0.07783728, 0.07904086 0.14072512 0.38712868 0.42586611 0.23979327 0.16311866, 0.12556216 0.09033935 0.07007928 0.06843774 0.07225344 0.06960935, 0.0768367  0.10271198 0.08247928 0.04614432 0.0326551  0.02665666, 0.02142759 0.01923219 0.02324126 0.03822345 0.0471205  0.02624367, 0.01472464 0.01070913 0.00970516 0.00996534 0.01008324 0.00943727, 0.00892354 0.00912344 0.00952494 0.00933786 0.00886902 0.00869515, 0.00870784 0.00879743 0.00933879 0.01031722 0.01054102 0.00962212, 0.00891587 0.00895356 0.00926285 0.00936925 0.00932128 0.00920339, 0.00906448 0.00910513 0.00928913 0.00931304 0.00926987 0.00945886, 0.00955397 0.0091363  0.00880344 0.00915649 0.00964945 0.00925807, 0.00861102 0.00873931 0.00944849 0.00969652 0.00937975 0.00928367, 0.0094252  0.00931359 0.00892575 0.00857931 0.00844526 0.00852012, 0.0085946  0.00852615 0.00857946 0.00884751 0.0087929  0.00832433, 0.00824768 0.00892042 0.00965973 0.00953931 0.00894629 0.00867136, 0.00898268 0.00968543 0.01003654 0.00966724 0.00933651 0.00948452, 0.0096311  0.00933618 0.00893761 0.0087913  0.00887967 0.00902898, 0.00924162 0.01011847 0.0126377  0.01612732 0.01638483 0.01574976, 0.01951752 0.02588928 0.02109792 0.01622179 0.01745685 0.01974278, 0.01535704 0.01231266 0.01365626 0.01637492 0.01328374 0.00971533, 0.0088622  0.00946602 0.00980274 0.00953602 0.00976105 0.01074094, 0.01148817 0.01095899 0.00984681 0.00927946 0.00940912 0.00950456, 0.0090168  0.00862029 0.00895676 0.00970298 0.00990441 0.00947267, 0.00916102 0.00929637 0.00971883 0.00999971 0.00978555 0.00932129, 0.00913256 0.00939863 0.00983782 0.00988932 0.00937228 0.00886862, 0.00894637 0.00947767 0.00959996 0.00892901 0.00822909 0.00806521, 0.00843528 0.00902849 0.00939034 0.00938951 0.00933146 0.00922162, 0.00870334 0.00803395 0.00789689 0.0084451  0.00905259 0.00905685, 0.0088573  0.00889936 0.00895425 0.00893979 0.00926524 0.00989581, 0.01001977 0.00956363 0.00937738 0.00947398 0.00914288 0.00856008, 0.00850098 0.00901313 0.00939497 0.00924789 0.00907136 0.00929998, 0.00978059 0.00980774 0.00900951 0.00819741 0.00819375 0.00901116, 0.00960735 0.00903225 0.0083512  0.00864589 0.00980142 0.01034044, 0.00943039 0.0084935  0.00826816 0.00845817 0.00872601 0.00908655, 0.00943397 0.00944175 0.00919164 0.0090261  0.00910382 0.00951533, 0.01008877 0.01002407 0.00909241 0.00829682 0.00822444 0.00869795, 0.00922406 0.00948007 0.00954205 0.0096418  0.00986179 0.01004027, 0.01001995 0.00981198 0.00942056 0.00894257 0.00861608 0.00848683, 0.00849047 0.00885875 0.00979805 0.01056202 0.01015922 0.00947583, 0.00929892 0.00898901 0.00821557 0.00787695 0.00843865 0.0091811, 0.00898356 0.00837422 0.00824972 0.0086232  0.00913555 0.00936242, 0.00899462 0.00850218 0.00869481 0.00950496 0.009548   0.00861389, 0.00832147 0.00909829 0.00960861 0.0088077  0.00816201 0.00851158, 0.00926855 0.00948387 0.00938234 0.00942211 0.00927748 0.00890949, 0.00892404 0.00948704 0.00996894 0.00994138 0.00988359 0.01003273, 0.0100783  0.00996991 0.0098548  0.0096243  0.0094102  0.00954933, 0.00977224 0.00956708 0.00933467 0.00948526 0.00950843 0.00920348, 0.00925684 0.00959814 0.0093193  0.00877768 0.0088791  0.00922775, 0.00893144 0.00851089 0.0086574  0.0090391  0.00917863 0.00920392, 0.00904192 0.00852612 0.00819021 0.0085264  0.00933008 0.00981394, 0.00954308 0.00900566 0.00881176 0.00908724 0.00942579 0.0094213, 0.00928506 0.00929221 0.00933168 0.00927538 0.00919172 0.00920898, 0.00941589 0.00955463 0.00913687 0.00852896 0.00854279 0.00921436, 0.00972364 0.00970774 0.00969399 0.00960256 0.00911847 0.00866735, 0.00856717 0.00872093 0.00923317 0.01007467 0.01019404 0.00924904, 0.00858829 0.00869172 0.00903826 0.00925007 0.00925971 0.00868976, 0.00789202 0.00780626 0.00854671 0.00914967 0.00904738 0.00898897, 0.00900721 0.00860991 0.00832217 0.00870351 0.00918014 0.0088856, 0.00841416 0.00853733 0.00899448 0.00903868 0.00877682 0.00888667, 0.00926551 0.00904899 0.0083419  0.00815057 0.00871407 0.0093288, 0.00938795 0.00927478 0.00921416 0.00905485 0.00897395 0.0091045, 0.00916458 0.00909697 0.00922202 0.00955889 0.00982537 0.00983152, 0.00943831 0.00887588 0.00882806 0.00955517 0.01027136 0.01007569, 0.0096253  0.00950914 0.00936736 0.00907388 0.00908192 0.00948507, 0.00974451 0.00946425 0.00904457 0.00901801 0.00938569 0.00958868, 0.00936961 0.00930321 0.00963453 0.00973245 0.00935002 0.00923237, 0.00966002 0.01007247 0.01011688 0.00998724 0.00952646 0.00880285, 0.00851268 0.00890061 0.00937825 0.00944528 0.00951817 0.00968014, 0.00928887 0.00852972 0.00824661 0.00862798 0.00919477 0.00941053, 0.00927234 0.00899183 0.00869282 0.00850555 0.00852445]
