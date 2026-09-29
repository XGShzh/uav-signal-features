# -*- coding: utf-8 -*-
"""
channel.py
低空无人机信号特征 —— 第一部分（M1 信号建模）：真实信道与接收机损伤
====================================================================
参考文献依据（文献清单见 uav-literature-list.html）
--------------------------------------------------------------------
[#1] RFUAV: A Benchmark Dataset for UAV Detection and Identification,
     R. Shi et al., arXiv:2503.09033 (2025).
     —— 实测无人机 RF 信号存在明显频偏、相位噪声与硬件损伤，接收信号
        并非理想波形；本模块据此引入非理想损伤，使仿真贴近实测。
[#2] DroneRFa：用于侦测低空无人机的大规模无人机射频信号数据集,
     电子与信息学报, doi:10.11999/JEIT230570.
     —— 城市/户外实测表明遥控与图传链路受多径与接收机 IQ 失衡影响。
[#3] A Multi-Scenario UAV RF Dataset ..., arXiv:2603.00106 (2026).
     —— 多场景采集证明信道随时间/位置变化，需时变多径与时变多普勒。
[#7] Securing the Skies: A Comprehensive Survey on Anti-UAV Methods,
     Y. Dong et al., arXiv:2504.11967 (2025).
     —— 综述指出实际低空环境的多径与低 SNR 是检测识别的主要难点。

本模块在理想基带信号上叠加以下真实损伤（可通过参数开关）：
    1) 载波频偏 CFO（本振误差 + 残余同步误差）
    2) 时变多普勒（无人机机动导致）
    3) 振荡器相位噪声（Wiener 模型，linewidth 参数）
    4) 接收机 IQ 不平衡（镜像干扰，产生 EVM 误差底）
    5) 接收机非线性（三阶截点 IIP3 软压缩）
    6) 时变多径（抽头延迟线，航班机动导致时变）

设计说明：IQ 不平衡、相位噪声、非线性会产生“误差底”（error floor），
即高 SNR 下性能不再提升——这正是实测信号区别于理想仿真的关键，也是
本作业避免“过于理想”的依据。

依赖：numpy
"""

import numpy as np


# ======================================================================
# 1) 载波频偏 CFO
# ======================================================================
def add_cfo(sig, fs, cfo_hz):
    """施加载波频偏（本振误差 + 同步残留）。"""
    if cfo_hz == 0:
        return np.asarray(sig, dtype=np.complex128)
    t = np.arange(len(sig)) / fs
    return np.asarray(sig, dtype=np.complex128) * np.exp(1j * 2 * np.pi * cfo_hz * t)


# ======================================================================
# 2) 时变多普勒（无人机机动）
# ======================================================================
def add_doppler(sig, fs, fd_max_hz=100.0, fd_rate_hz=4.0, seed=None):
    """时变多普勒：正弦调制的瞬时频偏，模拟无人机周期性机动。

    fd_max_hz : 最大多普勒频移（Hz），无人机低速时典型值 10~500 Hz。
    fd_rate_hz: 机动摆动频率（Hz）。
    """
    rng = np.random.default_rng(seed)
    t = np.arange(len(sig)) / fs
    ph0 = rng.uniform(0, 2 * np.pi)
    fd = fd_max_hz * np.sin(2 * np.pi * fd_rate_hz * t + ph0)
    phase = 2 * np.pi * np.cumsum(fd) / fs
    return np.asarray(sig, dtype=np.complex128) * np.exp(1j * phase)


# ======================================================================
# 3) 相位噪声（Wiener / 随机游走模型）
# ======================================================================
def add_phase_noise(sig, fs, linewidth_hz=200.0, seed=None):
    """振荡器相位噪声：Wiener 随机游走相位。

    linewidth_hz : 3dB 线宽（Hz）。典型 SDR 本振 100~1000 Hz。
    """
    rng = np.random.default_rng(seed)
    n = len(sig)
    # 每采样相位增量标准差 sigma = sqrt(2*pi*linewidth/fs)
    sigma = np.sqrt(2 * np.pi * linewidth_hz / fs)
    phi = np.cumsum(rng.standard_normal(n)) * sigma
    return np.asarray(sig, dtype=np.complex128) * np.exp(1j * phi)


# ======================================================================
# 4) 接收机 IQ 不平衡（镜像干扰 → EVM 误差底）
# ======================================================================
def add_iq_imbalance(sig, gain_db=0.3, phase_deg=1.5):
    """IQ 不平衡：幅度失配 gain_db、相位正交误差 phase_deg。

    模型 y = alpha·x + beta·conj(x)，其中
        alpha = (1 + g·e^{jφ})/2,  beta = (1 - g·e^{jφ})/2
    产生镜像分量，形成不可通过增大 SNR 消除的误差底。
    gain_db 典型 0.1~0.5 dB，phase_deg 典型 1~3°（低成本 SDR）。
    """
    g = 10 ** (gain_db / 20.0)
    phi = np.deg2rad(phase_deg)
    alpha = (1 + g * np.exp(1j * phi)) / 2.0
    beta = (1 - g * np.exp(1j * phi)) / 2.0
    x = np.asarray(sig, dtype=np.complex128)
    return alpha * x + beta * np.conj(x)


# ======================================================================
# 5) 接收机非线性（三阶截点软压缩）
# ======================================================================
def add_nonlinearity(sig, backoff_db=20.0):
    """接收机非线性：三阶软压缩模型。

    y = x·(1 − k·|x|²)，k = 10^(−backoff_db/10)

    backoff_db : 信号功率低于三阶截点 IIP3 的 dB 数（输入回退）。
        实际接收机中信号远低于 IIP3（典型回退 15~30 dB），此时压缩很轻，
        仅产生小幅 EVM 底与频谱再生；若回退不足（信号接近/超过 IIP3），
        压缩极强会把波形压坏（错误建模）。
    """
    x = np.asarray(sig, dtype=np.complex128)
    p = np.mean(np.abs(x) ** 2) + 1e-30
    xn = x / np.sqrt(p)                     # 归一化到单位功率
    k = 10 ** (-backoff_db / 10.0)
    y = xn - k * xn * (np.abs(xn) ** 2)
    return y * np.sqrt(p)                   # 恢复功率量级


# ======================================================================
# 6) 时变多径信道
# ======================================================================
def multipath_channel(sig, fs, taps=((0.0, 1.0, 0.0), (0.4e-6, 0.5, 0.8),
                                     (0.9e-6, 0.3, -1.2), (1.6e-6, 0.2, 0.5)),
                      time_varying=False, doppler_hz=30.0, seed=None):
    """抽头延迟线多径信道（可时变）。

    每个抽头：(时延秒, 相对幅度, 初始相位弧度)。
    默认 4 径、时延 0~1.6us（低空城市场景典型值，参见 [#2][#3]）。
    time_varying=True 时各径相位按多普勒速率旋转，模拟无人机机动。
    """
    x = np.asarray(sig, dtype=np.complex128)
    L = x.shape[0]
    taps = sorted(taps, key=lambda p: p[0])
    max_del = int(np.ceil(taps[-1][0] * fs))
    h = np.zeros(max_del + 1, dtype=np.complex128)
    for (delay, amp, phase) in taps:
        d0 = int(round(delay * fs))
        if d0 <= max_del:
            h[d0] += amp * np.exp(1j * phase)
    h = h / (np.sum(np.abs(h)) + 1e-12)

    if not time_varying:
        return np.convolve(x, h, mode="same")[:L]

    # 时变：逐径相位旋转（简化分数时延，按整数时延施加）
    rng = np.random.default_rng(seed)
    t = np.arange(L) / fs
    y = np.zeros(L, dtype=np.complex128)
    for (delay, amp, phase) in taps:
        d0 = int(round(delay * fs))
        rot = np.exp(1j * (2 * np.pi * rng.uniform(0.5, 1.5) * doppler_hz * t + phase))
        xd = np.concatenate([np.zeros(d0, dtype=np.complex128), x[:L - d0]]) if d0 > 0 else x
        y += amp * xd * rot
    y = y / (np.sum(np.abs(h)) + 1e-12)
    # 时变信道不保时不变能量，重新归一化到与原信号同功率
    y = y * np.sqrt((np.mean(np.abs(x) ** 2) + 1e-30) / (np.mean(np.abs(y) ** 2) + 1e-30))
    return y[:L]


# ======================================================================
# 7) AWGN
# ======================================================================
def add_awgn(sig, snr_db, rng=None):
    """叠加 AWGN，返回加噪后信号与噪声功率（snr_db 参考信号平均功率）。"""
    if rng is None:
        rng = np.random.default_rng()
    sig = np.asarray(sig)
    p_sig = np.mean(np.abs(sig) ** 2)
    p_noise = p_sig / (10 ** (snr_db / 10.0))
    if np.iscomplexobj(sig):
        n = (rng.standard_normal(sig.shape) + 1j * rng.standard_normal(sig.shape))
    else:
        n = rng.standard_normal(sig.shape)
    noise = n / np.sqrt(np.mean(np.abs(n) ** 2) + 1e-12) * np.sqrt(p_noise)
    return sig + noise, p_noise


# ======================================================================
# 8) 统一入口：真实接收机链路
# ======================================================================
# 默认损伤参数（低成本 SDR 接收机 + 低空城市场景的典型量级）
DEFAULT_IMPAIR = dict(
    cfo_hz=800.0,          # 残余载波频偏 ≈ 0.2% 子载波间隔
    fd_max_hz=80.0,        # 无人机机动多普勒
    fd_rate_hz=4.0,
    linewidth_hz=150.0,    # 本振相位噪声线宽
    iq_gain_db=0.25,       # IQ 幅度失配
    iq_phase_deg=1.2,      # IQ 相位失配
    backoff_db=20.0,       # 非线性输入回退（低于 IIP3 的 dB 数）
    doppler_hz=30.0,       # 多径时变速率
)


def add_channel(sig, fs, snr_db=None, use_multipath=True, seed=5,
                impair=True, time_varying_mp=False, impair_kw=None):
    """真实接收机链路：多径 → IQ 不平衡 → 非线性 → CFO/多普勒 → 相噪 → AWGN。

    参数
    ----
    snr_db        : 若为 None 则不加噪声。
    use_multipath : 是否经多径信道。
    impair        : 是否施加接收机损伤（IQ/非线性/CFO/多普勒/相噪）。
                    设为 False 可复现“理想信道”情形做对比。
    impair_kw     : 覆盖默认损伤参数（dict）。
    返回 (接收信号, 噪声功率)。
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(sig, dtype=np.complex128)
    y = x.copy()

    # --- 多径 ---
    if use_multipath:
        y = multipath_channel(y, fs, time_varying=time_varying_mp,
                              seed=rng.integers(1 << 30))

    # --- 接收机损伤（在 AWGN 之前施加，形成误差底） ---
    if impair:
        p = dict(DEFAULT_IMPAIR)
        if impair_kw:
            p.update(impair_kw)
        y = add_iq_imbalance(y, p["iq_gain_db"], p["iq_phase_deg"])
        y = add_nonlinearity(y, p["backoff_db"])
        y = add_cfo(y, fs, p["cfo_hz"])
        y = add_doppler(y, fs, p["fd_max_hz"], p["fd_rate_hz"],
                        seed=rng.integers(1 << 30))
        y = add_phase_noise(y, fs, p["linewidth_hz"], seed=rng.integers(1 << 30))

    # --- 热噪声 ---
    p_noise = 0.0
    if snr_db is not None:
        y, p_noise = add_awgn(y, snr_db, rng=rng)
    return y, p_noise