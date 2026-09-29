# -*- coding: utf-8 -*-
"""
features.py
低空无人机信号特征 —— 第二部分（M2 信号特征分析）
====================================================================
多域特征提取函数，支撑「低空无人机信号特征体系（特征库）」。

覆盖维度：
    时域/频域特征   : 平均功率、峰均比 PAPR、包络抖动、谱平坦度、中心频率、带宽
    时频特征       : STFT（FHSS 跳频图案 / OFDM 突发-帧结构）
    循环平稳特征   : 循环谱函数 SCF → 峰值循环频率及强度
    调制域特征     : OFDM 解调 + 星座图 + EVM

====================================================================
文献依据（对应 uav-literature-list.html）
====================================================================
[#1] RFUAV（arXiv:2503.09033, 2025）：定义“RF 无人机指纹”特征体系，
     强调多域特征（时频图、循环特征）是识别与分选的核心依据。
[#5] Al-Sa'd et al., FGCS (2019)：无人机 RF 信号的时/频域统计特征
     用于检测与分类，是本文特征库的基线方法来源。
[#7] Securing the Skies 综述（arXiv:2504.11967, 2025）：低空监视依赖
     稳健的频谱特征；循环平稳特征对低 SNR 更鲁棒。
[#2] DroneRFa（电子与信息学报）：频域带宽/中心频率特征的实测参照。
循环平稳特征（谱相关函数 SCF）理论基础源自 Gardner 的循环平稳信号处理，
是区分 FHSS / OFDM(含 CP) / C/A 码扩频的经典稳健特征。

依赖：numpy / scipy
"""

import numpy as np
from scipy.signal import welch


# ----------------------------------------------------------------------
# 1) 时域特征
# ----------------------------------------------------------------------
def time_features(sig):
    """返回时域特征字典（标量）。sig: 复数基带信号。"""
    x = np.asarray(sig, dtype=np.complex128)
    env = np.abs(x)
    p_mean = np.mean(env ** 2)
    p_peak = np.max(env ** 2)
    return dict(
        平均功率_dB=10 * np.log10(p_mean + 1e-30),
        峰均比PAPR_dB=10 * np.log10(p_peak / (p_mean + 1e-30) + 1e-30),
        包络抖动=float(np.std(env) / (np.mean(env) + 1e-30)),   # 归一化包络标准差
        峰值幅度=float(np.max(env)),
    )


# ----------------------------------------------------------------------
# 2) 频域特征
# ----------------------------------------------------------------------
def freq_features(sig, fs):
    """基于 Welch 功率谱返回频域特征字典。"""
    x = np.asarray(sig, dtype=np.complex128)
    f, P = welch(x, fs=fs, nperseg=2048, noverlap=1024, return_onesided=False)
    P = P + 1e-30
    # 把周期序（0..fs）重排为 -fs/2..fs/2，并配对功率谱
    order = np.fft.fftshift(np.arange(len(f)))
    fm = np.linspace(-fs / 2, fs / 2, len(order), endpoint=False)
    Ps = P[order]

    Ptot = float(np.sum(Ps))
    # 中心频率（一阶谱矩）
    fc = float(np.sum(fm * Ps) / Ptot)
    # 99% 占用带宽：包含 99% 能量的最小频带宽度
    cum = np.cumsum(Ps) / Ptot
    lo = fm[np.searchsorted(cum, 0.005)]
    hi = fm[np.searchsorted(cum, 0.995)]
    bw99 = float(hi - lo)
    # -3dB 带宽：主瓣能量降到峰值一半处的带宽
    pmax = np.max(Ps)
    half_bins = np.where(Ps >= pmax / 2)[0]
    b3 = (half_bins[-1] - half_bins[0] + 1) * (fm[1] - fm[0]) if half_bins.size else 0.0
    # 谱平坦度（0~1，1 表示白噪声）
    Psn = Ps / Ptot
    flat = float(np.exp(np.mean(np.log(Psn + 1e-30))) / (np.mean(Psn) + 1e-30))
    return dict(
        中心频率_MHz=fc / 1e6,
        占用带宽99_pct_MHz=bw99 / 1e6,
        带宽3dB_MHz=b3 / 1e6,
        谱平坦度=flat,
    )


# ----------------------------------------------------------------------
# 3) 循环平稳特征：谱相关函数 SCF 沿循环频率 alpha 的强度曲线
# ----------------------------------------------------------------------
def scf_profile(sig, fs, alpha_max=2.5e6, n_alpha=60, nperseg=1024):
    """用“平均循环周期图（频域累加）”估计 SCF 幅度随循环频率 alpha 的变化。

    定义（共轭循环谱）：
        S^alpha_x(f) = < X_u(f+alpha/2) · X_u^*(f-alpha/2) >_u
    返回 (alpha, mag)：alpha 循环频率轴，mag 为在频率方向的平均幅度。
    峰值循环频率是区分 FHSS/OFDM/C/A 的关键特征。

    可选参数供下游快速调用（减小 nperseg / n_alpha 换取速度）。
    """
    x = np.asarray(sig, dtype=np.complex128)
    N = x.shape[0]
    hop = nperseg // 2
    nf = 1 + (N - nperseg) // hop
    if nf < 4:
        raise ValueError("信号过短，无法估计循环谱")
    win = np.hanning(nperseg)
    X = np.empty((nf, nperseg), dtype=np.complex128)
    for m in range(nf):
        X[m] = np.fft.fft(x[m * hop:m * hop + nperseg] * win, nperseg)
    df = fs / nperseg

    alphas = np.linspace(0.05e6, alpha_max, n_alpha)
    mag = np.zeros(n_alpha)
    for i, al in enumerate(alphas):
        off = int(round(al / 2.0 / df))
        if off <= 0 or off >= nperseg // 2:
            mag[i] = 0.0
            continue
        # X(f+alpha/2) 与 X(f-alpha/2) 对应 bin 对
        hi = X[:, off:nperseg - off]        # 频率 f+alpha/2
        lo = X[:, :nperseg - 2 * off]       # 频率 f-alpha/2
        cross = hi * np.conj(lo)            # M × L
        scf = np.mean(cross, axis=0)        # 对帧平均 -> S_f(alpha)
        mag[i] = float(np.mean(np.abs(scf)))  # 沿频率平均
    # 返回绝对谱相关幅度（不归一化，保证三类信号在相同功率下可比）
    return alphas, mag


def cyclic_feature(sig, fs, **kw):
    """返回循环平稳特征：循环谱强度（dB，绝对可比）与高频保持度。

    由于三类信号功率归一一致，循环谱强度绝对量级可直接对比：
      FHSS（逐跳 CW）循环谱快速衰减 → 强度最低；
      OFDM（循环前缀）宽频带维持强循环 → 强度最高；
      C/A（恒模随机码）居中。
    **kw 透传给 scf_profile（可调 nperseg 等换取速度）。
    """
    al, mag = scf_profile(sig, fs, **kw)
    hi_mask = al > 0.5e6                    # 避开 alpha→0 的谱基底
    strength = 10 * np.log10(np.mean(mag[hi_mask]) + 1e-30)
    low = np.mean(mag[~hi_mask])
    keep = float(np.mean(mag[hi_mask]) / (low + 1e-30))
    return dict(
        循环特征强度_dB=strength,
        循环谱高频保持度=keep,
    )


# ----------------------------------------------------------------------
# 4) 调制域特征：OFDM 解调 + 星座 + EVM
# ----------------------------------------------------------------------
def ofdm_demod(sig_rx, fs, fft_size, cp_len, n_data_sub, fc_shift,
               cfo_comp=True, cpe_comp=True):
    """OFDM 接收解调（含真实接收机同步），返回解调符号矩阵 n_sym×n_data。

    接收机同步（依据 [#1] 实测接收机必需环节）：
      1) 搬移回基带：按已知 fc_shift 去载波；
      2) **CP 频偏估计与校正**：用循环前缀与其对应符号尾部的相关相位
         估计残余载波频偏 Δf = φ·fs/(2π·N_fft)，逐点去旋转
         （否则在 4ms 观测内微弱频偏会累积数圈相位，星座被抹成圆环）；
      3) **盲相位跟踪（CPE）**：QPSK 用 4 次方去调制估计每符号公共相位
         φ = ∠(mean(x⁴))/4 并校正（本振相位噪声的慢漂移由接收机跟踪掉）。
    这两步是真实 OFDM 接收机的标准同步功能，不做则 EVM 会被非理想损伤
    完全支配，无法反映真实的调制域质量。
    """
    sym_len = fft_size + cp_len
    half = n_data_sub // 2
    t = np.arange(len(sig_rx)) / fs
    y = np.asarray(sig_rx, dtype=np.complex128) * np.exp(-1j * 2 * np.pi * fc_shift * t)
    n_sym = len(y) // sym_len

    # --- 2) CP 频偏估计与校正 ---
    if cfo_comp and n_sym > 1:
        ang = []
        for s in range(n_sym):
            blk = y[s * sym_len:(s + 1) * sym_len]
            cp = blk[:cp_len]                  # 循环前缀
            tail = blk[sym_len - cp_len:]      # 对应符号尾部（两者本应相同）
            ang.append(np.angle(np.vdot(tail, cp)))
        cfo = np.mean(ang) * fs / (2 * np.pi * fft_size)
        y = y * np.exp(-1j * 2 * np.pi * cfo * t)

    hi_idx = np.arange(1, 1 + half)
    lo_idx = np.arange(fft_size - (n_data_sub - half), fft_size)
    raw = []
    for s in range(n_sym):
        blk = y[s * sym_len:(s + 1) * sym_len]
        X = np.fft.fft(blk[cp_len:], fft_size)
        raw.append(np.concatenate([X[hi_idx], X[lo_idx]]))

    if not cpe_comp:
        return np.stack(raw)

    # --- 3) 盲公共相位跟踪（QPSK，含 π/2 模糊消除） ---
    # 4 次方去调制给出的相位存在 π/2 整数倍模糊；若逐符号独立取主值，
    # 浮点噪声会让分支随机跳变（星座被毁）。故按符号连续性选择分支
    # （相位跟踪/解缠），这也是真实接收机跟踪本振相位漂移的做法。
    phis = []
    prev = None
    for sym in raw:
        acc = np.mean(sym ** 4)
        if np.abs(acc) < 1e-12:            # 低 SNR 下估计不可靠，沿用上一相位
            phis.append(prev if prev is not None else 0.0)
            continue
        phi = np.angle(acc) / 4.0
        if prev is not None:
            phi += np.round((prev - phi) / (np.pi / 2)) * (np.pi / 2)
        phis.append(phi)
        prev = phi
    out = [sym * np.exp(-1j * phi) for sym, phi in zip(raw, phis)]
    return np.stack(out)


def compute_evm(ref, rx):
    """EVM 计算（含复数增益拟合，消除尺度/相位模糊），单位 %。

    增益用全部符号整体最小二乘估计，抗噪更稳。
    """
    ref = np.asarray(ref, dtype=np.complex128)
    rx = np.asarray(rx, dtype=np.complex128)
    g = np.sum(ref * np.conj(rx)) / (np.sum(np.abs(rx) ** 2) + 1e-30)
    err = np.abs(rx * g - ref)
    evm = np.sqrt(np.mean(err ** 2) / (np.mean(np.abs(ref) ** 2) + 1e-30)) * 100.0
    return evm, g


# ----------------------------------------------------------------------
# 5) 汇总：构建单条特征记录
# ----------------------------------------------------------------------
def extract_feature_row(name, sig, fs, extra=None):
    """对单个信号聚合全部标量特征，返回 dict（特征库一行）。"""
    row = {"信号类型": name}
    row.update(time_features(sig))
    row.update(freq_features(sig, fs))
    cyc = cyclic_feature(sig, fs)
    row.update(cyc)
    if extra:
        row.update(extra)
    return row