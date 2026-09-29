# -*- coding: utf-8 -*-
"""
signal_models.py
低空无人机信号特征 —— 第一部分（M1 信号建模）
====================================================================
对低空无人机三类链路的基带信号进行等效建模：
    1) 遥控链路 : FHSS + GFSK（跳频扩频 + 高斯频移键控）
    2) 图传链路 : OFDM（正交频分复用）—— QPSK + 循环前缀
    3) 导航信号 : GNSS GPS L1 C/A 码 —— 1.023 Mchip/s Gold 码 BPSK 扩频

====================================================================
文献依据（对应 uav-literature-list.html）
====================================================================
[#1] RFUAV: A Benchmark Dataset for UAV Detection and Identification,
     R. Shi et al., arXiv:2503.09033 (2025).
     支撑：REAL 无人机 RF 信号定义与三类链路划分；实测遥控链路为跳频+GFSK
     调制、图传为宽带 OFDM（宽带突发）、导航为 GNSS 扩频。
[#2] DroneRFa：用于侦测低空无人机的大规模无人机射频信号数据集,
     电子与信息学报, doi:10.11999/JEIT230570.
     支撑：城市/户外场景下遥控与图传工作在 ISM 频段、带宽量级。
[#6] DroneRF dataset: A dataset of drones for RF-based detection,
     classification and identification, Data in Brief (2019).
     支撑：遥控/图传/导航三类信号的类别划分依据。
[#3] A Multi-Scenario UAV RF Dataset ..., arXiv:2603.00106 (2026).
     支撑：多场景采集，说明观测时长与采样率的统一设置。

关键建模选择（区别于“理想仿真”）
--------------------------------
1) 遥控链路采用 **FHSS+GFSK**（非纯载波）：依据 [#1][#6]，真实遥控链路
   每跳携带调制数据。纯 CW 载波会高估跳频图案可分性。
2) 图传 OFDM 采用 52 数据子载波 + CP=16/64（类 802.11a 结构），依据 [#1]
   实测图传宽带突发特性。
3) 三类链路统一采样率 fs=25 MHz、观测时长约 4 ms，依据 [#3] 的统一采集
   口径，避免采样率/时长不一致导致特征泄漏。
4) 真实信道损伤（CFO/多普勒/相噪/IQ 不平衡/非线性/时变多径）在 channel.py
   中叠加，依据 [#1][#2][#7] 实测信号的非理想性。

每个函数独立、可单独调用，返回 (波形, 时间轴, ...) 或 np.ndarray。
依赖：numpy
"""

import numpy as np


# ----------------------------------------------------------------------
# 1) 遥控链路：FHSS + GFSK（跳频扩频 + 高斯频移键控）
# ----------------------------------------------------------------------
def gaussian_pulse(bt=1.0, sps=8, span=2):
    """GFSK 高斯频率成形脉冲 g(t)（BT 乘积控制频谱集中度）。

    BT=1.0：脉冲时宽约 1.2 个符号，在标准非相干双相关器接收机下无 ISI，
    对应低成本 FHSS 遥控链路的常见取值（参见 [#1] RFUAV 实测遥控信号）。
    注：BT=0.5（蓝牙类）脉冲跨约 4 个符号，需 MLSE/维特比接收机，
    与本文的简单接收机不匹配，故此处取 BT=1.0。
    """
    n = np.arange(-span * sps, span * sps + 1) / sps
    alpha = np.sqrt(np.log(2.0) / 2.0) / bt
    h = np.exp(-(n ** 2) / (2.0 * alpha ** 2))
    return h / h.sum()


def gfsk_modulate(bits, sps=8, h=0.5, bt=1.0, phase0=0.0):
    """GFSK 调制：NRZ → 高斯成形 → 频率积分 → 复包络。

    bits : ±1 比特序列
    h    : 调制指数（0.5 为常见 MSK 型 GFSK）
    bt   : 高斯带宽-时间乘积

    注意：高斯脉冲 g 具有群时延 (Lg-1)/2 个采样，此处显式补偿，
    使输出第 n 个采样的瞬时频率对应第 n//sps 个比特——否则解调会
    产生系统性符号错位（无噪误码率异常升高）。
    """
    up = np.repeat(np.asarray(bits, dtype=float), sps)
    g = gaussian_pulse(bt, sps)
    Lg = len(g)
    freq_full = np.convolve(up, g, mode="full")
    d = (Lg - 1) // 2                      # 补偿群时延，对齐输入序列
    freq = freq_full[d:d + len(up)] * (h / 2.0)
    phase = phase0 + 2 * np.pi * np.cumsum(freq) / sps
    return np.exp(1j * phase)


def gen_fhss(
    fs=25e6,
    n_hop=40,
    t_hop=1e-4,          # 每跳驻留时间 100 us
    hop_band_hz=(-5e6, 5e6),
    n_channels=64,       # 跳频信道数（伪随机图案）
    baud=40e3,           # GFSK 符号率（每跳约 4 个符号）
    mod_index=0.5,       # GFSK 调制指数
    bt=1.0,              # GFSK 高斯 BT 乘积（见 gaussian_pulse 说明）
    dwell_edges=True,    # 是否在每跳起止加缓变窗（抑制频谱泄漏）
    seed=1,
):
    """生成跳频 GFSK 遥控链路信号（每跳：随机信道 + GFSK 调制数据）。

    文献依据
    --------
    [#1] RFUAV（arXiv:2503.09033, 2025）：实测无人机遥控链路为跳频体制，
         各跳携带 GFSK 调制数据，并非单纯载波——本函数据此建模。
    [#6] DroneRF dataset（Data in Brief 2019）：遥控链路信号含调制信息，
         仅用 CW 载波会高估跳频图案的可分性。

    返回
    ----
    sig : (N,) complex128
    t   : (N,) float64 时间轴（秒）
    hop_freqs : (n_hop,) 每跳中心频率（供 M2 跳频图案提取对照）
    params : dict（含 modulation='FHSS-GFSK'）
    """
    rng = np.random.default_rng(seed + 1000)
    channel_freqs = np.linspace(hop_band_hz[0], hop_band_hz[1], n_channels)
    hop_freqs = rng.choice(channel_freqs, size=n_hop, replace=True)

    n_per_hop = max(1, int(round(t_hop * fs)))
    total = n_hop * n_per_hop
    t = np.arange(total) / fs
    sps = max(2, int(round(fs / baud)))       # 每符号采样数

    sig = np.zeros(total, dtype=complex)
    phase_acc = 0.0
    for k, fh in enumerate(hop_freqs):
        idx = np.arange(k * n_per_hop, (k + 1) * n_per_hop)
        # 本跳 GFSK 数据（连续相位：跨跳保持相位）
        n_sym = int(np.ceil(n_per_hop / sps))
        bits = rng.integers(0, 2, size=n_sym) * 2 - 1
        baseband = gfsk_modulate(bits, sps=sps, h=mod_index, bt=bt,
                                 phase0=phase_acc)[:n_per_hop]
        phase_acc = np.angle(baseband[-1]) if baseband.size else phase_acc

        ts = idx / fs
        hop = baseband * np.exp(1j * 2 * np.pi * fh * ts)   # 搬移到本跳信道

        if dwell_edges:  # 加升余弦缓变窗，抑制邻道泄漏
            n_edge = max(1, int(0.05 * n_per_hop))
            rise = np.hanning(2 * n_edge + 1)
            if len(rise) > n_per_hop:
                n_edge = n_per_hop // 2
                rise = np.hanning(2 * n_edge + 1)
            hop[:n_edge] *= rise[:n_edge]
            hop[-n_edge:] *= rise[n_edge + 1:]
        sig[idx] = hop

    params = dict(fs=fs, n_hop=n_hop, t_hop=t_hop,
                  hop_band_hz=list(hop_band_hz), n_channels=n_channels,
                  baud=baud, mod_index=mod_index, bt=bt,
                  modulation="FHSS-GFSK")
    return sig, t, hop_freqs, params


# ----------------------------------------------------------------------
# 2) 图传链路：OFDM（QPSK + 循环前缀）
# ----------------------------------------------------------------------
def gen_ofdm(
    fs=25e6,
    fft_size=64,
    n_data_sub=52,
    cp_len=16,
    n_symbols=1250,    # 1250×80=100k 样本=4ms：与 FHSS(4ms)/GNSS(5ms) 等观测时长，
                       # 保证后续特征对比时噪声统计量（PAPR/谱平均次数）不因观测长度不同而泄漏类别信息
    fc_shift=1e6,       # 带内中心频偏；约束 |fc_shift|+BW/2<=fs/2，即 1+10.16<=12.5 MHz，无混叠且留足余量
    qam_order=4,        # 4 = QPSK
    seed=2,
):
    """生成一组 OFDM 基带符号（含循环前缀），并搬移到 fc_shift 频带。

    文献依据：[#1] RFUAV 实测图传链路为宽带 OFDM 突发；[#2] DroneRFa
    实测图传带宽达数十 MHz 量级。此处采用 52 数据子载波、CP=16/64 的
    类 802.11a 结构等效建模（图传常用 OFDM 体制的典型参数）。

    返回
    ----
    sig : (N,) complex128
    t   : (N,) float64
    symbols : (n_symbols,) 调制符号（用于后续 EVM/星座图）
    params  : dict
    """
    rng = np.random.default_rng(seed + 2000)
    guard_sub = fft_size - n_data_sub
    # 数据子载波放在两侧，中间留保护（直流与高频 guard）
    half = n_data_sub // 2

    ofdm_sym_len = fft_size + cp_len
    t_up = np.arange(n_symbols * ofdm_sym_len) / fs
    sig_slot = np.zeros(n_symbols * ofdm_sym_len, dtype=complex)
    symbols = []

    for s in range(n_symbols):
        data = rng.integers(0, qam_order, size=n_data_sub)  # QPSK: 0-3
        map_qpsk = {0: 1 + 1j, 1: 1 - 1j, 2: -1 + 1j, 3: -1 - 1j}
        syms = np.array([map_qpsk[d] for d in data]) / np.sqrt(2)
        symbols.append(syms)

        x = np.zeros(fft_size, dtype=complex)
        # 低频段填 half 个，高频段填剩余 (n_data_sub-half)
        x[1:1 + half] = syms[:half]
        x[fft_size - (n_data_sub - half):fft_size] = syms[half:]

        t_sym = np.fft.ifft(x, fft_size) * np.sqrt(fft_size)  # IFFT
        t_cp = t_sym[-cp_len:]                                  # 加循环前缀
        tx = np.concatenate([t_cp, t_sym])
        sig_slot[s * ofdm_sym_len:(s + 1) * ofdm_sym_len] = tx

    # 把基带 OFDM 频带搬移到 fc_shift，使时频图带中心清晰可辨
    fc = fc_shift
    sig = sig_slot * np.exp(1j * 2 * np.pi * fc * t_up)

    params = dict(fs=fs, fft_size=fft_size, n_data_sub=n_data_sub,
                  cp_len=cp_len, n_symbols=n_symbols,
                  fc_shift=fc_shift, qam_order=qam_order)
    return sig, t_up, symbols, params


# ----------------------------------------------------------------------
# 3) 导航信号：GPS L1 C/A 码（Gold 码 + BPSK 扩频）
# ----------------------------------------------------------------------
def ca_code(sv=1):
    """生成 GPS L1 C/A 码（1023 chip Gold 码），1-indexed tap 实现。

    返回 (1023,) int8，取值为 +-1。
    """
    # G1 反馈抽头：3, 10；G2 反馈抽头：2,3,6,8,9,10（1-indexed）
    g1_taps = [3, 10]
    g2_taps = [2, 3, 6, 8, 9, 10]
    sv_taps = {  # SV 号 -> G2 延迟抽头对（1-indexed）
        1: [2, 6], 2: [3, 7], 3: [4, 8], 4: [5, 9], 5: [1, 9],
        6: [2, 10], 7: [1, 8], 8: [2, 9], 9: [3, 10],
    }
    a, b = sv_taps.get(sv, [2, 6])

    g1 = np.ones(10, dtype=int)
    g2 = np.ones(10, dtype=int)
    ca = np.zeros(1023, dtype=int)
    for i in range(1023):
        # 输出 chip = G1[10] xor G2[a] xor G2[b]
        ca[i] = g1[-1] ^ g2[a - 1] ^ g2[b - 1]
        # 反馈
        fb1 = g1[g1_taps[0] - 1] ^ g1[g1_taps[1] - 1]
        fb2 = g2[g2_taps[0] - 1] ^ g2[g2_taps[1] - 1] ^ g2[g2_taps[2] - 1] \
            ^ g2[g2_taps[3] - 1] ^ g2[g2_taps[4] - 1] ^ g2[g2_taps[5] - 1]
        g1 = np.roll(g1, 1)
        g1[0] = fb1
        g2 = np.roll(g2, 1)
        g2[0] = fb2
    return (2 * ca - 1).astype(np.int8)


def gen_gnss_ca(
    fs=25e6,            # 与遥控/图传链路统一采样率（同一宽带接收机），并使特征噪声底
                        # 不随采样率变化产生类别泄漏；25MHz 带宽 ±12.5MHz 覆盖三条链路
    n_code_repeat=4,    # 4 个码周期 = 4ms = 100k 样本，与其他链路等观测时长
    fc_shift=0.0,        # 导航信号带中心频偏（相对载波）
    bit_rate=50.0,       # 导航电文 50 bps
    sv=1,
    seed=3,
):
    """生成 GPS L1 C/A 码 BPSK 扩频基带信号。

    文献依据：[#9] GNSS Jamming and Spoofing Threats in UAV Navigation
    (IEEE COMST 2026) 指出无人机导航依赖 GNSS L1 C/A 码，其码结构
    （1023 chip Gold 码、1.023 Mchip/s）是抗欺骗与欺骗检测的基础；
    [#1] RFUAV 亦将导航链路作为第三类信号特征。

    chip 率 1.023 Mchip/s，chip 长度 code[sv]，导航 bit 时长 20 ms = 20460 chip。
    返回
    ----
    sig : (N,) complex128
    t   : (N,) float64
    chips : (N,) int8 扩频序列（+-1），供后续相关检测使用
    params : dict
    """
    rng = np.random.default_rng(seed + 3000)
    cod = ca_code(sv).astype(np.int8)
    L = cod.size                       # 1023
    samples_per_chip = fs / (1.023e6)  # = L / 1e-3 * ... ; 20 个采样/chip
    n_per_code = int(round(L * samples_per_chip))     # = 20460

    # 把码重复 n_code_repeat 个周期
    n_chips_total = L * n_code_repeat
    chips_full = np.tile(cod, n_code_repeat)

    # 按 code 时长抽样（每 chip 若干个采样），生成连续波形
    n_samp = int(n_chips_total * samples_per_chip)
    idx_samp = np.arange(n_samp)
    chip_idx = (idx_samp / samples_per_chip).astype(int) % n_chips_total
    bpsk = chips_full[chip_idx].astype(np.float64)

    # 导航电文调制：每 20 ms 翻转一次（20460 chip）
    chips_per_bit = int(round(1.023e6 / bit_rate))              # 20460 chip per bit
    n_bits = n_chips_total // chips_per_bit + 1
    nav = rng.integers(0, 2, size=n_bits) * 2 - 1
    nav_idx = (chip_idx // chips_per_bit).astype(int)
    data = nav[np.minimum(nav_idx, len(nav) - 1)]
    sig_base = bpsk * data

    t = np.arange(n_samp) / fs
    # 搬移到 fc_shift（若为非零则在接收端可解析到带内调频）
    sig = sig_base.astype(complex) * np.exp(1j * 2 * np.pi * fc_shift * t)

    params = dict(fs=fs, sv=sv, n_code_repeat=n_code_repeat,
                  bit_rate=bit_rate, fc_shift=fc_shift, chips_per_bit=chips_per_bit)
    return sig, t, chips_full, params


# ----------------------------------------------------------------------
# 小工具：统一生成三类信号，用于其他模块复用
# ----------------------------------------------------------------------
def generate_all(seed=7):
    """生成三类链路信号的去相关波形。返回 dict，供 run 脚本与后续 M2+ 模块调用。"""
    fhss_sig, fhss_t, fhss_f, fhss_p = gen_fhss(seed=seed)
    ofdm_sig, ofdm_t, ofdm_sym, ofdm_p = gen_ofdm(seed=seed)
    gnss_sig, gnss_t, gnss_chips, gnss_p = gen_gnss_ca(seed=seed)
    return {
        "fhss": dict(sig=fhss_sig, t=fhss_t, hop_freqs=fhss_f, params=fhss_p),
        "ofdm": dict(sig=ofdm_sig, t=ofdm_t, symbols=ofdm_sym, params=ofdm_p),
        "gnss": dict(sig=gnss_sig, t=gnss_t, chips=gnss_chips, params=gnss_p),
    }