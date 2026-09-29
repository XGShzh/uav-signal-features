# -*- coding: utf-8 -*-
"""
jamming.py
低空无人机信号特征 —— 第四部分（M4 特征指导电子干扰）
====================================================================
用 M1 的真实 GFSK 调制与 M2 提取的跳频图案特征，构建干扰效果模型，
核心对比「特征指导精准干扰」vs「盲目干扰」。

====================================================================
文献依据（对应 uav-literature-list.html）
====================================================================
[#9] GNSS Jamming and Spoofing Threats in UAV Navigation: Countermeasure
     Status and Challenges, Y. Zeng, Z. Lu et al., IEEE COMST (2026).
     —— 系统梳理 GNSS 压制干扰与欺骗威胁：压制式抬高噪声致失锁，
        欺骗式用对准伪码牵引定位。本文 GNSS 分支据此建模（C/N0 退化 +
        捕获门限 + 欺骗牵引）。
[#10] Rigid-Covert GNSS Spoofing of UAV Swarms ... (Park, Yoo et al.).
     —— 隐蔽欺骗的攻击侧建模：误差受控、隐蔽，与压制式的“直接失锁”
        形成对比。本文欺骗分支据此输出“受控偏移”而非发散误差。
[#14] Adaptive RL-Based FHSS Strategies vs. 1st-Order Markov Jammer.
     —— 干扰机与跳频链路的对抗建模；本文的“反应式/跟随式干扰机”
        即该类对抗模型的基础形态。
[#1] RFUAV（arXiv:2503.09033, 2025）：遥控链路为跳频 + GFSK 体制，
     干扰需对准瞬时跳频信道才高效——这是“特征指导干扰”的物理依据。

方法说明（相对上一版的改进）
--------------------------
1) FHSS 分支由**解析 BER 公式**改为**蒙特卡洛仿真**：用 M1 的真实 GFSK
   调制 + 鉴频器解调，逐比特统计误码率，结果含真实调制特性与统计涨落，
   避免“曲线过于平滑理想”。
2) 引入**命中概率**与**干扰带宽占比**刻画“精准 vs 盲目”的功率集中度，
   增益来源清晰（功率集中于当前跳频信道）。
3) GNSS 分支由示意曲线改为 **C/N0 退化 + 捕获/跟踪门限** 的物理模型。

依赖：numpy / scipy（复用 part1 的 signal_models.GFSK）
"""

import os
import sys

import numpy as np

# 复用 M1 的 GFSK 调制（保证干扰评估与实际信号体制一致）
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_BASE, "part1_M1_信号建模"))
from signal_models import gfsk_modulate


# ======================================================================
# 1) GFSK 鉴频器解调（与 M1 调制配套）
# ======================================================================
def gfsk_demod(rx, sps, h=0.5):
    """GFSK 非相干双相关器解调（正交频率检测）。

    rx : 复基带接收信号（含噪）
    返回 ±1 比特序列（长度 = len(rx)//sps）与有效样本数。
    依据：GFSK 非相干接收标准结构（匹配正交 f1/f0 参考波形取模比较）。
    """
    rx = np.asarray(rx, dtype=np.complex128)
    ns = len(rx) // sps
    R = rx[:ns * sps].reshape(ns, sps)
    t = np.arange(sps) / sps
    f1 = np.exp(1j * np.pi * h * t)          # +h/2 频偏参考
    f0 = np.conj(f1)                          # -h/2 频偏参考
    c1 = np.abs(R @ np.conj(f1))
    c0 = np.abs(R @ np.conj(f0))
    return np.sign(c1 - c0), ns * sps


# ======================================================================
# 2) 蒙特卡洛 GFSK 误码率
# ======================================================================
def gfsk_ber_mc(nbits=4000, ebn0_db=10.0, sps=8, h=0.5, bt=1.0, seed=0):
    """蒙特卡洛估计 GFSK 非相干解调误码率（双相关器，BT=1.0）。

    信号 |x|=1（每符号能量 Es=sps），每符号 1 bit。
    噪声：每采样功率 σ² = 1/EbN0_lin（使 Es/N0 = EbN0）。
    BT=1.0：BT 过小（如 0.5）时高斯脉冲跨多符号，会引入严重 ISI，
    使无噪也产生 ~24% 误码；BT=1.0 在朴素接收机下无 ISI，曲线真实。
    """
    rng = np.random.default_rng(seed)
    ebn0 = 10 ** (ebn0_db / 10.0)
    bits = rng.integers(0, 2, size=nbits) * 2 - 1
    tx = gfsk_modulate(bits, sps=sps, h=h, bt=bt)
    n = (rng.standard_normal(tx.size) + 1j * rng.standard_normal(tx.size)) / np.sqrt(2.0)
    rx = tx + n * np.sqrt(1.0 / ebn0)
    bits_hat, n_used = gfsk_demod(rx, sps, h)
    m = min(len(bits_hat), bits.size)
    return float(np.mean(bits_hat[:m] != bits[:m]))


def _hit_mask(rng, n_hop, hit):
    """命中掩码：反应式干扰机以概率 hit 命中每一跳（未命中则该跳无干扰）。"""
    return rng.random(n_hop) < hit if hit < 1.0 else np.ones(n_hop, dtype=bool)


def fhss_ber_curve(jsr_db_list, snr_db=20.0, b_ch=100e3, b_tot=10e6,
                   hit=0.9, jam_type="follow", n_hop=120, bits_per_hop=40,
                   sps=8, bt=1.0, seed=0):
    """蒙特卡洛 FHSS 链路 BER–JSR 曲线（逐跳、含命中随机性）。

    jsr_db_list : 总干信比 JSR（dB）序列
    hit         : 反应式干扰机的单跳命中概率（特征指导：高命中；
                  盲目：等效为低功率占比——见 jam_type）
    jam_type    : 'follow' 跟随式（功率集中于当前跳频信道）
                  'wideband' 盲目宽带（功率按带宽占比漏入目标信道）
    返回 BER 数组（与 jsr_db_list 等长）。
    """
    snr = 10 ** (snr_db / 10.0)
    if jam_type == "follow":
        frac = hit                          # 特征聚焦：命中则全功率落入本信道
    else:
        frac = b_ch / b_tot                 # 盲目：按带宽占比漏入

    ber = []
    for i, jsr_db in enumerate(jsr_db_list):
        jsr = 10 ** (jsr_db / 10.0)
        rng = np.random.default_rng(seed + 977 * i)
        # 逐跳：本跳有效干噪比 → Eb/N0_eff → 该跳 GFSK 误码
        hitm = _hit_mask(rng, n_hop, hit) if jam_type == "follow" \
            else np.zeros(n_hop, dtype=bool)
        n_err = n_tot = 0
        nbit = bits_per_hop
        for k in range(n_hop):
            j_eff = jsr * frac * (1.0 if (jam_type == "wideband" or hitm[k]) else 0.0)
            ebn0_eff = 1.0 / (1.0 / snr + j_eff)
            # 每跳独立噪声实现 → 统计涨落真实
            b = gfsk_ber_mc(nbit, 10 * np.log10(ebn0_eff + 1e-30),
                            sps=sps, bt=bt, seed=int(rng.integers(1 << 30)))
            n_err += b * nbit
            n_tot += nbit
        ber.append(n_err / max(n_tot, 1))
    return np.array(ber)


def jsr_at_ber(jsr_db_list, ber_list, target=1e-2):
    """从 BER–JSR 曲线插值求“达到 target BER 所需的最小 JSR(dB)”。

    注意：干扰场景下 BER 随 JSR **上升**（干扰越强误码越高），
    因此寻找 BER 首次“超过”门限的位置，而非低于门限。
    """
    jsr = np.asarray(jsr_db_list, dtype=float)
    b = np.asarray(ber_list, dtype=float)
    idx = np.where(b >= target)[0]          # BER 首次达到/超过目标
    if idx.size == 0:
        return np.nan
    i = idx[0]
    if i == 0:
        return float(jsr[0])
    x0, x1 = jsr[i - 1], jsr[i]
    y0, y1 = np.log10(b[i - 1] + 1e-12), np.log10(b[i] + 1e-12)
    if y1 == y0:
        return float(x1)
    return float(x0 + (np.log10(target) - y0) * (x1 - x0) / (y1 - y0))


# ======================================================================
# 3) GNSS C/A：C/N0 退化 + 捕获门限 + 欺骗牵引（依据 [#9][#10]）
# ======================================================================
def gnss_cn0(jsr_db, cn0_nom=45.0):
    """干扰导致的载噪比退化：C/(N0+J0) = C/N0 − 10lg(1+JSR)。"""
    jsr = 10 ** (np.asarray(jsr_db, dtype=float) / 10.0)
    return cn0_nom - 10 * np.log10(1.0 + jsr)


def gnss_position_error(jsr_db, mode="spoof", cn0_nom=45.0,
                        cn0_track=28.0, cn0_acq=32.0,
                        off_m=30.0, base_m=2.0, loo_m=120.0,
                        spoof_margin_db=-3.0, slope=1.5):
    """GNSS 接收机定位误差(m) 随干信比 JSR(dB) 变化。

    mode='jamming' 压制式（依据 [#9]）：
        C/N0 随 JSR 退化；低于跟踪门限 cn0_track → 失锁，误差发散到 loo_m；
        低于捕获门限则完全不可用。
    mode='spoof' 欺骗式（依据 [#10]）：
        欺骗信号功率超过真实信号（JSR > spoof_margin_db）后逐渐被接收机
        “捕获”，定位误差被牵引到受控偏移 off_m（隐蔽、有界）。
    """
    jsr = np.asarray(jsr_db, dtype=float)
    if mode == "spoof":
        # 捕获概率随 JSR 平滑上升（功率优势越大越易捕获）
        p = 1.0 / (1.0 + np.exp(-(jsr - spoof_margin_db) / slope))
        return base_m * (1 - p) + off_m * p
    else:
        cn0 = gnss_cn0(jsr, cn0_nom)
        # 跟踪门限过渡 + 捕获门限：C/N0 越低误差越大
        p_track = 1.0 / (1.0 + np.exp(-(cn0 - cn0_track) / 1.0))
        # 门限以上：码跟踪精度随 C/N0 下降而轻微恶化
        tracking_sigma = base_m + 2.0 * np.maximum(cn0_nom - cn0, 0.0) / 10.0
        return tracking_sigma * p_track + loo_m * (1 - p_track)