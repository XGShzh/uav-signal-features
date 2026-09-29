# -*- coding: utf-8 -*-
"""
detection.py
低空无人机信号特征 —— 第三部分（M3 电子侦察：信号检测）
====================================================================
两类信号检测器（对同一观测窗做 H0/H1 二元检验）：
    1) 能量检测（ED）：帧平均能量 vs 卡方门限。
       —— 设定门限时计入“噪声功率不确定度”ru dB（保守抬高噪声假设），
          会产生经典的 SNR 墙（ED 在低 SNR 失效）。
    2) 循环平稳结构检测（CFD）：对图传 OFDM，利用循环前缀导致的
       时延=有用符号长（L）相关峰作为检验统计量（归一化相关）。
       —— 不依赖噪声功率，对噪声不确定度鲁棒，低 SNR 优于 ED。

====================================================================
文献依据（对应 uav-literature-list.html）
====================================================================
[#4] Robust Low-Cost Drone Detection and Classification in Low SNR
     Environments, S. Glüge et al., arXiv:2406.18624 (2024).
     —— 低 SNR 环境下无人机检测是核心难点；本文的 ED/CFD 对比即针对
        “低 SNR 弱信号检测”，并与该文谱图 CNN 路线形成呼应。
[#1] RFUAV（arXiv:2503.09033, 2025）：检测需利用无人机信号的体制结构，
     OFDM 循环前缀即典型可利用结构。
[#7] Securing the Skies 综述（arXiv:2504.11967, 2025）：低空监视的首要
     环节是可靠检测，噪声不确定性是实际系统的主要制约。
能量检测 SNR 墙结论源自经典检测理论（Tandra & Sahai, IEEE JSP 2005），
本文用仿真复现该理论值以验证实现正确性。

门限都在“纯噪声”下按虚警率 Pf 标定（ED 解析、CFD 蒙特卡洛分位数）。
依赖：numpy / scipy.stats
"""

import numpy as np
from scipy.stats import chi2


# ----------------------------------------------------------------------
# 能量检测（ED）
# ----------------------------------------------------------------------
def ed_statistic(y):
    """ED 检验统计量：观测窗 y 的平均能量。"""
    return float(np.mean(np.abs(np.asarray(y, dtype=np.complex128)) ** 2))


def ed_threshold(p_noise, n, pf, noise_uncertainty_db=0.0):
    """ED 门限：计入噪声功率不确定度 ru dB（保守用 ru·p_noise 作为设计噪声）。

    P(Sum > λ)=Pf，Sum ~ (ru·p_noise/2)·χ²(2n)；返回平均能量门限。
    """
    eta = 10 ** (noise_uncertainty_db / 10.0)
    return eta * p_noise * chi2.ppf(1 - pf, 2 * n) / (2 * n)


# ----------------------------------------------------------------------
# 循环平稳结构检测（CFD，基于 OFDM 循环前缀相关峰）
# ----------------------------------------------------------------------
def cfd_cp_statistic(y, lag=64):
    """归一化 CP 相关：测试时延=有用符号长 L 处是否出现相关峰。

        γ = |Σ y[n]·y*[n-L]| / sqrt(Σ|y[n]|² · Σ|y[n-L]|²)
    只有信号存在 L 处相关峰（OFDM 的 CP 结构）时 γ 显著增大；噪声下→小值。
    """
    y = np.asarray(y, dtype=np.complex128)
    n1 = y[:-lag]
    n2 = y[lag:]
    num = np.abs(np.vdot(n1, n2))
    den = np.sqrt(np.sum(np.abs(n1) ** 2) * np.sum(np.abs(n2) ** 2))
    return float(num / (den + 1e-30))


# ----------------------------------------------------------------------
# 统一蒙特卡洛：估计 ED 与 CFD 的 Pd~SNR 曲线
# ----------------------------------------------------------------------
def snr_wall_db(noise_uncertainty_db):
    """保守门限设计下 ED 的渐近 SNR 墙：1+SNR > eta 才可能检测 → SNR* = eta-1。"""
    eta = 10 ** (noise_uncertainty_db / 10.0)
    return float(10 * np.log10(eta - 1.0))


def detection_curve(sig_clean, fs, snr_db_list, pf=0.01, win=4096,
                    cfd_lag=64, noise_uncertainty_db=1.0, mc=400, n_cal=800,
                    rng=None):
    """对干净 OFDM 参考信号在不同 SNR 下估计 ED/CFD 检测概率 Pd。

    每次观测 = sig[:win] + 一次独立噪声实现；重复 mc 次取平均。
    CFD 门限用 n_cal 次纯噪声实现标定（99% 分位需要足够样本才稳定）。
    """
    if rng is None:
        rng = np.random.default_rng(2024)
    sig = np.asarray(sig_clean, dtype=np.complex128)
    p_sig = np.mean(np.abs(sig[:win]) ** 2)

    pd_ed, pd_cfd = [], []
    for snr in snr_db_list:
        p_noise = p_sig / (10 ** (snr / 10.0))
        lam_ed = ed_threshold(p_noise, win, pf, noise_uncertainty_db)
        thr_cfd = np.quantile(
            [cfd_cp_statistic(_noise(win, p_noise, rng), cfd_lag)
             for _ in range(n_cal)], 1 - pf)

        d_ed = d_cfd = 0
        for _ in range(mc):
            y = sig[:win] + _noise(win, p_noise, rng)
            d_ed += float(ed_statistic(y) > lam_ed)
            d_cfd += float(cfd_cp_statistic(y, cfd_lag) > thr_cfd)
        pd_ed.append(d_ed / mc)
        pd_cfd.append(d_cfd / mc)
    return dict(snr=snr_db_list, pd_ed=pd_ed, pd_cfd=pd_cfd,
                noise_uncertainty_db=noise_uncertainty_db,
                snr_wall_db=snr_wall_db(noise_uncertainty_db))


def _noise(n, p_noise, rng):
    """生成长度 n、总功率 p_noise 的复高斯白噪声。"""
    z = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2.0)
    return z * np.sqrt(p_noise / np.mean(np.abs(z) ** 2))