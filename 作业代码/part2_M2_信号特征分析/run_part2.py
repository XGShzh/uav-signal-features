# -*- coding: utf-8 -*-
"""
run_part2.py
低空无人机信号特征 —— 第二部分（M2 信号特征分析）主入口
====================================================================
对三类链路信号做多域特征提取，构建「低空无人机信号特征体系（特征库）」，
输出特征库表格（CSV + 控制台）+ 特征图：
    fig1_循环谱特征.png   循环谱 SCF 沿循环频率的曲线（区分三类信号，核心特征图）
    fig2_OFDM星座与EVM.png  OFDM 解调星座图（干净/加噪）与 EVM 百分比
    fig3_时频跳频图案.png  FHSS 跳频图案提取（时频图 + 逐跳中心频率）
    features.csv          特征库数值表

复用 part1 的 signal_models（无信道干净信号 = 信号固有特征）。
用法：
    python run_part2.py
依赖：numpy / scipy / matplotlib
"""

import os
import sys

# matplotlib 缓存目录指到工程内写目录（须在 import matplotlib 之前设置）
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_FIG_DIR = os.path.join(_MODULE_DIR, "figures")
os.environ["MPLCONFIGDIR"] = os.path.join(_MODULE_DIR, ".mplcache")

# 允许导入同项目 part1 的信号建模与信道模块
_PART1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "part1_M1_信号建模")
sys.path.insert(0, os.path.abspath(_PART1))

from signal_models import generate_all
from channel import add_channel

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from features import scf_profile, ofdm_demod, compute_evm, extract_feature_row

for _f in ["Microsoft YaHei", "SimHei", "KaiTi"]:
    if any(_f.lower() in f.lower() for f in matplotlib.font_manager.get_font_names()):
        plt.rcParams["font.sans-serif"] = [_f]
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10


def savefig(fig, name):
    os.makedirs(_FIG_DIR, exist_ok=True)
    path = os.path.join(_FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[已保存] {path}")


def main():
    print(">>> 生成三类链路信号（无信道，提取固有特征）…")
    sigs = generate_all()
    names = ["fhss", "ofdm", "gnss"]
    labels = ["遥控链路 FHSS", "图传链路 OFDM", "导航信号 GPS C/A"]

    # ------------------------------------------------------------
    # 1) 循环谱特征（核心）—— scf_profile，绝对 dB 对比
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    styles = ["-", "--", ":"]
    for name, lab, st in zip(names[:3], labels, styles):
        fs = sigs[name]["params"]["fs"]
        al, mag = scf_profile(sigs[name]["sig"], fs)
        ax.plot(al / 1e6, 10 * np.log10(mag + 1e-30), st, lw=1.6, label=lab)
    ax.set_xlabel("循环频率 α (MHz)")
    ax.set_ylabel("谱相关强度 SCF (dB)")
    ax.set_title("图2-1 低空无人机三类信号：循环谱函数（SCF）特征")
    ax.grid(alpha=0.3)
    ax.legend()
    savefig(fig, "fig1_循环谱特征.png")
    plt.close(fig)

    # ------------------------------------------------------------
    # 2) OFDM 星座图 + EVM
    # ------------------------------------------------------------
    ref_list = sigs["ofdm"]["symbols"]          # 发射符号 (n_sym,)
    ref = np.stack(ref_list)                     # (n_sym, n_data)
    ofdm_p = sigs["ofdm"]["params"]
    fs_o = ofdm_p["fs"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    # EVM 用于演示 AWGN 对 QPSK 星座的影响：仅加噪声，不用长延迟多径
    #（M1 默认多径延迟达 3.5us，远超 OFDM 的 CP=0.64us，会破坏星座）
    lim = 1.5
    evm_by_snr = {}
    for axx, snr in zip(axes, [None, 15.0]):
        sig_rx = sigs["ofdm"]["sig"]
        if snr is not None:
            sig_rx, _ = add_channel(sig_rx, fs_o, snr_db=snr, use_multipath=False, seed=11)
        rx = ofdm_demod(sig_rx, fs_o, ofdm_p["fft_size"], ofdm_p["cp_len"],
                        ofdm_p["n_data_sub"], ofdm_p["fc_shift"])
        evm, g = compute_evm(ref, rx)
        evm_by_snr[snr] = float(evm)
        pts = rx.ravel() * g
        axx.scatter(pts.real, pts.imag, s=8, alpha=0.55)
        axx.set_xlim(-lim, lim); axx.set_ylim(-lim, lim)
        axx.set_aspect("equal")
        axx.axhline(0, color="0.8", lw=0.6)
        axx.axvline(0, color="0.8", lw=0.6)
        axx.set_title(f"OFDM 星座 — {'干净信号' if snr is None else f'SNR={snr}dB'}\nEVM ≈ {evm:.2f}%")
        axx.set_xlabel("I"); axx.set_ylabel("Q")
    fig.suptitle("图2-2 图传链路 OFDM：解调星座与 EVM（调制域特征）")
    savefig(fig, "fig2_OFDM星座与EVM.png")
    plt.close(fig)

    # ------------------------------------------------------------
    # 3) FHSS 跳频图案提取（时频特征）
    # ------------------------------------------------------------
    from scipy.signal import spectrogram
    fh_p = sigs["fhss"]["params"]
    fs_f = fh_p["fs"]
    n_hop = fh_p["n_hop"]
    t_hop_ms = fh_p["t_hop"] * 1e3
    sig_f = sigs["fhss"]["sig"]

    # 逐跳中心频率估计：每跳窗口内做 FFT，取主瓣能量重心（真实从信号提取）
    n_per_hop = int(round(fh_p["t_hop"] * fs_f))
    est = []
    for k in range(n_hop):
        seg = sig_f[k * n_per_hop:(k + 1) * n_per_hop]
        m = max(1, int(0.05 * len(seg)))          # 去掉跳边缓变窗影响
        seg = seg[m:-m]
        X = np.abs(np.fft.fftshift(np.fft.fft(seg))) ** 2
        f_ax = np.fft.fftshift(np.fft.fftfreq(len(seg), 1 / fs_f))
        mask = X > X.max() * 0.2
        est.append(float(np.sum(f_ax[mask] * X[mask]) / np.sum(X[mask])))
    est_mhz = np.array(est) / 1e6
    true_mhz = sigs["fhss"]["hop_freqs"] / 1e6
    ch_spacing = (fh_p["hop_band_hz"][1] - fh_p["hop_band_hz"][0]) / (fh_p["n_channels"] - 1) / 1e3
    print(f"[FHSS 跳频提取] 估计 vs 真值 最大误差 = {np.max(np.abs(est_mhz - true_mhz)) * 1e3:.1f} kHz"
          f"（信道间隔 {ch_spacing:.0f} kHz）")

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    f, tt, Sxx = spectrogram(sig_f, fs=fs_f,
                             nperseg=512, noverlap=400, nfft=1024,
                             scaling="density", return_onesided=False)
    f = np.fft.fftshift(f); Sxx = np.fft.fftshift(Sxx, axes=0)
    axes[0].pcolormesh(tt * 1e3, f / 1e6, 10 * np.log10(Sxx + 1e-30),
                       cmap="jet", shading="auto", vmin=-80, vmax=5)
    axes[0].set_ylabel("f (MHz)")
    axes[0].set_title("FHSS 跳频图案（STFT）")
    t_hop_center = (np.arange(n_hop) + 0.5) * t_hop_ms
    axes[1].plot(t_hop_center, true_mhz, "o-", ms=4, label="真值跳频")
    axes[1].plot(t_hop_center, est_mhz, "x--", ms=5, mew=1.5, label="信号提取")
    axes[1].set_xlabel("t (ms)")
    axes[1].set_ylabel("跳频中心频率 (MHz)")
    axes[1].set_title("逐跳中心频率：信号提取 vs 真值（时频特征）")
    axes[1].legend()
    fig.suptitle("图2-3 遥控链路 FHSS：跳频图案特征")
    savefig(fig, "fig3_时频跳频图案.png")
    plt.close(fig)

    # ------------------------------------------------------------
    # 4) 构建特征库（表格）+ 时频/突发统计
    # ------------------------------------------------------------
    rows = []
    for name, lab in zip(names, labels):
        fs = sigs[name]["params"]["fs"]
        extra = {}
        if name == "fhss":
            extra["跳频信道数"] = fh_p["n_channels"]
            extra["每跳驻留时间_us"] = fh_p["t_hop"] * 1e6
        if name == "ofdm":
            extra["子载波数"] = ofdm_p["n_data_sub"]
            extra["循环前缀"] = ofdm_p["cp_len"]
        if name == "gnss":
            extra["码片速率_Mchip_s"] = 1.023
            extra["码长_chip"] = 1023
        row = extract_feature_row(lab, sigs[name]["sig"], fs, extra=extra)
        rows.append(row)

    df = pd.DataFrame(rows)
    # 排序展示易读性
    df = df[df.columns.tolist()]
    print()
    print("=" * 78)
    print("低空无人机信号特征库（M2）")
    print("=" * 78)
    print(df.to_string(index=False))
    print("=" * 78)

    # 转置赏析版（一行一特征）
    df_t = df.set_index("信号类型").T
    print()
    print("特征库转置表：")
    print(df_t.round(4).to_string())

    csv_path = os.path.join(_FIG_DIR, "..", "特征库.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print()
    print(f"[已保存] 特征库 → {os.path.abspath(csv_path)}")
    print(f"[已保存] 图片目录 → {_FIG_DIR}")

    # 保存关键指标供 M6 闭环汇总（唯一数据源）
    import json
    metrics = {
        "module": "M2 信号特征分析（选题主体）",
        "n_features": int(df.shape[1] - 1),
        "n_signal_types": int(df.shape[0]),
        "hop_extract_max_error_khz": float(np.max(np.abs(est_mhz - true_mhz)) * 1e3),
        "hop_channel_spacing_khz": float(ch_spacing),
        # 循环谱可分性关键量（三类信号）
        "cyclic_strength_db": {str(r["信号类型"]): float(r["循环特征强度_dB"])
                               for _, r in df.iterrows()},
        "papr_db": {str(r["信号类型"]): float(r["峰均比PAPR_dB"])
                    for _, r in df.iterrows()},
        "evm_clean_pct": float(evm_by_snr[None]),
        "evm_snr15_pct": float(evm_by_snr[15.0]),
    }
    with open(os.path.join(_MODULE_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[已保存] 指标 → {os.path.join(_MODULE_DIR, 'metrics.json')}")


if __name__ == "__main__":
    main()