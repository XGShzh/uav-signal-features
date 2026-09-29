# -*- coding: utf-8 -*-
"""
run_part1.py
低空无人机信号特征 —— 第一部分（M1 信号建模）主入口
====================================================================
生成三类链路基带信号 -> 叠加多径 + AWGN 信道 -> 绘制：
    fig1_时域波形.png    三类信号时域包络/实部
    fig2_时频图.png      三类信号 STFT 时频图（核心图，对应方案 图2）
    fig3_功率谱.png      三类信号功率谱密度

用法：
    python run_part1.py            # 默认保存图片到 figures/
    python run_part1.py --show     # 额外弹出窗口显示
依赖：numpy / scipy / matplotlib
"""

import argparse
import os

# 将 matplotlib 缓存目录指到工程内可写目录（须在 import matplotlib 之前设置）
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_FIG_DIR = os.path.join(_MODULE_DIR, "figures")
os.environ["MPLCONFIGDIR"] = os.path.join(_MODULE_DIR, ".mplcache")

import numpy as np
from scipy.signal import spectrogram
import matplotlib
import matplotlib.pyplot as plt

from signal_models import generate_all
from channel import add_channel


# ----------------------------- 通用配置 -----------------------------
FIG_DIR = _FIG_DIR

# 中文字体（Windows）
for _f in ["Microsoft YaHei", "SimHei", "KaiTi"]:
    if any(_f.lower() in f.lower() for f in matplotlib.font_manager.get_font_names()):
        plt.rcParams["font.sans-serif"] = [_f]
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10


def savefig(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[已保存] {path}")
    return path


def plot_stft(ax, sig, fs, title, vmin=-80, vmax=10):
    """绘制 STFT 时频图（相对时间 ms、相对频率 MHz）"""
    f, tt, Sxx = spectrogram(sig, fs=fs, nperseg=512, noverlap=400,
                             nfft=1024, scaling="density",
                             return_onesided=False)
    # 复数信号返回的双边频点为 fftfreq 非单调顺序，需 fftshift 到 -fs/2..fs/2
    f = np.fft.fftshift(f)
    Sxx = np.fft.fftshift(Sxx, axes=0)
    P = 10 * np.log10(Sxx + 1e-30)
    ax.pcolormesh(tt * 1e3, f / 1e6, P, cmap="jet",
                  shading="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("f (MHz)")
    ax.set_ylim(f[0] / 1e6, f[-1] / 1e6)


def main():
    ap = argparse.ArgumentParser(description="M1 信号建模")
    ap.add_argument("--show", action="store_true", help="弹出显示窗口")
    ap.add_argument("--snr", type=float, default=10.0,
                    help="接收信噪比 dB（叠加 AWGN，-1 表示不加噪）")
    args = ap.parse_args()

    print(">>> 生成三类链路基带信号 ...")
    sigs = generate_all()

    print(">>> 叠加信道（多径 + AWGN, SNR=%s dB）..." % (args.snr if args.snr >= 0 else "off"))
    recv = {}
    for name in ["fhss", "ofdm", "gnss"]:
        fs = sigs[name]["params"]["fs"]
        snr = None if args.snr < 0 else args.snr
        y, p_noise = add_channel(sigs[name]["sig"], fs, snr_db=snr)
        recv[name] = dict(sig=y, t=sigs[name]["t"], fs=fs)

    # ------------------------------------------------------------
    # fig1 时域波形
    # ------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    fig.suptitle("M1 信号建模：三类链路时域波形（含多径 + AWGN）")
    rows = [("fhss", "遥控链路 FHSS"), ("ofdm", "图传链路 OFDM"), ("gnss", "导航信号 GPS C/A")]
    for r, (name, label) in enumerate(rows):
        sig = recv[name]["sig"]
        t = sigs[name]["t"] * 1e3
        axes[r, 0].plot(t[::4], sig.real[::4], lw=0.5)
        axes[r, 0].set_title(f"{label}：实部")
        axes[r, 0].set_xlabel("t (ms)")
        axes[r, 1].plot(t[::4], np.abs(sig[::4]), lw=0.5, color="C1")
        axes[r, 1].set_title(f"{label}：包络 |x|")
        axes[r, 1].set_xlabel("t (ms)")
    savefig(fig, "fig1_时域波形.png")

    # ------------------------------------------------------------
    # fig2 时频图（核心图）
    # ------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    fig.suptitle("M1 信号建模：三类链路 STFT 时频图（核心图）")
    vlims = [(-80, 5), (-80, 5), (-90, 5)]
    for r, (name, label) in enumerate(rows):
        plot_stft(axes[r], recv[name]["sig"], recv[name]["fs"],
                  label, vmin=vlims[r][0], vmax=vlims[r][1])
    savefig(fig, "fig2_时频图.png")

    # ------------------------------------------------------------
    # fig3 功率谱密度
    # ------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    fig.suptitle("M1 信号建模：三类链路功率谱密度")
    for r, (name, label) in enumerate(rows):
        sig = recv[name]["sig"]
        fs = recv[name]["fs"]
        Pxx, freqs = axes[r].psd(sig, NFFT=2048, Fs=fs, scale_by_freq=True)
        axes[r].set_title(f"{label}：PSD")
        axes[r].set_xlabel("频率 (MHz)")
    savefig(fig, "fig3_功率谱.png")

    print()
    print("=== M1 信号建模完成 ===")
    print(f"图片输出目录：{FIG_DIR}")
    print("三类信号关键参数：")
    for name in ["fhss", "ofdm", "gnss"]:
        print(f"  - {name}: {sigs[name]['params']}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()