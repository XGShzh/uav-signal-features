# -*- coding: utf-8 -*-
"""
run_part4.py
低空无人机信号特征 —— 第四部分（M4 特征指导电子干扰）主入口
====================================================================
用 M1 的真实 GFSK 调制与 M2 的跳频图案特征，对比
「特征指导精准干扰」vs「盲目干扰」的效能：

    fig1_FHSS跟随vs宽带_BER_JSR.png
        遥控 FHSS：跟随式 vs 盲目宽带（蒙特卡洛 GFSK 实测 BER–JSR）+ 增益标注
    fig2_GNSS欺骗vs压制_定位误差_JSR.png
        导航 GNSS：欺骗（受控牵引）vs 压制（C/N0 退化致失锁）

文献依据
--------
[#9]  GNSS 干扰/欺骗综述（IEEE COMST 2026）—— 压制式 C/N0 退化与失锁
[#10] 隐蔽欺骗攻击建模 —— 误差受控、隐蔽
[#14] RL 自适应跳频对抗 —— 干扰机与跳频链路对抗模型
[#1]  RFUAV —— 遥控链路为跳频+GFSK，干扰需对准瞬时信道

用法：
    python run_part4.py
依赖：numpy / scipy / matplotlib（复用 part1 的 GFSK 调制）
"""

import os
import sys

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_FIG_DIR = os.path.join(_MODULE_DIR, "figures")
os.environ["MPLCONFIGDIR"] = os.path.join(_MODULE_DIR, ".mplcache")

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from jamming import (fhss_ber_curve, jsr_at_ber,
                     gnss_position_error, gnss_cn0)

for _f in ["Microsoft YaHei", "SimHei", "KaiTi"]:
    if any(_f.lower() in f.lower() for f in matplotlib.font_manager.get_font_names()):
        plt.rcParams["font.sans-serif"] = [_f]
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10


def savefig(fig, name, tight=True):
    os.makedirs(_FIG_DIR, exist_ok=True)
    path = os.path.join(_FIG_DIR, name)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[已保存] {path}")


def main():
    # ---------- FHSS 链路参数（与 M1/M2 一致）----------
    b_ch = 100e3      # 单个跳频信道 / GFSK 占用带宽
    b_tot = 10e6      # 跳频总带宽（-5~5 MHz）
    snr = 20.0        # 无干扰时链路 SNR (dB)
    # JSR 范围需覆盖深度负值：干扰只需与“热噪声”可比即可生效（非与信号比），
    # 故跟随式干扰在 -25 dB 量级就能把链路压垮。
    JSR = np.arange(-35, 11, 1.0)

    print(">>> 蒙特卡洛仿真 FHSS-GFSK 干扰误码率（跟随式 vs 盲目宽带）…")
    ber_follow = fhss_ber_curve(JSR, snr, b_ch, b_tot,
                                hit=0.9, jam_type="follow",
                                n_hop=200, bits_per_hop=50, seed=1)
    ber_wide = fhss_ber_curve(JSR, snr, b_ch, b_tot,
                              hit=0.9, jam_type="wideband",
                              n_hop=200, bits_per_hop=50, seed=2)

    JSR_f = jsr_at_ber(JSR, ber_follow, 1e-2)
    JSR_w = jsr_at_ber(JSR, ber_wide, 1e-2)
    gain = JSR_w - JSR_f
    print(f"  BER=1e-2：跟随式需 JSR={JSR_f:.1f} dB，盲目宽带需 {JSR_w:.1f} dB，"
          f"精准干扰增益 ≈ {gain:.1f} dB")

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    ax.semilogy(JSR, np.maximum(ber_follow, 1e-6), "o-", ms=3.5, lw=1.6,
                label="特征指导 跟随式干扰（对准当前跳频信道，命中率 90%）")
    ax.semilogy(JSR, np.maximum(ber_wide, 1e-6), "s--", lw=1.6,
                label="盲目 宽带干扰（功率摊满整个跳频带）")
    ax.axhline(1e-2, color="0.4", ls=":", lw=1.2)
    ax.annotate("目标 BER = 1e-2", xy=(-33, 1.3e-2), fontsize=9, color="0.3")
    if np.isfinite(JSR_f) and np.isfinite(JSR_w):
        ax.axvline(JSR_f, color="C0", ls=":", lw=1.0)
        ax.axvline(JSR_w, color="C1", ls=":", lw=1.0)
        ax.annotate("", xy=(JSR_w, 2e-3), xytext=(JSR_f, 2e-3),
                    arrowprops=dict(arrowstyle="<->", color="C2", lw=1.5))
        ax.text((JSR_f + JSR_w) / 2, 4e-3, f"精准干扰增益 ≈ {gain:.0f} dB\n"
                f"（跟随 {JSR_f:.0f} dB vs 宽带 {JSR_w:.0f} dB）",
                ha="center", fontsize=9.5, color="C2")
    ax.set_xlabel("干信比 JSR (dB)")
    ax.set_ylabel("误码率 BER（蒙特卡洛实测）")
    ax.set_title("图4-1 遥控链路 FHSS-GFSK：特征指导跟随式 vs 盲目宽带干扰")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(1e-6, 0.7)
    savefig(fig, "fig1_FHSS跟随vs宽带_BER_JSR.png")
    plt.close(fig)

    # ---------- GNSS：欺骗 vs 压制 ----------
    JSR2 = np.arange(-15, 41, 0.5)
    err_spoof = gnss_position_error(JSR2, mode="spoof", off_m=30.0)
    err_jam = gnss_position_error(JSR2, mode="jamming", cn0_nom=45.0,
                                  cn0_track=28.0, loo_m=120.0)

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    ax.plot(JSR2, err_spoof, "o-", ms=3, lw=1.7,
            label="特征指导 欺骗干扰（对准伪码，受控牵引）")
    ax.plot(JSR2, err_jam, "s--", lw=1.7,
            label="盲目 压制式干扰（C/N0 退化致失锁）")

    # 次轴：压制干扰下的 C/N0 退化（物理量，增强说服力）
    ax2 = ax.twinx()
    cn0 = gnss_cn0(JSR2, 45.0)
    ax2.plot(JSR2, cn0, color="0.5", lw=1.2, ls=":")
    ax2.axhline(28.0, color="0.5", lw=1.0, ls=":")
    ax2.text(41, 29.0, "C/A 码跟踪门限 28 dB-Hz", fontsize=8.5, color="0.35",
             ha="right", va="bottom")
    ax2.set_ylabel("压制干扰下 C/N0 (dB-Hz)", color="0.4")
    ax2.tick_params(axis="y", colors="0.4")
    ax2.set_ylim(10, 50)

    ax.set_xlabel("干信比 JSR (dB)")
    ax.set_ylabel("导航定位误差 (m)")
    ax.set_title("图4-2 导航 GNSS C/A：欺骗 vs 压制干扰（依据 [#9][#10]）")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(0, 135)
    ax.annotate("欺骗捕获后：误差被受控牵引到 ≈30 m\n（隐蔽、有界、可诱导）",
                xy=(24, 30), xytext=(-13, 78), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="C0", lw=0.8))
    ax.annotate("压制：C/N0 跌破跟踪门限即失锁\n（误差发散，导航不可用）",
                xy=(24, 115), xytext=(14, 96), fontsize=9, color="C1",
                arrowprops=dict(arrowstyle="->", color="C1", lw=0.8))
    savefig(fig, "fig2_GNSS欺骗vs压制_定位误差_JSR.png")
    plt.close(fig)

    print()
    print("=== M4 电子干扰完成 ===")
    print(f"图片输出目录：{_FIG_DIR}")

    # 保存关键指标供 M6 闭环汇总（唯一数据源，避免手工抄录不一致）
    import json
    metrics = {
        "module": "M4 特征指导电子干扰",
        "jsr_follow_db": float(JSR_f),
        "jsr_wideband_db": float(JSR_w),
        "precision_jamming_gain_db": float(gain),
        "target_ber": 1e-2,
        "fhss_channel_gain_db": float(10 * np.log10(b_tot / b_ch)),
    }
    with open(os.path.join(_MODULE_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[已保存] 指标 → {os.path.join(_MODULE_DIR, 'metrics.json')}")


if __name__ == "__main__":
    main()