# -*- coding: utf-8 -*-
"""
run_part3.py
低空无人机信号特征 —— 第三部分（M3 电子侦察）主入口
====================================================================
用 M2 特征做两类侦察任务：
    1) 信号检测（OFDM 弱信号）：能量检测 ED vs 循环平稳检测 CFD，输出 Pd~SNR
       —— ED 采用保守门限设计（计入噪声不确定度 ru dB），呈现 SNR 墙；
          CFD 利用循环前缀结构特征，不依赖噪声功率，低 SNR 显著占优。
    2) 链路识别：多域特征向量 + 随机森林，输出 识别准确率~SNR 与混淆矩阵
       —— 对比 多SNR训练 与 单SNR训练：后者在低 SNR 下因脆弱特征
          （PAPR/占用带宽/平坦度）漂出训练分布而塌缩到单一类别。

输出图：
    fig1_检测Pd_SNR.png      ED vs CFD 的检测概率曲线（含理论 SNR 墙标注）
    fig2_识别准确率_SNR.png   三类链路识别准确率 vs SNR
                             （手工特征+RF 双训练协议 vs 深度学习 CNN 谱图）
    fig3_识别混淆矩阵.png     低/中/高 SNR 三联混淆矩阵（多SNR训练）
    fig4_谱图样本.png         三类信号在不同 SNR 下的谱图（CNN 输入可视化）

文献依据：
    [#4] 低 SNR CNN 检测（arXiv:2406.18624）—— 谱图 + CNN 路线
    [#5] Al-Sa'd FGCS 2019 —— 手工特征基线 vs 深度学习方法
    [#1] RFUAV —— RF 指纹识别

复用 part1(signal_models/channel)、part2(features)。
用法：
    python run_part3.py
依赖：numpy / scipy / sklearn / matplotlib
"""

import os
import sys

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_FIG_DIR = os.path.join(_MODULE_DIR, "figures")
os.environ["MPLCONFIGDIR"] = os.path.join(_MODULE_DIR, ".mplcache")

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ["part1_M1_信号建模", "part2_M2_信号特征分析"]:
    sys.path.insert(0, os.path.join(_BASE, _p))
from signal_models import generate_all

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from detection import detection_curve
from identify import evaluate_snr_curve, confusion_matrices
from dl_identify import evaluate_cnn_curve
from channel import add_channel

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
    print(">>> 生成三类链路信号 …")
    sigs = generate_all()

    # ------------------------------------------------------------
    # 1) 信号检测：ED（保守门限，含噪声不确定度）vs CFD（循环前缀结构特征）
    #    —— 用循环特征明显的 OFDM 做弱信号检测，展示 CFD 低 SNR 优势
    # ------------------------------------------------------------
    snr_det = [-16, -14, -12, -10, -8, -6, -5, -4, -2]
    ofdm_p = sigs["ofdm"]["params"]
    res = detection_curve(sigs["ofdm"]["sig"], ofdm_p["fs"], snr_det,
                          pf=0.01, mc=400, cfd_lag=ofdm_p["fft_size"])
    nu = res["noise_uncertainty_db"]
    wall = res["snr_wall_db"]
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.plot(res["snr"], np.array(res["pd_ed"]) * 100, "o-", lw=1.8,
            label=f"能量检测 ED（保守门限：噪声不确定度 {nu:.0f} dB）")
    ax.plot(res["snr"], np.array(res["pd_cfd"]) * 100, "s--", lw=1.8,
            label="循环前缀结构检测 CFD（不依赖噪声功率）")
    ax.axvline(wall, color="0.4", ls=":", lw=1.4)
    ax.annotate(f"理论 SNR 墙\n10lg(η-1)≈{wall:.1f} dB",
                xy=(wall, 50), xytext=(wall + 0.6, 42), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8))
    ax.set_xlabel("信噪比 SNR (dB)")
    ax.set_ylabel("检测概率 Pd (%)")
    ax.set_title("图3-1 图传 OFDM 弱信号检测：能量 vs 循环平稳结构特征（Pf=1%）")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_ylim(0, 105)
    savefig(fig, "fig1_检测Pd_SNR.png")
    plt.close(fig)

    # ------------------------------------------------------------
    # 2) 链路识别：识别准确率 ~ SNR
    #    传统路线：手工特征 + 随机森林（多SNR / 单SNR 训练）
    #    前沿路线：STFT 谱图 + CNN（多SNR 训练），依据 [#4][#5]
    # ------------------------------------------------------------
    snr_id = [-25, -18, -12, -8, -4, 0, 6, 12, 20]
    print(">>> 链路识别（手工特征 + 随机森林，双训练协议 / 3 种子）…")
    res_id, labels = evaluate_snr_curve(sigs, snr_id, n_seeds=3,
                                        m_train_per_snr=14, m_test=50)

    print(">>> 链路识别（STFT 谱图 + CNN，多SNR 训练）…")
    res_cnn = evaluate_cnn_curve(sigs, snr_id, add_channel,
                                 m_train_per_snr=14, m_test=40,
                                 epochs=35, n_seeds=2)

    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    styles = {"多SNR训练(-25~20dB)": ("o-", "C0"),
              "单SNR训练(20dB)": ("s--", "C1")}
    for pname, snrs in res_id.items():
        fmt, color = styles[pname]
        means = [snrs[s]["mean"] * 100 for s in snr_id]
        stds = [snrs[s]["std"] * 100 for s in snr_id]
        ax.errorbar(snr_id, means, yerr=stds, fmt=fmt, capsize=4, lw=1.6,
                    color=color, label=f"手工特征+RF：{pname}")
    # CNN 谱图分支
    cnn_mean = [res_cnn[s]["mean"] * 100 for s in snr_id]
    cnn_std = [res_cnn[s]["std"] * 100 for s in snr_id]
    ax.errorbar(snr_id, cnn_mean, yerr=cnn_std, fmt="^-.", capsize=4, lw=1.8,
                color="C2", label="深度学习 CNN：谱图输入（多SNR训练）")

    ax.axhline(100 / 3, color="0.4", ls=":", lw=1.2)
    ax.annotate("随机猜测 33.3%", xy=(-24, 100 / 3 + 2), fontsize=9, color="0.3")
    ax.set_xlabel("信噪比 SNR (dB)")
    ax.set_ylabel("识别准确率 (%)")
    ax.set_title("图3-2 三类链路识别：手工特征 vs 深度学习谱图（含真实信道损伤）")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 108)
    ax.legend(loc="lower right", fontsize=9)
    savefig(fig, "fig2_识别准确率_SNR.png")
    plt.close(fig)

    # ------------------------------------------------------------
    # 3) 混淆矩阵三联图：低/中/高 SNR（多SNR训练）
    #    —— 用 constrained_layout，避免图级 colorbar 与 tight_layout 冲突造成遮挡
    # ------------------------------------------------------------
    snr_conf = [-18, -12, 0]
    cms, labels = confusion_matrices(sigs, snr_conf, m_test=100)
    disp = {"FHSS": "遥控FHSS", "OFDM": "图传OFDM", "C/A": "导航C/A"}
    dlabels = [disp[l] for l in labels]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6), layout="constrained")
    for ax, cm, snr in zip(axes, cms, snr_conf):
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(3), dlabels, rotation=30, ha="right")
        ax.set_yticks(range(3), dlabels)
        acc = np.trace(cm) / cm.sum() * 100
        ax.set_title(f"测试 SNR = {snr} dB（准确率 {acc:.0f}%）")
        th = cm.max() / 2
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=10,
                        color="white" if cm[i, j] > th else "black")
    axes[0].set_xlabel("预测类别")
    axes[0].set_ylabel("真实类别")
    fig.suptitle("图3-3 链路识别混淆矩阵（多域特征 + 随机森林，多SNR训练）")
    cbar = fig.colorbar(im, ax=axes, pad=0.02, label="样本数")
    savefig(fig, "fig3_识别混淆矩阵.png", tight=False)
    plt.close(fig)

    # ------------------------------------------------------------
    # 4) 谱图样本可视化（CNN 输入）：三类信号 × 低/高 SNR
    # ------------------------------------------------------------
    from dl_identify import stft_image
    snr_show = [-10, 15]
    names3 = ["fhss", "ofdm", "gnss"]
    labels3 = ["遥控FHSS", "图传OFDM", "导航C/A"]
    fig, axes = plt.subplots(3, len(snr_show), figsize=(7.2, 9.6),
                             layout="constrained")
    rng_show = np.random.default_rng(2024)
    for ri, (name, lab) in enumerate(zip(names3, labels3)):
        fs = sigs[name]["params"]["fs"]
        base = sigs[name]["sig"]
        base = base / np.sqrt(np.mean(np.abs(base) ** 2))
        for ci, snr in enumerate(snr_show):
            rx, _ = add_channel(base, fs, snr_db=snr, use_multipath=False,
                                seed=rng_show)
            img = stft_image(rx, fs, 24, 24)
            ax = axes[ri, ci]
            ax.imshow(img, cmap="viridis", aspect="auto",
                      extent=[0, 4, -12.5, 12.5])
            ax.set_title(f"{lab}  SNR={snr}dB", fontsize=10)
            if ri == 2:
                ax.set_xlabel("t (ms)")
            if ci == 0:
                ax.set_ylabel("f (MHz)")
    fig.suptitle("图3-4 CNN 输入谱图样本（三类链路 × 低/高 SNR，含真实信道损伤）")
    savefig(fig, "fig4_谱图样本.png", tight=False)
    plt.close(fig)

    print()
    print("=== M3 电子侦察完成 ===")
    print(f"图片输出目录：{_FIG_DIR}")
    for pname, snrs in res_id.items():
        print(f"{pname} 识别准确率：",
              {s: round(snrs[s]["mean"] * 100, 1) for s in snr_id})
    print("CNN 谱图识别准确率：",
          {s: round(res_cnn[s]["mean"] * 100, 1) for s in snr_id})

    # 保存关键指标供 M6 闭环汇总（唯一数据源，避免手工抄录不一致）
    import json
    snr_det_arr = np.asarray(res["snr"])
    pd_cfd = np.asarray(res["pd_cfd"])
    pd_ed = np.asarray(res["pd_ed"])
    # 检出 (Pd>=10%) 的最低 SNR，用于度量 CFD 相对 ED 的灵敏度增益
    def _lowest_detect(snr_arr, pd_arr):
        idx = np.where(pd_arr >= 0.10)[0]
        return float(snr_arr[idx[0]]) if idx.size else None
    snr_cfd10 = _lowest_detect(snr_det_arr, pd_cfd)
    snr_ed10 = _lowest_detect(snr_det_arr, pd_ed)
    metrics = {
        "module": "M3 特征支撑电子侦察",
        "snr_wall_db": float(res["snr_wall_db"]),
        "pd_cfd_at_-8db": float(pd_cfd[np.argmin(np.abs(snr_det_arr + 8))]),
        "pd_ed_at_-8db": float(pd_ed[np.argmin(np.abs(snr_det_arr + 8))]),
        "lowest_detect_snr_cfd_db": snr_cfd10,
        "lowest_detect_snr_ed_db": snr_ed10,
        "cfd_sensitivity_gain_db": (float(snr_ed10 - snr_cfd10)
                                    if (snr_cfd10 is not None and snr_ed10 is not None)
                                    else None),
        "acc_rf_multisnr": {int(s): round(res_id["多SNR训练(-25~20dB)"][s]["mean"] * 100, 1)
                            for s in snr_id},
        "acc_rf_singlesnr": {int(s): round(res_id["单SNR训练(20dB)"][s]["mean"] * 100, 1)
                             for s in snr_id},
        "acc_cnn": {int(s): round(res_cnn[s]["mean"] * 100, 1) for s in snr_id},
    }
    with open(os.path.join(_MODULE_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[已保存] 指标 → {os.path.join(_MODULE_DIR, 'metrics.json')}")


if __name__ == "__main__":
    main()
