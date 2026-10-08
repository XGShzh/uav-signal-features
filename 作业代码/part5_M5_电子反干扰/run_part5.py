# -*- coding: utf-8 -*-
"""
run_part5.py
低空无人机信号特征 —— 第五部分（M5 电子反干扰）主入口
====================================================================
用 M2 特征为三类链路设计并评估电子防护：

    fig1_自适应跳频_抗干扰.png
        遥控 FHSS：固定跳频表 / 均匀伪随机 / 混沌跳频 / RL 自适应跳频
        （左）学习曲线（链路可用度 vs 跳数）
        （右）稳态可用度 vs 被干扰信道占比
    fig2_欺骗检测_Pd_JSR.png
        导航 GNSS：欺骗检测三条路线（功率异常 / 相关峰 / 惯导一致性）
    fig3_防护前后导航误差.png
        导航 GNSS：无防护 / 检测剔除 / 惯导辅助 三种策略的误差时域演化

文献依据
--------
[#11] SemperFi（arXiv:2105.01860）—— 惯导辅助抗欺骗
[#12] GPS-Spoofing Attack Detection（arXiv:2301.12766）—— 一致性/功率检测
[#13] Chaotic-FH（IEEE Access 2024）—— 混沌跳频
[#14] Adaptive RL-Based FHSS（Preprints 202605.1672）—— Q-learning 自适应跳频
[#9]  GNSS 干扰/欺骗综述（IEEE COMST 2026）—— 反欺骗技术路线

用法：
    python run_part5.py
依赖：numpy / matplotlib
"""

import os

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_FIG_DIR = os.path.join(_MODULE_DIR, "figures")
os.environ["MPLCONFIGDIR"] = os.path.join(_MODULE_DIR, ".mplcache")

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from ecm import (processing_gain_fhss, processing_gain_dsss,
                 hop_pattern_uniform, hop_pattern_fixed_table,
                 hop_pattern_chaotic, q_learning_afh,
                 static_pattern_availability, steady_availability,
                 spoof_detect_pd, spoof_detect_roc_auc,
                 navigation_error_vs_time)

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


def _pd50_jsr(method):
    """从检测曲线实算 Pd=50% 对应的 JSR（不手工抄录，保证与图一致）。

    在越过 0.5 的两点间线性插值，消除网格步长带来的离散误差。
    """
    j = np.arange(-40.0, 41.0, 0.05)
    p = spoof_detect_pd(j, method)
    idx = int(np.argmax(p >= 0.5))
    if idx == 0:
        return float(j[0])
    j0, j1, p0, p1 = j[idx - 1], j[idx], p[idx - 1], p[idx]
    return float(j0 + (0.5 - p0) * (j1 - j0) / (p1 - p0))


def main():
    # ================= 处理增益（报告引用） =================
    g_fhss = processing_gain_fhss(10e6, 100e3)
    g_dsss = processing_gain_dsss(1.023e6, 50.0)
    print(">>> 处理增益：FHSS = %.1f dB（10MHz/100kHz）；"
          "DSSS(GPS C/A) = %.1f dB（1.023Mchip/50bps）" % (g_fhss, g_dsss))

    # ================= 1) 自适应跳频抗干扰 =================
    N_CH = 64
    N_HOP = 20000
    WINDOW = 200
    # 场景：静态部分频带干扰机占据 20 个信道（31%）
    n_jam = 20
    rng = np.random.default_rng(2024)
    jammed = rng.choice(N_CH, size=n_jam, replace=False)

    pat_fixed = hop_pattern_fixed_table(N_HOP, N_CH, seed=1)
    pat_uniform = hop_pattern_uniform(N_HOP, N_CH, seed=2)
    pat_chaotic = hop_pattern_chaotic(N_HOP, N_CH, seed=3)
    ch_rl, avail_rl = q_learning_afh(N_CH, jammed, n_hop=N_HOP, seed=4)

    avail_fixed = static_pattern_availability(pat_fixed, jammed, WINDOW)
    avail_uniform = static_pattern_availability(pat_uniform, jammed, WINDOW)
    avail_chaotic = static_pattern_availability(pat_chaotic, jammed, WINDOW)

    print(">>> 抗干扰：静态部分频带干扰占 %d/%d 信道" % (n_jam, N_CH))
    for nm, a in [("固定跳频表", avail_fixed), ("均匀伪随机", avail_uniform),
                  ("混沌跳频", avail_chaotic), ("RL 自适应", avail_rl)]:
        print("    %-8s 稳态可用度 = %.3f" % (nm, a[-2000:].mean()))

    # 稳态可用度 vs 被干扰信道占比
    ratios = np.arange(0.05, 0.65, 0.05)
    sweep_rng = np.random.default_rng(7)
    curves = {"固定跳频表": [], "均匀伪随机": [], "混沌跳频": [], "RL 自适应": []}
    for rho in ratios:
        nj = max(1, int(round(rho * N_CH)))
        js = sweep_rng.choice(N_CH, size=nj, replace=False)
        curves["固定跳频表"].append(steady_availability(pat_fixed, js))
        curves["均匀伪随机"].append(steady_availability(pat_uniform, js))
        curves["混沌跳频"].append(steady_availability(pat_chaotic, js))
        _, a_rl = q_learning_afh(N_CH, js, n_hop=6000, seed=5)
        curves["RL 自适应"].append(float(a_rl[-2000:].mean()))

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.0), layout="constrained")
    styles = {
        "RL 自适应":   ("-", "C2", 2.0),
        "混沌跳频":    ("-.", "C3", 1.5),
        "均匀伪随机":  ("--", "C0", 1.5),
        "固定跳频表":  (":", "C1", 1.5),
    }
    # 左：学习曲线
    axL = axes[0]
    axL.plot(np.arange(N_HOP), avail_fixed * 100, **dict(
        color=styles["固定跳频表"][1], ls=styles["固定跳频表"][0], lw=1.2),
        label="固定跳频表（周期重复，可被掌握）")
    axL.plot(np.arange(N_HOP), avail_uniform * 100, **dict(
        color=styles["均匀伪随机"][1], ls=styles["均匀伪随机"][0], lw=1.3),
        label="均匀伪随机跳频（基线）")
    axL.plot(np.arange(N_HOP), avail_chaotic * 100, **dict(
        color=styles["混沌跳频"][1], ls=styles["混沌跳频"][0], lw=1.4),
        label="混沌跳频（无周期图案）")
    axL.plot(np.arange(N_HOP), avail_rl * 100, **dict(
        color=styles["RL 自适应"][1], ls=styles["RL 自适应"][0], lw=2.0),
        label="RL 自适应跳频（Q-learning 避让）")
    axL.set_xscale("log")
    axL.set_xlabel("跳数（log）")
    axL.set_ylabel("链路可用度 (%)")
    axL.set_title("学习曲线：干扰占据 20/64 信道")
    axL.grid(alpha=0.3)
    axL.legend(loc="center right", fontsize=9)
    axL.set_ylim(0, 105)

    # 右：稳态可用度 vs 被干扰信道占比
    axR = axes[1]
    for nm in ["固定跳频表", "均匀伪随机", "混沌跳频", "RL 自适应"]:
        axR.plot(ratios * 100, np.array(curves[nm]) * 100,
                 color=styles[nm][1], ls=styles[nm][0],
                 lw=styles[nm][2], marker="o", ms=3.5, label=nm)
    axR.plot(ratios * 100, (1 - ratios) * 100, color="0.6", ls="",
             marker="x", ms=5, label="理论基线 1−干扰占比")
    axR.set_xlabel("被干扰信道占比 (%)")
    axR.set_ylabel("稳态链路可用度 (%)")
    axR.set_title("稳态可用度 vs 干扰占比")
    axR.grid(alpha=0.3)
    axR.legend(loc="lower left", fontsize=9)
    axR.set_ylim(0, 105)
    axR.annotate("三种非自适应图案（固定表/均匀/混沌）\n均重合于理论基线 1−ρ：\n不避让被干扰信道，故无增益",
                 xy=(40, 60), xytext=(14, 26), fontsize=8.5, color="0.35",
                 arrowprops=dict(arrowstyle="->", color="0.5", lw=0.8))

    fig.suptitle("图5-1 遥控链路抗干扰：跳频图案与 RL 自适应跳频（依据 [#13][#14]）")
    savefig(fig, "fig1_自适应跳频_抗干扰.png", tight=False)
    plt.close(fig)

    # ================= 2) 欺骗检测 Pd–JSR =================
    JSR = np.arange(-15, 21, 0.5)
    methods = [("power", "功率异常检测", "C1"),
               ("correlation", "相关峰特征检测", "C0"),
               ("ins", "惯导一致性检测", "C2")]
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for key, lab, col in methods:
        pd = spoof_detect_pd(JSR, key) * 100
        ax.plot(JSR, pd, lw=1.8, color=col,
                label=f"{lab}（AUC≈{spoof_detect_roc_auc(key):.2f}）")
    ax.axhline(50, color="0.6", ls=":", lw=1.0)
    ax.axhline(90, color="0.6", ls=":", lw=1.0)
    ax.text(-14.5, 51.5, "Pd=50%", fontsize=8.5, color="0.4")
    ax.text(-14.5, 91.5, "Pd=90%", fontsize=8.5, color="0.4")
    ax.set_xlabel("干信比 JSR (dB)")
    ax.set_ylabel("欺骗检测概率 Pd (%)")
    ax.set_title("图5-2 导航 GNSS 欺骗检测：三条技术路线（依据 [#11][#12]）")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 105)
    ax.annotate("惯导一致性：牵引初起即可察觉\n（独立参考，灵敏度最高）",
                xy=(-6, 50), xytext=(-14, 68), fontsize=9, color="C2",
                arrowprops=dict(arrowstyle="->", color="C2", lw=0.8))
    ax.annotate("功率异常需欺骗强于真实信号\n（约 +3 dB）",
                xy=(3, 50), xytext=(7, 22), fontsize=9, color="C1",
                arrowprops=dict(arrowstyle="->", color="C1", lw=0.8))
    savefig(fig, "fig2_欺骗检测_Pd_JSR.png")
    plt.close(fig)

    # ================= 3) 防护前后导航误差（时域） =================
    t = np.linspace(0, 20, 1000)
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    ax.plot(t, navigation_error_vs_time(t, "none"), "s-", ms=2.5, lw=1.8,
            color="C3", label="无防护：被牵引到 ≈30 m（隐蔽失守）")
    ax.plot(t, navigation_error_vs_time(t, "detect"), "o--", ms=2.5, lw=1.8,
            color="C0", label="检测并剔除欺骗（相关峰/功率，检测延时 ≈1 s）")
    ax.plot(t, navigation_error_vs_time(t, "ins"), "^-", ms=2.5, lw=2.0,
            color="C2", label="惯导辅助一致性检测（IMU 桥接，误差全程受控）")
    ax.axvline(0, color="0.6", lw=1.0, ls=":")
    ax.annotate("欺骗开始", xy=(0.2, 28), fontsize=9, color="0.4")
    ax.axhline(30, color="0.75", ls=":", lw=1.0)
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("导航定位误差 (m)")
    ax.set_title("图5-3 导航 GNSS 抗欺骗：防护策略前后定位误差对比（依据 [#11]）")
    ax.grid(alpha=0.3)
    ax.legend(loc="center right", fontsize=9)
    ax.set_ylim(0, 36)
    savefig(fig, "fig3_防护前后导航误差.png")
    plt.close(fig)

    print()
    print("=== M5 电子反干扰完成 ===")
    print(f"图片输出目录：{_FIG_DIR}")

    # 保存关键指标供 M6 闭环汇总（唯一数据源，避免手工抄录不一致）
    import json
    metrics = {
        "module": "M5 特征支撑电子反干扰",
        "processing_gain_fhss_db": float(g_fhss),
        "processing_gain_dsss_db": float(g_dsss),
        "avail_baseline": float(avail_uniform[-2000:].mean()),
        "avail_rl_afh": float(avail_rl[-2000:].mean()),
        "spoof_detect_pd50_jsr_db": {
            m: _pd50_jsr(m) for m in ("power", "correlation", "ins")},
        "ins_aided_max_error_m": float(
            navigation_error_vs_time(np.linspace(0, 20, 400), "ins").max()),
        "unprotected_max_error_m": float(
            navigation_error_vs_time(np.linspace(0, 20, 400), "none").max()),
    }
    with open(os.path.join(_MODULE_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[已保存] 指标 → {os.path.join(_MODULE_DIR, 'metrics.json')}")


if __name__ == "__main__":
    main()