# -*- coding: utf-8 -*-
"""
run_part6.py
低空无人机信号特征 —— 第六部分（M6 特征价值闭环评估）主入口
====================================================================
汇总 M2–M5 的实测指标，闭合「一套信号特征支撑对抗三线」的论证闭环：

    fig1_闭环总览图.png    信号特征为中心 → 三线辐射，标注各线关键指标
                           + 返回路径（三线结果反哺特征库 → 闭环）
    fig2_三线效能总表.png   一页总表：侦察 / 干扰 / 反干扰 的量化效能
    fig3_特征价值对比.png   三条线各自的「特征带来多少增益」柱状对比

数据来源
--------
**不手工抄录**，全部读取各模块运行后被自动写出的 metrics.json：
    part2_M2_信号特征分析/metrics.json
    part3_M3_电子侦察/metrics.json
    part4_M4_电子干扰/metrics.json
    part5_M5_电子反干扰/metrics.json
若某模块 metrics.json 缺失，对应指标显示为"—"并提示先行运行该模块。

用法：
    # 先分别运行 M2-M5，再运行本脚本
    python run_part6.py
依赖：numpy / matplotlib
"""

import json
import os

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_FIG_DIR = os.path.join(_MODULE_DIR, "figures")
os.environ["MPLCONFIGDIR"] = os.path.join(_MODULE_DIR, ".mplcache")

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

for _f in ["Microsoft YaHei", "SimHei", "KaiTi"]:
    if any(_f.lower() in f.lower() for f in matplotlib.font_manager.get_font_names()):
        plt.rcParams["font.sans-serif"] = [_f]
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10

_BASE = os.path.dirname(_MODULE_DIR)
MODULES = {
    "M2": os.path.join(_BASE, "part2_M2_信号特征分析"),
    "M3": os.path.join(_BASE, "part3_M3_电子侦察"),
    "M4": os.path.join(_BASE, "part4_M4_电子干扰"),
    "M5": os.path.join(_BASE, "part5_M5_电子反干扰"),
}


def load_metrics():
    """读取各模块 metrics.json，缺失则给出提示。"""
    out = {}
    for key, path in MODULES.items():
        fp = os.path.join(path, "metrics.json")
        if os.path.exists(fp):
            with open(fp, "r", encoding="utf-8") as f:
                out[key] = json.load(f)
        else:
            out[key] = None
            print(f"[提示] 未找到 {key} 的 metrics.json，请先运行该模块：{path}")
    return out


def savefig(fig, name, tight=True):
    os.makedirs(_FIG_DIR, exist_ok=True)
    path = os.path.join(_FIG_DIR, name)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[已保存] {path}")


def _g(m, *keys, default=None):
    """安全取值：_g(m, 'a', 'b') 等价 m['a']['b']。"""
    cur = m
    for k in keys:
        if cur is None:
            return default
        try:
            cur = cur[k] if not isinstance(k, int) else cur[k]
        except (KeyError, IndexError, TypeError):
            return default
        if cur is None:
            return default
    return cur


# ======================================================================
# 汇总表数据（从 metrics.json 提取，缺失则为 None）
# ======================================================================
def build_summary(met):
    m2, m3, m4, m5 = met.get("M2"), met.get("M3"), met.get("M4"), met.get("M5")

    rows = []

    # ---- 特征底座（M2） ----
    rows.append([
        "特征底座", "M2",
        "特征库规模",
        (f"{_g(m2,'n_signal_types')} 类信号 × {_g(m2,'n_features')} 维"
         if m2 else "—"),
        "三类链路多域特征可完全分离",
    ])
    rows.append([
        "特征底座", "M2",
        "跳频图案提取精度",
        (f"{_g(m2,'hop_extract_max_error_khz'):.1f} kHz"
         if m2 else "—"),
        (f"信道间隔 {_g(m2,'hop_channel_spacing_khz'):.0f} kHz，误差 < 1/12"
         if m2 else "—"),
    ])
    rows.append([
        "特征底座", "M2",
        "OFDM 调制域 EVM",
        (f"{_g(m2,'evm_clean_pct'):.2f}% (干净) / "
         f"{_g(m2,'evm_snr15_pct'):.2f}% (15dB)" if m2 else "—"),
        "观测底噪声下的调制质量特征",
    ])

    # ---- 电子侦察（M3） ----
    gain = _g(m3, "cfd_sensitivity_gain_db") if m3 else None
    rows.append([
        "电子侦察", "M3",
        "弱信号检测灵敏度增益",
        (f"CFD 优于 ED {gain:.1f} dB" if gain is not None else "—"),
        (f"ED 存在 SNR 墙 {_g(m3,'snr_wall_db'):.1f} dB（理论吻合）"
         if m3 else "—"),
    ])
    acc_rf = _g(m3, "acc_rf_multisnr") if m3 else None
    acc_cnn = _g(m3, "acc_cnn") if m3 else None
    rows.append([
        "电子侦察", "M3",
        "链路识别准确率 @-12dB",
        (f"手工特征+RF {acc_rf.get('-12', acc_rf.get(-12, '—'))}% / "
         f"CNN {acc_cnn.get('-12', acc_cnn.get(-12, '—'))}%"
         if (acc_rf and acc_cnn) else "—"),
        "手工特征 vs 深度学习两条路线对比",
    ])
    rows.append([
        "电子侦察", "M3",
        "单/多 SNR 训练对比 @-18dB",
        (f"多SNR {acc_rf.get('-18', acc_rf.get(-18,'—'))}% vs "
         f"单SNR {_g(m3,'acc_rf_singlesnr','-18', default=_g(m3,'acc_rf_singlesnr',-18))}%"
         if acc_rf else "—"),
        "训练协议设计决定低 SNR 可用性",
    ])

    # ---- 电子干扰（M4） ----
    rows.append([
        "电子干扰", "M4",
        "精准干扰增益",
        (f"{_g(m4,'precision_jamming_gain_db'):.1f} dB"
         if m4 else "—"),
        (f"跟随式 {_g(m4,'jsr_follow_db'):.1f} dB vs 宽带 "
         f"{_g(m4,'jsr_wideband_db'):.1f} dB 达 BER=1e-2" if m4 else "—"),
    ])
    rows.append([
        "电子干扰", "M4",
        "跳频处理增益（理论）",
        (f"{_g(m4,'fhss_channel_gain_db'):.1f} dB" if m4 else "—"),
        "干扰增益 ≈ 处理增益，物理自洽",
    ])

    # ---- 电子反干扰（M5） ----
    av_b = _g(m5, "avail_baseline") if m5 else None
    av_rl = _g(m5, "avail_rl_afh") if m5 else None
    rows.append([
        "电子反干扰", "M5",
        "自适应跳频链路可用度",
        (f"{av_rl*100:.1f}% vs 基线 {av_b*100:.1f}%" if (av_b and av_rl) else "—"),
        "RL（Q-learning）避让被干扰信道",
    ])
    pd50 = _g(m5, "spoof_detect_pd50_jsr_db") or {}
    if all(k in pd50 for k in ("ins", "correlation", "power")):
        def _sgn(v):
            v = round(v)
            return (("−" if v < 0 else "+") if v != 0 else "") + f"{abs(v):.0f}"
        spoof_txt = (f"惯导 {_sgn(pd50['ins'])} dB / "
                     f"相关峰 {_sgn(pd50['correlation'])} dB / "
                     f"功率 {_sgn(pd50['power'])} dB")
    else:
        spoof_txt = "—"
    rows.append([
        "电子反干扰", "M5",
        "欺骗检测灵敏度（Pd=50%）",
        spoof_txt,
        "惯导一致性独立参考最优（功率检测需欺骗强于真实信号）",
    ])
    rows.append([
        "电子反干扰", "M5",
        "抗欺骗定位误差",
        (f"无防护 {_g(m5,'unprotected_max_error_m'):.0f} m → "
         f"惯导辅助 {_g(m5,'ins_aided_max_error_m'):.1f} m" if m5 else "—"),
        "误差全程受控，回扣 #11 SemperFi",
    ])
    return rows


# ======================================================================
# fig1 闭环总览图
# ======================================================================
def draw_closed_loop(met, rows):
    m3, m4, m5 = met.get("M3"), met.get("M4"), met.get("M5")
    fig, ax = plt.subplots(figsize=(12.4, 7.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec, fs=10, tc="black", weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.6,rounding_size=1.6",
                                    fc=fc, ec=ec, lw=1.8, mutation_aspect=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, weight=weight, linespacing=1.5)

    def arrow(x1, y1, x2, y2, color, style="-|>", lw=2.0, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle=style, mutation_scale=16,
                                     color=color, lw=lw, ls=ls,
                                     connectionstyle="arc3,rad=0.0"))

    # 中心：特征
    core_txt = ("低空无人机信号特征\n（特征库）\n"
                "遥控 FHSS · 图传 OFDM · 导航 GNSS\n"
                "时/频/时频/循环平稳/调制域")
    box(31, 39, 38, 22, core_txt, "#F3EEFC", "#7C3AED", fs=11,
        tc="#5B21B6", weight="bold")

    # 三条线
    gain = _g(m3, "cfd_sensitivity_gain_db")
    acc_rf = _g(m3, "acc_rf_multisnr") or {}
    acc_cnn = _g(m3, "acc_cnn") or {}
    recog = acc_rf.get("-12", acc_rf.get(-12, "—"))
    cnnv = acc_cnn.get("-12", acc_cnn.get(-12, "—"))
    detect_txt = (f"弱信号检测：CFD 优于 ED {gain:.1f} dB" if gain is not None
                  else "弱信号检测")
    box(4, 72, 30, 20,
        "电子侦察（M3）\n" + detect_txt +
        f"\n识别 @-12dB：手工 {recog}% / CNN {cnnv}%",
        "#E6F6F9", "#0E7490", fs=9.5, tc="#0B4A5A")

    g_jam = _g(m4, "precision_jamming_gain_db")
    box(66, 72, 30, 20,
        "电子干扰（M4）\n" +
        (f"精准干扰增益 {g_jam:.1f} dB\n" if g_jam is not None else "") +
        "跟随式 vs 盲目宽带\nGNSS 欺骗（受控牵引）/ 压制（失锁）",
        "#FDF1E7", "#C2410C", fs=9.5, tc="#7C2D12")

    av_b = _g(m5, "avail_baseline")
    av_rl = _g(m5, "avail_rl_afh")
    err_u = _g(m5, "unprotected_max_error_m")
    err_i = _g(m5, "ins_aided_max_error_m")
    anti_txt = "电子反干扰（M5）\n"
    if av_rl is not None and av_b is not None:
        anti_txt += f"RL 自适应跳频可用度 {av_rl*100:.1f}%\n（基线 {av_b*100:.1f}%）\n"
    if err_u is not None and err_i is not None:
        anti_txt += f"惯导辅助：误差 {err_u:.0f} m → {err_i:.1f} m"
    box(35, 8, 30, 22, anti_txt, "#EFF8E8", "#347A0E", fs=9.5, tc="#22510A")

    # 辐射箭头
    arrow(38, 61, 19, 72, "#0E7490")     # → 侦察
    arrow(50, 61, 81, 72, "#C2410C")     # → 干扰
    arrow(50, 39, 50, 30, "#347A0E")     # → 反干扰

    # 闭环返回路径（三线结果反哺特征库）
    arrow(19, 72, 33, 58, "#7C3AED", lw=1.6, ls="--")
    arrow(81, 72, 67, 58, "#7C3AED", lw=1.6, ls="--")
    arrow(65, 19, 63, 39, "#7C3AED", lw=1.6, ls="--")
    ax.text(50, 92.5, "闭环：三线的对抗效能反过来验证 / 迭代特征库",
            ha="center", fontsize=10, color="#5B21B6")
    ax.text(50, 3.2, "创新点：一套信号特征，支撑对抗三线（认知 → 利用）",
            ha="center", fontsize=10.5, color="#5B21B6", weight="bold")

    ax.set_title("图6-1 特征价值闭环总览：一套信号特征支撑对抗三线",
                 fontsize=13, pad=6)
    savefig(fig, "fig1_闭环总览图.png", tight=False)
    plt.close(fig)


# ======================================================================
# fig2 三线效能总表（表格图）
# ======================================================================
def draw_summary_table(rows):
    fig, ax = plt.subplots(figsize=(13.2, 6.2))
    ax.axis("off")
    col_labels = ["对抗线", "模块", "关键指标", "实测结果", "说明"]
    cell_text = [[r[0], r[1], r[2], r[3], r[4]] for r in rows]

    tbl = ax.table(cellText=cell_text, colLabels=col_labels,
                   cellLoc="left", loc="center",
                   colWidths=[0.10, 0.05, 0.20, 0.27, 0.38])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.65)

    line_color = {"特征底座": "#7C3AED", "电子侦察": "#0E7490",
                  "电子干扰": "#C2410C", "电子反干扰": "#347A0E"}
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#D9E1EC")
        if r == 0:
            cell.set_facecolor("#EEF3F9")
            cell.set_text_props(weight="bold", color="#1B2430")
            cell.set_height(0.075)
        else:
            key = cell_text[r - 1][0]
            cell.set_facecolor("#FFFFFF" if r % 2 else "#F7F9FC")
            if c == 0:
                cell.set_text_props(weight="bold", color=line_color.get(key, "black"))
    ax.set_title("图6-2 三线效能总表（数据取自各模块运行输出的 metrics.json，非手工抄录）",
                 fontsize=12.5, pad=14)
    savefig(fig, "fig2_三线效能总表.png")
    plt.close(fig)


# ======================================================================
# fig3 特征价值对比（柱状）
# ======================================================================
def draw_value_bars(met):
    m3, m4, m5 = met.get("M3"), met.get("M4"), met.get("M5")
    items, values, colors = [], [], []

    g = _g(m3, "cfd_sensitivity_gain_db")
    if g is not None:
        items.append("侦察：弱信号检测\n灵敏度增益 (dB)"); values.append(g); colors.append("#0E7490")
    g = _g(m4, "precision_jamming_gain_db")
    if g is not None:
        items.append("干扰：精准干扰\n功率增益 (dB)"); values.append(g); colors.append("#C2410C")
    av_b, av_rl = _g(m5, "avail_baseline"), _g(m5, "avail_rl_afh")
    if av_b is not None and av_rl is not None:
        items.append("反干扰：自适应跳频\n可用度提升 (百分点)")
        values.append((av_rl - av_b) * 100); colors.append("#347A0E")
    err_u, err_i = _g(m5, "unprotected_max_error_m"), _g(m5, "ins_aided_max_error_m")
    if err_u is not None and err_i is not None:
        items.append("反干扰：抗欺骗\n误差下降倍数 (×)")
        values.append(err_u / max(err_i, 1e-9)); colors.append("#347A0E")

    if not items:
        print("[提示] 无可用指标，跳过 fig3")
        return

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    bars = ax.bar(range(len(items)), values, color=colors, width=0.55)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.02,
                (f"{v:.1f}" if v < 100 else f"{v:.0f}"),
                ha="center", va="bottom", fontsize=10.5)
    ax.set_xticks(range(len(items)), items, fontsize=9.5)
    ax.set_ylabel("增益 / 提升幅度")
    ax.set_title("图6-3 特征价值：三条线各自的量化增益")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(values) * 1.18)
    ax.text(0.99, -0.30, "注：各线指标单位不同（dB / 百分点 / 倍数），已在横轴标注，"
                         "此图仅作量级对比",
            transform=ax.transAxes, ha="right", fontsize=8.5, color="0.4")
    savefig(fig, "fig3_特征价值对比.png")
    plt.close(fig)


def main():
    print(">>> 读取各模块 metrics.json …")
    met = load_metrics()
    missing = [k for k, v in met.items() if v is None]
    if missing:
        print(f"[警告] 缺少指标：{missing}（对应指标将以 '—' 显示）")

    rows = build_summary(met)

    print("\n================= 三线效能总表（M6 闭环汇总） =================")
    for r in rows:
        print(f"  [{r[0]:<5}] {r[2]:<18} : {r[3]}")
    print("=============================================================")

    draw_closed_loop(met, rows)
    draw_summary_table(rows)
    draw_value_bars(met)

    # 保存汇总表 CSV（与图一致的数据源）
    csv_path = os.path.join(_MODULE_DIR, "三线效能总表.csv")
    with open(csv_path, "w", encoding="utf-8-sig") as f:
        f.write("对抗线,模块,关键指标,实测结果,说明\n")
        for r in rows:
            f.write(",".join(str(x).replace(",", "，") for x in r) + "\n")
    print(f"\n[已保存] 总表 CSV → {csv_path}")
    print("=== M6 特征价值闭环评估完成 ===")
    print(f"图片输出目录：{_FIG_DIR}")


if __name__ == "__main__":
    main()