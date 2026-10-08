# -*- coding: utf-8 -*-
"""
ecm.py
低空无人机信号特征 —— 第五部分（M5 电子反干扰：ECCM）
====================================================================
用 M2 提取的信号特征，为三类链路设计并评估电子防护（反干扰 / 抗欺骗）：

  A. 抗干扰（遥控链路 FHSS）
     1) 处理增益分析：FHSS 与 DSSS 的扩频处理增益，解释扩频体制的
        天然抗干扰能力（与 M4 的干扰容限对称）。
     2) 跳频图案：固定跳频表 / 均匀伪随机 / **混沌跳频** 三种图案生成。
     3) **RL 自适应跳频（AFH）**：用 Q-learning（bandit 形式）依据链路
        反馈学习避让被干扰信道 —— 反干扰的核心前沿方法。

  B. 抗欺骗（导航链路 GNSS）
     4) 欺骗检测三条技术路线：相关峰特征、功率异常、**惯导一致性**。
     5) 惯导辅助抗欺骗：欺骗牵引期间用 IMU 短时递推桥接，抑制误差。

====================================================================
文献依据（对应 uav-literature-list.html）
====================================================================
[#11] SemperFi: A Spoofer Eliminating GPS Receiver for UAVs,
      H. Sathaye et al., arXiv:2105.01860 (2021).
      —— 惯导辅助 + 连续干扰消除，欺骗下约 10 s 内恢复真实位置；
         本文“惯导一致性检测 + 桥接”据此建模。
[#12] GPS-Spoofing Attack Detection Mechanism for UAV Swarms,
      P. Mykytyn et al., arXiv:2301.12766 (2023).
      —— 基于冗余/一致性（距离、功率）的欺骗检测；本文功率异常与
         一致性检测路线据此。
[#13] Securing UAV Networks: A Lightweight Chaotic-Frequency Hopping
      Approach to Counter Jamming Attacks, IEEE Access (2024).
      —— 混沌跳频序列无周期、图案难被掌握；本文实现混沌跳频图案。
[#14] Adaptive RL-Based FHSS Strategies: A Comparative Analysis of
      Baseline, Tabular Q-Learning, and DQN vs. 1st-Order Markov Jammer.
      —— **表格型 Q-learning 自适应跳频**；本文的 Q-learning AFH 据此，
         并与“基线跳频”“混沌跳频”做对比。
[#9]  GNSS 干扰/欺骗综述（IEEE COMST 2026）—— 反欺骗技术路线梳理。

依赖：numpy
"""

import numpy as np


# ======================================================================
# A1. 处理增益
# ======================================================================
def processing_gain_fhss(b_total_hz, b_channel_hz):
    """跳频处理增益（dB）：G = B_total / B_channel。"""
    return float(10 * np.log10(b_total_hz / b_channel_hz))


def processing_gain_dsss(chip_rate, data_rate):
    """直扩处理增益（dB）：G = R_chip / R_data。"""
    return float(10 * np.log10(chip_rate / data_rate))


# ======================================================================
# A2. 跳频图案生成
# ======================================================================
def hop_pattern_uniform(n_hop, n_ch, seed=0):
    """均匀伪随机跳频图案（基线，等价于 M1 的跳频图案）。"""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_ch, size=n_hop)


def hop_pattern_fixed_table(n_hop, n_ch, seed=0):
    """固定跳频表（周期性重复）——真实系统的常见实现，但图案可被掌握。"""
    rng = np.random.default_rng(seed)
    table_len = min(n_ch, n_hop)
    table = rng.permutation(n_ch)[:table_len]
    return np.tile(table, int(np.ceil(n_hop / table_len)))[:n_hop]


def hop_pattern_chaotic(n_hop, n_ch, seed=0, r=4.0, n_bit=None):
    """混沌跳频图案（Logistic 映射 + 混沌二进制序列抽取）。

    x_{n+1} = r·x_n·(1-x_n)，r=4 时处于混沌态；其二进制位近似独立均匀，
    故用 x 的高位抽取 log2(n_ch) 个比特合成信道号 → 图案无周期、均匀性好。
    依据 [#13]：混沌跳频难以被干扰机预测/掌握。
    """
    if n_bit is None:
        n_bit = int(np.ceil(np.log2(n_ch)))
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.1, 0.9)
    # 预热，消除初值暂态
    for _ in range(200):
        x = r * x * (1 - x)
    out = np.zeros(n_hop, dtype=int)
    for i in range(n_hop):
        idx = 0
        for k in range(n_bit):
            x = r * x * (1 - x)
            bit = int((x * 2.0 ** (k + 1)) % 1.0 >= 0.5)   # 混沌二进制抽取
            idx |= bit << k
        out[i] = idx % n_ch
    return out


# ======================================================================
# A3. Q-learning 自适应跳频（AFH）
# ======================================================================
def q_learning_afh(n_ch, jammed_channels, n_hop=20000, alpha=0.15,
                   eps0=0.30, eps_min=0.01, eps_decay=0.9995,
                   window=200, seed=0):
    """表格型 Q-learning 自适应跳频（bandit 形式）。

    状态：无（bandit）——对每个信道维护价值 Q[c]；
    动作：选择下一跳信道（ε-greedy）；
    奖励：成功（未落入被干扰信道）+1，失败 −1。
    依据 [#14]：用 RL 学习信道质量，避让被干扰信道。

    返回 (channels, avail_curve)：
        channels    : 逐跳选用的信道号
        avail_curve : 滑动窗口链路可用度（长度 n_hop），用于画学习曲线
    """
    jammed = np.zeros(n_ch, dtype=bool)
    jammed[np.asarray(jammed_channels, dtype=int)] = True
    Q = np.zeros(n_ch)
    rng = np.random.default_rng(seed)
    channels = np.zeros(n_hop, dtype=int)
    success = np.zeros(n_hop, dtype=float)
    eps = eps0
    for i in range(n_hop):
        if rng.random() < eps:
            a = int(rng.integers(0, n_ch))          # 探索
        else:
            qmax = Q.max()
            cand = np.flatnonzero(Q >= qmax - 1e-12)
            a = int(rng.choice(cand))               # 利用（并列随机）
        ok = not jammed[a]
        r = 1.0 if ok else -1.0
        Q[a] += alpha * (r - Q[a])
        channels[i] = a
        success[i] = ok
        eps = max(eps_min, eps * eps_decay)
    # 滑动窗口可用度
    cs = np.cumsum(success)
    avail = np.empty(n_hop)
    avail[:window] = cs[:window] / np.arange(1, window + 1)
    avail[window:] = (cs[window:] - cs[:-window]) / window
    return channels, avail


def static_pattern_availability(pattern, jammed_channels, window=200):
    """给定跳频图案与被干扰信道集合，返回滑动窗口链路可用度曲线。"""
    jammed = np.zeros(int(pattern.max()) + 1, dtype=bool)
    jammed[np.asarray(jammed_channels, dtype=int)] = True
    success = (~jammed[pattern]).astype(float)
    cs = np.cumsum(success)
    n = len(success)
    avail = np.empty(n)
    w = min(window, n)
    avail[:w] = cs[:w] / np.arange(1, w + 1)
    avail[w:] = (cs[w:] - cs[:-w]) / w
    return avail


def steady_availability(pattern_or_channels, jammed_channels, tail=2000):
    """稳态链路可用度（取尾部平均）。"""
    p = np.asarray(pattern_or_channels)
    jammed = set(int(c) for c in np.asarray(jammed_channels, dtype=int))
    tail_seq = p[-tail:] if len(p) > tail else p
    return float(np.mean([int(c) not in jammed for c in tail_seq]))


# ======================================================================
# B4. 欺骗检测（三条路线）
# ======================================================================
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def spoof_detect_pd(jsr_db, method="correlation"):
    """欺骗检测概率 Pd 随干信比 JSR 变化（三种技术路线）。

    method:
      'power'       功率异常检测：欺骗信号须强于真实信号，总功率异常升高
                     → 需 JSR 高于约 +3 dB 才可靠（依据 [#12]）。
      'correlation' 相关峰特征检测：欺骗码与真实码不完全对齐，捕获相关峰
                     出现双峰/畸变 → JSR > 约 0 dB 可检（依据 [#12]）。
      'ins'         惯导一致性检测：GNSS 位置与 IMU 递推位置残差增大
                     → 牵引初起即可察觉，JSR > 约 −6 dB 即可检（依据 [#11]）。
    """
    jsr = np.asarray(jsr_db, dtype=float)
    params = {
        "power":       dict(th=+3.0, slope=1.0),   # 50% 检测点约 +3 dB
        "correlation": dict(th=0.0, slope=1.0),    # 50% 检测点约 0 dB
        "ins":         dict(th=-6.0, slope=1.0),   # 50% 检测点约 −6 dB
    }[method]
    return _sigmoid((jsr - params["th"]) / params["slope"])


def spoof_detect_roc_auc(method="ins"):
    """各检测方法的等效 ROC 面积（示意指标，用于对比表）。"""
    return {"power": 0.86, "correlation": 0.93, "ins": 0.98}[method]


# ======================================================================
# B5. 惯导辅助抗欺骗：防护前后导航误差（时域）
# ======================================================================
def navigation_error_vs_time(t, protect="none", off_m=30.0, base_m=2.0,
                             t_capture=5.0, t_detect=1.0, ins_drift_mps=0.6):
    """欺骗开始后，导航定位误差随时间的演化（三种防护策略）。

    t : 时间轴（s），t=0 为欺骗起始
    protect:
      'none'        无防护：误差按捕获过程从 base_m 牵引到 off_m（约 t_capture 秒）
      'detect'      相关峰/功率检测：检测到后剔除欺骗信号 → 误差回落（有检测延时）
      'ins'         惯导辅助一致性检测：检测更早，且牵引期内由 IMU 桥接
                    （误差仅随 IMU 漂移缓慢增长）→ 误差始终受控
    """
    t = np.asarray(t, dtype=float)
    # 欺骗捕获过程（S 形牵引）
    pull = _sigmoid((t - t_capture / 2) / (t_capture / 6))
    err_unprot = base_m + (off_m - base_m) * pull

    if protect == "none":
        return err_unprot
    if protect == "detect":
        # 检测前：与无防护相同；检测时刻后误差迅速回落到真实水平
        post = _sigmoid((t - t_detect) / 0.4)
        # 检测期间已产生的牵引量被修正
        back = _sigmoid((t - t_detect) / 0.4)
        return err_unprot * (1 - back) + base_m * back
    if protect == "ins":
        # 牵引期内 IMU 桥接：误差仅随 IMU 漂移线性增长；检测后修正回真实值
        bridged = base_m + ins_drift_mps * np.clip(t, 0, None)
        hold = _sigmoid((t - t_detect) / 0.4)
        return bridged * (1 - hold) + base_m * hold
    raise ValueError(f"未知 protect 方式：{protect}")