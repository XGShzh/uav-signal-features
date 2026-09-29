# -*- coding: utf-8 -*-
"""
identify.py
低空无人机信号特征 —— 第三部分（M3 电子侦察：链路识别）
====================================================================
用 M2 提取的多域特征作为特征向量，训练随机森林判别三类链路
（遥控 FHSS / 图传 OFDM / 导航 C/A），输出 识别准确率~SNR 曲线 与混淆矩阵。

特征向量（每样本）：
    [峰均比PAPR, 带宽3dB, 占用带宽99, 谱平坦度, 循环特征强度_dB, 循环谱高频保持度]

训练协议（两种对比）：
    1) 单 SNR 训练：仅在 20 dB 生成训练样本 —— 低 SNR 下脆弱特征
       （PAPR/占用带宽/平坦度）漂出训练分布，分类器塌缩到单一类别；
    2) 多 SNR 训练：在 -25~20 dB 多个信噪比混合训练 —— 覆盖特征随
       SNR 退化的整个流形，低 SNR 识别率大幅提升。

实现要点：
    * 基带信号先归一化到单位功率再加噪：否则三类信号功率差（~1 dB）
      会通过"绝对量级"的循环谱强度泄漏给分类器，在极低 SNR 下产生
      虚假的高识别率（此时所有真实特征均已淹没在噪声里）。
    * 带宽类特征按 fs 归一化（无量纲），避免分类器利用采样率差异。
    * 接收信号经真实信道损伤（CFO/多普勒/相噪/IQ 不平衡/非线性），
      与 M1 的 channel.py 一致，避免"过于理想"的识别率。

====================================================================
文献依据（对应 uav-literature-list.html）
====================================================================
[#5] RF-based drone detection and identification using deep learning,
     M. F. Al-Sa'd et al., FGCS (2019).
     —— 早期基线：存在性检测 99.7%、4 类识别 84.5%；手工特征+分类器
        是其对照基线，本文的随机森林分支对应这一路线。
[#6] DroneRF dataset, Data in Brief (2019)：提供类别划分与识别任务定义。
[#8] ML algorithms for drone detection and classification,
     Frontiers (2024)：对比传统 ML 与 DL 的性能与适用条件。
[#1] RFUAV（arXiv:2503.09033, 2025）：RF 指纹识别是侦察核心任务。

注：与本文的深度学习谱图分支（dl_identify.py，对应 [#4][#5] 的 CNN 路线）
    形成“手工特征 vs 深度特征”的路线对比。
依赖：numpy / sklearn / （复用 part1, part2）
"""

import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 复用 part1（channel）与 part2（features）
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ["part1_M1_信号建模", "part2_M2_信号特征分析"]:
    sys.path.insert(0, os.path.join(_BASE, _p))
from channel import add_channel
from features import time_features, freq_features, cyclic_feature


# 多 SNR 训练网格：覆盖 -25~20 dB（含过渡区 -15~-3 加密点，
# 否则随机森林在特征过渡区插值差，会出现准确率非单调凹陷）
TRAIN_SNRS = [-25.0, -20.0, -15.0, -12.0, -9.0, -6.0, -3.0,
              0.0, 5.0, 10.0, 15.0, 20.0]

# 循环特征抽取参数。nperseg=512（df<=50kHz）：保证 alpha>=0.05MHz 时
# off=round(alpha/2/df)>=1，避免小 alpha 因 off=0 被置零的个数随 fs 不同
# 而不同，在纯噪声下产生随采样率变化的伪特征（类别泄漏）。
_CYC_KW = dict(nperseg=512, n_alpha=30)


def feature_vector(sig, fs):
    """从接收信号提取识别特征向量（复用 part2 的特征函数）。

    带宽类特征按 fs 归一化（无量纲），避免分类器利用"不同采样率在纯噪声
    下带宽仍不同"的伪特征（否则低 SNR 会得到虚假的高识别率）。
    """
    tf = time_features(sig)
    ff = freq_features(sig, fs)
    cf = cyclic_feature(sig, fs, **_CYC_KW)
    return np.array([
        tf["峰均比PAPR_dB"],
        ff["带宽3dB_MHz"] / (fs / 1e6),
        ff["占用带宽99_pct_MHz"] / (fs / 1e6),
        ff["谱平坦度"],
        cf["循环特征强度_dB"],
        cf["循环谱高频保持度"],
    ], dtype=float)


def build_dataset(sigs, snr_db, m_per_class=40, rng=None):
    """在某 SNR 下生成三类信号各 m 个加噪样本 → 特征矩阵 X 与标签 y。

    基带先归一化到单位功率，保证三类信号在同一 SNR 下噪声功率相同
    （消除功率泄漏，见模块 docstring）。
    """
    if rng is None:
        rng = np.random.default_rng(7)
    names = ["fhss", "ofdm", "gnss"]
    labels = ["FHSS", "OFDM", "C/A"]
    X, y = [], []
    for ci, (name, lab) in enumerate(zip(names, labels)):
        fs = sigs[name]["params"]["fs"]
        base = sigs[name]["sig"]
        base = base / np.sqrt(np.mean(np.abs(base) ** 2))   # 单位功率，消除功率泄漏
        for _ in range(m_per_class):
            rx, _ = add_channel(base, fs, snr_db=snr_db, use_multipath=False, seed=rng)
            # 接收端 AGC 归一化：消除“含损伤后信号功率随类别微变”导致的
            # 绝对功率尺度泄漏（否则极低 SNR 下循环特征强度仍可区分类别，
            # 产生虚假高识别率——此时所有真实特征均已淹没在噪声中）。
            rx = rx / np.sqrt(np.mean(np.abs(rx) ** 2) + 1e-30)
            X.append(feature_vector(rx, fs))
            y.append(labels.index(lab))
    return np.array(X), np.array(y), labels


def _train_clf(sigs, train_snrs, m_per_snr, rng):
    """在多个 SNR 上混合采样训练随机森林。"""
    Xs, ys = [], []
    for ts in train_snrs:
        X, y, _ = build_dataset(sigs, ts, m_per_snr, rng=rng)
        Xs.append(X)
        ys.append(y)
    clf = RandomForestClassifier(n_estimators=250, random_state=0, n_jobs=-1)
    clf.fit(np.vstack(Xs), np.concatenate(ys))
    return clf


def evaluate_snr_curve(sigs, snr_list, protocols=None, m_train_per_snr=24,
                       m_test=60, n_seeds=3, seed=7):
    """对比多种训练协议：逐 SNR 泛化测试，多种子给出均值±标准差。

    protocols: {曲线名: 训练SNR列表}。默认对比 多SNR训练 与 单SNR(20dB)训练。
    同一种子的测试集在协议间共享，保证对比公平。
    返回 {曲线名: {SNR: dict(mean, std)}} 与 labels。
    """
    if protocols is None:
        protocols = {
            "多SNR训练(-25~20dB)": TRAIN_SNRS,
            "单SNR训练(20dB)": [20.0],
        }
    out = {p: {s: [] for s in snr_list} for p in protocols}
    labels = None
    for si in range(n_seeds):
        # 测试集：每种子独立，且各协议共用（公平对比）
        tests = {}
        for s in snr_list:
            Xt, yt, labels = build_dataset(
                sigs, s, m_test, rng=np.random.default_rng(seed + 5000 + 97 * si + int(s)))
            tests[s] = (Xt, yt)
        for pname, train_snrs in protocols.items():
            rng_tr = np.random.default_rng(seed + 1000 * si + abs(int(sum(train_snrs))))
            clf = _train_clf(sigs, train_snrs, m_train_per_snr, rng_tr)
            for s in snr_list:
                Xt, yt = tests[s]
                out[pname][s].append(float(np.mean(clf.predict(Xt) == yt)))
    return {p: {s: dict(mean=float(np.mean(v)), std=float(np.std(v)))
                for s, v in snrs.items()}
            for p, snrs in out.items()}, labels


def confusion_matrices(sigs, snr_list, train_snrs=None, m_train_per_snr=14,
                       m_test=100, seed=7):
    """多 SNR 训练一个分类器，在多个 test SNR 下分别评估混淆矩阵。

    返回 (cms, labels)：cms[i] 为 snr_list[i] 处的混淆矩阵（行=真实，列=预测）。
    """
    from sklearn.metrics import confusion_matrix
    if train_snrs is None:
        train_snrs = TRAIN_SNRS
    clf = _train_clf(sigs, train_snrs, m_train_per_snr,
                     np.random.default_rng(seed + 123))
    cms = []
    for snr in snr_list:
        Xt, yt, labels = build_dataset(sigs, snr, m_test,
                                       rng=np.random.default_rng(seed + 456 + int(snr)))
        cm = confusion_matrix(yt, clf.predict(Xt), labels=np.arange(len(labels)))
        cms.append(cm)
    return cms, labels
