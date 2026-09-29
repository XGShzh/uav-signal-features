# -*- coding: utf-8 -*-
"""
dl_identify.py
低空无人机信号特征 —— 第三部分（M3 电子侦察：深度学习谱图识别）
====================================================================
用 STFT 时频图（谱图）作为输入，训练轻量卷积神经网络（CNN）判别三类链路，
与手工特征 + 随机森林形成“前沿方法 vs 传统方法”的对照。

====================================================================
文献依据（对应 uav-literature-list.html）
====================================================================
[#4] Robust Low-Cost Drone Detection and Classification in Low SNR
     Environments, S. Glüge et al., arXiv:2406.18624 (2024).
     —— 用 SDR + **谱图 + CNN** 做无人机检测，SNR > -12 dB 准确率 ≥85%。
        本文据其范式，用谱图 + CNN 复现该类方法。
[#1] RFUAV: A Benchmark Dataset for UAV Detection and Identification,
     R. Shi et al., arXiv:2503.09033 (2025).
     —— 定义“RF 无人机指纹”，强调时频图像特征是识别核心。
[#5] RF-based drone detection and identification using deep learning,
     M. F. Al-Sa'd et al., Future Generation Computer Systems (2019).
     —— 深度网络直接从原始 RF 序列学习，优于手工特征基线。
[#7] Securing the Skies: A Comprehensive Survey on Anti-UAV Methods,
     Y. Dong et al., arXiv:2504.11967 (2025).
     —— 综述指出深度学习是低空无人机 RF 识别的主流方向。
[#8] Machine learning algorithms applied for drone detection and
     classification, Frontiers in Communications and Networks (2024).
     —— 对比各类 ML/DL 方法的性能与挑战。

实现说明
--------
本机无 GPU 且未安装 PyTorch/TensorFlow，故用 **纯 numpy 实现一个小型 CNN**
（im2col 卷积 + 最大池化 + 全连接 + Softmax），保证可复现、无需外部依赖。
文献 [#4] 使用更深网络；本文以同族方法（谱图 + 卷积）在小样本下对标，
用于体现“深度特征自学习 vs 手工特征”的路线差异，而非追求 SOTA 精度。

依赖：numpy / scipy
"""

import numpy as np
from scipy.signal import spectrogram


# ======================================================================
# 1) 谱图（STFT 时频图）预处理
# ======================================================================
def stft_image(sig, fs, n_f=24, n_t=24, nperseg=256):
    """把接收信号转成归一化对数谱图（n_f × n_t）。

    步骤：STFT → 双边谱 fftshift → 幅度对数压缩 → 频/时重采样
          → 逐样本零均值单位方差归一化（去除绝对功率差异）。
    归一化很关键：否则分类器会利用“绝对功率”而非“调制结构”区分信号，
    在极低 SNR 下产生虚假高识别率。
    """
    f, t, Z = spectrogram(sig, fs=fs, nperseg=nperseg,
                          noverlap=nperseg - nperseg // 4,
                          nfft=nperseg, return_onesided=False)
    Z = np.abs(np.fft.fftshift(Z, axes=0))          # 双边谱，频率单调
    A = 10.0 * np.log10(Z + 1e-12)                  # 对数压缩（dB）
    # 频率/时间方向线性重采样到固定尺寸
    fi = np.linspace(0, A.shape[0] - 1, n_f).astype(int)
    ti = np.linspace(0, A.shape[1] - 1, n_t).astype(int)
    A = A[np.ix_(fi, ti)]
    # 逐样本归一化
    A = (A - A.mean()) / (A.std() + 1e-8)
    return A.astype(np.float32)


# ======================================================================
# 2) 卷积/池化基础算子（im2col 实现）
# ======================================================================
def _im2col(x, kh, kw, stride=1, pad=1):
    """(N,C,H,W) → (N*out_h*out_w, C*kh*kw)"""
    N, C, H, W = x.shape
    out_h = (H + 2 * pad - kh) // stride + 1
    out_w = (W + 2 * pad - kw) // stride + 1
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    cols = np.zeros((N, C, kh, kw, out_h, out_w), dtype=x.dtype)
    for i in range(kh):
        i_max = i + stride * out_h
        for j in range(kw):
            j_max = j + stride * out_w
            cols[:, :, i, j, :, :] = xp[:, :, i:i_max:stride, j:j_max:stride]
    return cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1), out_h, out_w


def _col2im(cols, x_shape, kh, kw, stride=1, pad=1):
    """im2col 的逆运算（梯度反传）"""
    N, C, H, W = x_shape
    out_h = (H + 2 * pad - kh) // stride + 1
    out_w = (W + 2 * pad - kw) // stride + 1
    cols = cols.reshape(N, out_h, out_w, C, kh, kw).transpose(0, 3, 4, 5, 1, 2)
    xp = np.zeros((N, C, H + 2 * pad, W + 2 * pad), dtype=cols.dtype)
    for i in range(kh):
        i_max = i + stride * out_h
        for j in range(kw):
            j_max = j + stride * out_w
            xp[:, :, i:i_max:stride, j:j_max:stride] += cols[:, :, i, j, :, :]
    return xp[:, :, pad:pad + H, pad:pad + W]


def _maxpool(x, size=2, stride=2):
    N, C, H, W = x.shape
    out_h = (H - size) // stride + 1
    out_w = (W - size) // stride + 1
    out = np.zeros((N, C, out_h, out_w), dtype=x.dtype)
    argmax = np.zeros((N, C, out_h, out_w), dtype=np.int64)
    for i in range(out_h):
        for j in range(out_w):
            blk = x[:, :, i * stride:i * stride + size,
                    j * stride:j * stride + size].reshape(N, C, -1)
            argmax[:, :, i, j] = blk.argmax(axis=2)
            out[:, :, i, j] = blk.max(axis=2)
    return out, argmax


def _maxpool_back(dout, argmax, x_shape, size=2, stride=2):
    N, C, H, W = x_shape
    out_h, out_w = dout.shape[2], dout.shape[3]
    dx = np.zeros((N, C, H, W), dtype=dout.dtype)
    for i in range(out_h):
        for j in range(out_w):
            blk = dx[:, :, i * stride:i * stride + size,
                     j * stride:j * stride + size].reshape(N, C, -1)
            np.put_along_axis(blk, argmax[:, :, i, j][:, :, None],
                              dout[:, :, i, j][:, :, None], axis=2)
            dx[:, :, i * stride:i * stride + size,
               j * stride:j * stride + size] = blk.reshape(N, C, size, size)
    return dx


# ======================================================================
# 3) 轻量 CNN（谱图 → 三类）
# ======================================================================
class SmallCNN:
    """小型 CNN：Conv(6)-ReLU-Pool → Conv(12)-ReLU-Pool → FC(64)-ReLU → FC(K)。

    纯 numpy 实现（前进/反传手写），Adam 优化，Softmax 交叉熵。
    """

    def __init__(self, in_shape=(24, 24), n_class=3, seed=0,
                 f1=6, f2=12, fc=64, lr=3e-3, l2=1e-4):
        rng = np.random.default_rng(seed)
        self.in_shape = in_shape
        self.n_class = n_class
        self.lr = lr
        self.l2 = l2
        H, W = in_shape

        def he(shape, fan_in):
            return rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)

        # 卷积核 (F, C, kh, kw)
        self.W1 = he((f1, 1, 3, 3), 9)
        self.b1 = np.zeros(f1)
        self.W2 = he((f2, f1, 3, 3), f1 * 9)
        self.b2 = np.zeros(f2)

        # 两次 3×3(pad1) 不改变尺寸；两次 2×2 池化后尺寸减半两次
        h1, w1 = H, W
        h2, w2 = h1 // 2, w1 // 2
        h3, w3 = h2 // 2, w2 // 2
        self._flat = f2 * h3 * w3
        self.W3 = he((fc, self._flat), self._flat)
        self.b3 = np.zeros(fc)
        self.W4 = he((n_class, fc), fc)
        self.b4 = np.zeros(n_class)

        self._init_adam()

    # ---------------- Adam 状态 ----------------
    def _init_adam(self):
        self._params = ["W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4"]
        self._m = {k: np.zeros_like(getattr(self, k)) for k in self._params}
        self._v = {k: np.zeros_like(getattr(self, k)) for k in self._params}
        self._t = 0
        self._beta1, self._beta2, self._eps = 0.9, 0.999, 1e-8

    # ---------------- 前向 ----------------
    def forward(self, X, cache=None):
        N = X.shape[0]
        x = X.reshape(N, 1, *self.in_shape)

        c1, oh, ow = _im2col(x, 3, 3, 1, 1)
        z1 = c1 @ self.W1.reshape(self.W1.shape[0], -1).T + self.b1
        z1 = z1.reshape(N, oh, ow, -1).transpose(0, 3, 1, 2)   # (N,F1,oh,ow)
        a1 = np.maximum(z1, 0)
        p1, am1 = _maxpool(a1, 2, 2)

        c2, oh2, ow2 = _im2col(p1, 3, 3, 1, 1)
        z2 = c2 @ self.W2.reshape(self.W2.shape[0], -1).T + self.b2
        z2 = z2.reshape(N, oh2, ow2, -1).transpose(0, 3, 1, 2)
        a2 = np.maximum(z2, 0)
        p2, am2 = _maxpool(a2, 2, 2)

        flat = p2.reshape(N, -1)
        z3 = flat @ self.W3.T + self.b3
        a3 = np.maximum(z3, 0)
        logits = a3 @ self.W4.T + self.b4

        if cache is not None:
            cache.update(dict(x=x, c1=c1, z1=z1, a1=a1, p1=p1, am1=am1,
                              c2=c2, z2=z2, a2=a2, p2=p2, am2=am2,
                              flat=flat, z3=z3, a3=a3, oh=oh, ow=ow))
        return logits

    @staticmethod
    def _softmax(logits):
        z = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    # ---------------- 反传 + 更新 ----------------
    def _step(self, X, y):
        N = X.shape[0]
        cache = {}
        logits = self.forward(X, cache)
        prob = self._softmax(logits)
        loss = -np.mean(np.log(prob[np.arange(N), y] + 1e-12))

        # dlogits
        dlogits = prob.copy()
        dlogits[np.arange(N), y] -= 1.0
        dlogits /= N

        g = {}
        g["W4"] = dlogits.T @ cache["a3"] + self.l2 * self.W4
        g["b4"] = dlogits.sum(0)
        da3 = dlogits @ self.W4
        dz3 = da3 * (cache["z3"] > 0)
        g["W3"] = dz3.T @ cache["flat"] + self.l2 * self.W3
        g["b3"] = dz3.sum(0)

        # --- 全连接 → 卷积层 ---
        dflat = dz3 @ self.W3                                  # (N, f2*h3*w3)
        dp2 = dflat.reshape(cache["p2"].shape)
        da2 = _maxpool_back(dp2, cache["am2"], cache["a2"].shape, 2, 2)
        dz2 = da2 * (cache["z2"] > 0)                          # (N,F2,12,12)

        W2f = self.W2.reshape(self.W2.shape[0], -1)             # (F2, C*9)
        dz2_cols = dz2.transpose(0, 2, 3, 1).reshape(-1, self.W2.shape[0])
        g["W2"] = (dz2_cols.T @ cache["c2"] + self.l2 * W2f).reshape(self.W2.shape)
        g["b2"] = dz2.sum(axis=(0, 2, 3))
        dcols2 = dz2_cols @ W2f
        dp1 = _col2im(dcols2, cache["p1"].shape, 3, 3, 1, 1)    # (N,F1,12,12)

        da1 = _maxpool_back(dp1, cache["am1"], cache["a1"].shape, 2, 2)
        dz1 = da1 * (cache["z1"] > 0)                          # (N,F1,24,24)

        W1f = self.W1.reshape(self.W1.shape[0], -1)             # (F1, 9)
        dz1_cols = dz1.transpose(0, 2, 3, 1).reshape(-1, self.W1.shape[0])
        g["W1"] = (dz1_cols.T @ cache["c1"] + self.l2 * W1f).reshape(self.W1.shape)
        g["b1"] = dz1.sum(axis=(0, 2, 3))

        # Adam 更新
        self._t += 1
        for k in self._params:
            self._m[k] = self._beta1 * self._m[k] + (1 - self._beta1) * g[k]
            self._v[k] = self._beta2 * self._v[k] + (1 - self._beta2) * (g[k] ** 2)
            mhat = self._m[k] / (1 - self._beta1 ** self._t)
            vhat = self._v[k] / (1 - self._beta2 ** self._t)
            setattr(self, k, getattr(self, k) - self.lr * mhat / (np.sqrt(vhat) + self._eps))
        return loss

    # ---------------- 训练/预测 ----------------
    def fit(self, X, y, epochs=40, batch=32, seed=0, verbose=False):
        rng = np.random.default_rng(seed)
        n = X.shape[0]
        for ep in range(epochs):
            idx = rng.permutation(n)
            losses = []
            for i in range(0, n, batch):
                b = idx[i:i + batch]
                losses.append(self._step(X[b], y[b]))
            if verbose and (ep + 1) % 10 == 0:
                print(f"    epoch {ep+1}/{epochs} loss={np.mean(losses):.4f}")
        return self

    def predict_proba(self, X, batch=128):
        out = []
        for i in range(0, X.shape[0], batch):
            out.append(self._softmax(self.forward(X[i:i + batch])))
        return np.vstack(out)

    def predict(self, X, batch=128):
        return self.predict_proba(X, batch).argmax(axis=1)


# ======================================================================
# 4) 数据集构建与评测（供 run_part3 调用）
# ======================================================================
def build_image_dataset(sigs, snr_db, m_per_class=40, n_f=24, n_t=24,
                        add_channel_fn=None, seed=7):
    """生成三类信号的谱图数据集，返回 (X, y, labels)。

    接收端加入真实信道损伤（与手工特征分支一致），保证两路线可比。
    """
    names = ["fhss", "ofdm", "gnss"]
    labels = ["FHSS", "OFDM", "C/A"]
    rng = np.random.default_rng(seed)
    X, y = [], []
    for ci, name in enumerate(names):
        fs = sigs[name]["params"]["fs"]
        base = sigs[name]["sig"]
        base = base / np.sqrt(np.mean(np.abs(base) ** 2))
        for _ in range(m_per_class):
            rx, _ = add_channel_fn(base, fs, snr_db=snr_db,
                                   use_multipath=False, seed=rng)
            # 接收端 AGC 归一化，与手工特征分支一致（消除功率尺度泄漏）
            rx = rx / (np.sqrt(np.mean(np.abs(rx) ** 2)) + 1e-30)
            X.append(stft_image(rx, fs, n_f, n_t))
            y.append(ci)
    return np.array(X, dtype=np.float32), np.array(y), labels


def evaluate_cnn_curve(sigs, snr_list, add_channel_fn, train_snrs=None,
                       m_train_per_snr=16, m_test=50, n_f=24, n_t=24,
                       epochs=40, n_seeds=2, seed=7, verbose=True):
    """多 SNR 训练 CNN，逐 SNR 泛化测试（与随机森林分支同协议）。

    返回 {snr: dict(mean, std)}，与 identify.evaluate_snr_curve 同格式。
    """
    if train_snrs is None:
        train_snrs = [-25.0, -20.0, -15.0, -12.0, -9.0, -6.0,
                      -3.0, 0.0, 5.0, 10.0, 15.0, 20.0]
    out = {s: [] for s in snr_list}
    for si in range(n_seeds):
        rng = np.random.default_rng(seed + 1000 * si)
        Xs, ys = [], []
        for ts in train_snrs:
            Xi, yi, labels = build_image_dataset(
                sigs, ts, m_train_per_snr, n_f, n_t, add_channel_fn,
                seed=int(rng.integers(1 << 30)))
            Xs.append(Xi); ys.append(yi)
        Xtr = np.vstack(Xs); ytr = np.concatenate(ys)
        if verbose:
            print(f"  [CNN] 训练集 {Xtr.shape}，种子 {si+1}/{n_seeds}")
        net = SmallCNN(in_shape=(n_f, n_t), n_class=len(labels), seed=si)
        net.fit(Xtr, ytr, epochs=epochs, seed=si)
        for s in snr_list:
            Xt, yt, _ = build_image_dataset(
                sigs, s, m_test, n_f, n_t, add_channel_fn,
                seed=int(rng.integers(1 << 30)))
            out[s].append(float(np.mean(net.predict(Xt) == yt)))
    return {s: dict(mean=float(np.mean(v)), std=float(np.std(v)))
            for s, v in out.items()}


if __name__ == "__main__":
    # 自检：验证 CNN 在可分数据上能收敛
    import os, sys
    _BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(_BASE, "part1_M1_信号建模"))
    from signal_models import generate_all
    from channel import add_channel

    sigs = generate_all()
    print("自检：构造 SNR=10dB 数据集并训练 15 epoch …")
    X, y, labels = build_image_dataset(sigs, 10.0, 30, add_channel_fn=add_channel)
    print("  数据:", X.shape, "类别分布:", np.bincount(y))
    net = SmallCNN(in_shape=(24, 24), n_class=3, seed=0)
    net.fit(X, y, epochs=15, seed=0, verbose=True)
    acc = np.mean(net.predict(X) == y)
    print(f"  训练集准确率 = {acc*100:.1f}%（应明显高于随机 33.3%）")