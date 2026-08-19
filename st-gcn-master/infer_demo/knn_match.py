"""k-NN 模板匹配工具。

逻辑：
- 训练集的 84 个样本（已存为 train_data.npy）+ 它们的标签
- 推理时输入一个新骨架张量（3, T, 18, M），与所有训练样本算 L2 距离
- 取 Top-K 邻居，按 1/(distance + eps) 距离倒数加权投票
- 票数最多的类别 = 预测类别
- 对训练样本本身，1-NN 准确率就是 100%（最近邻就是自己）

效果：把欠拟合/小数据问题转化为"模式匹配"，对训练集完美过拟合，对验证集能反映训练集学到的模式。
"""
from __future__ import annotations

import numpy as np


def knn_predict(query: np.ndarray,
                train_data: np.ndarray,
                train_labels: np.ndarray,
                k: int = 5) -> tuple:
    """
    Parameters
    ----------
    query : np.ndarray  shape=(3, T, 18, M)，推理样本骨架张量
    train_data : np.ndarray  shape=(N, 3, T, 18, M)，训练样本张量库
    train_labels : np.ndarray  shape=(N,)，训练样本的整型标签 (0..13)
    k : int，邻居数

    Returns
    -------
    top_idx : np.ndarray  shape=(k,)，Top-K 邻居在 train_data 中的下标
    top_dist : np.ndarray  shape=(k,)，对应的 L2 距离
    top_labels : np.ndarray  shape=(k,)，对应的标签
    """
    if query.shape != train_data.shape[1:]:
        raise ValueError(
            f"query 形状 {query.shape} 与 train 形状 {train_data.shape[1:]} 不一致"
        )
    q = query.astype(np.float32).reshape(-1)
    N = train_data.shape[0]
    # reshape 一次比每次循环快
    flat = train_data.reshape(N, -1).astype(np.float32)
    # 向量化 L2：||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
    dists2 = (flat ** 2).sum(axis=1) + (q ** 2).sum() - 2 * flat.dot(q)
    dists2 = np.clip(dists2, 0.0, None)
    dists = np.sqrt(dists2)
    k_actual = min(k, N)
    top_idx = dists.argsort()[:k_actual]
    top_dist = dists[top_idx]
    top_labels = train_labels[top_idx].astype(np.int64)
    return top_idx, top_dist, top_labels
