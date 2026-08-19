#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLO pose + ST-GCN 端到端动作识别推理脚本

流程：
  视频 / 图片序列
    -> YOLO26-pose 逐帧提 COCO 17 点
    -> 映射成 OpenPose 18 点（补 neck = 左右肩中点）
    -> 坐标归一化 + 中心化 (-0.5~0.5)
    -> ST-GCN 前向
    -> 输出 14 类动作预测（Top-K + 置信度）

用法（在 st-gcn-master 目录下，oldshen 环境）：
  # 视频
  python infer_demo/infer.py --source demo.mp4

  # 图片序列目录
  python infer_demo/infer.py --source ./mydata/SupC00/SupC00_r00_ch0

  # 自定义模型/类别名，并标注输出视频
  python infer_demo/infer.py --source demo.mp4 \
      --stgcn_model work_dir/recognition/my_dataset/ST_GCN/epoch30_model.pt \
      --annotate out.mp4 --topk 3
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# 让脚本能 import st-gcn 的 net 包 与 tools 里的映射函数
ROOT = Path(__file__).resolve().parent.parent          # st-gcn-master
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from yolo2stgcn import coco17_to_openpose18, select_cpr_person, NUM_JOINTS  # noqa: E402
from net.st_gcn import Model  # noqa: E402
from ultralytics import YOLO  # noqa: E402

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------
def load_stgcn(checkpoint: str, num_class: int, device: str) -> Model:
    model = Model(in_channels=3, num_class=num_class,
                  edge_importance_weighting=True,
                  graph_args={"layout": "openpose", "strategy": "spatial"})
    weights = torch.load(checkpoint, map_location=device)
    weights = {k.split("module.")[-1]: v for k, v in weights.items()}
    model.load_state_dict(weights)
    model.to(device)
    model.eval()
    return model


def load_label_map(path: str) -> dict:
    """读 label_map.json，返回 {label索引: 类别名}。优先用同级 class_names.json（中文名）。"""
    import json
    base = os.path.dirname(path)
    cn_path = os.path.join(base, "class_names.json")
    if os.path.exists(cn_path):
        with open(cn_path, "r", encoding="utf-8") as f:
            cn_map = json.load(f)
        return {int(k): v for k, v in cn_map.items()}
    with open(path, "r", encoding="utf-8") as f:
        name2idx = json.load(f)
    return {int(v): k for k, v in name2idx.items()}


def resolve_device(dev: str):
    """把 'auto'/'0'/'cpu' 统一成 (yolo_device, torch_device)。"""
    use_cuda = torch.cuda.is_available()
    if dev in ("", "auto", None):
        yolo_dev = "0" if use_cuda else "cpu"
    else:
        yolo_dev = dev
    torch_dev = "cuda:0" if yolo_dev in ("0", "cuda", "cuda:0") else "cpu"
    return yolo_dev, torch_dev


# ---------------------------------------------------------------------------
# 单帧骨架提取（逻辑与 yolo2stgcn.infer_frame 一致）
# ---------------------------------------------------------------------------
def extract_one(yolo, frame, w, h, num_person, imgsz, conf, device, select):
    res = yolo.predict(source=frame, conf=conf, imgsz=imgsz,
                       device=device, verbose=False)
    kpts = res[0].keypoints
    if kpts is None or kpts.shape[0] == 0:
        return None
    xy = kpts.xy.cpu().numpy()          # [P, 17, 2] 像素坐标
    confs = kpts.conf.cpu().numpy()     # [P, 17]
    xy[:, :, 0] /= w
    xy[:, :, 1] /= h                    # 归一化 0~1

    order = confs.sum(axis=1).argsort()[::-1].tolist()   # 默认置信度降序
    if select == "cpr":
        boxes = res[0].boxes.xyxy.cpu().numpy() if res[0].boxes is not None else None
        if boxes is not None and len(boxes) == confs.shape[0]:
            cpr_order = select_cpr_person(boxes, w, h)
            if cpr_order:
                order = cpr_order
    order = order[:num_person]

    out = np.zeros((3, NUM_JOINTS, num_person), dtype=np.float32)
    for m_i, p in enumerate(order):
        pose, sc = coco17_to_openpose18(xy[p], confs[p])
        out[0, :, m_i] = pose[:, 0]
        out[1, :, m_i] = pose[:, 1]
        out[2, :, m_i] = sc
    return out


def build_tensor(skels, max_frame, num_person):
    """把骨架列表 -> (3, T, 18, M) 张量并中心化。skels 元素为 (3,18,M) 或 None。"""
    total = len(skels)
    if total > max_frame:
        idx = np.linspace(0, total - 1, max_frame).round().astype(int)
    else:
        idx = list(range(total))
    T = len(idx)
    data = np.zeros((3, T, NUM_JOINTS, num_person), dtype=np.float32)
    last = None
    for t, i in enumerate(idx):
        s = skels[i]
        if s is not None:
            data[:, t, :, :] = s
            last = s
        elif last is not None:
            data[:, t, :, :] = last          # 空帧沿用上一帧
    data[0:2] = data[0:2] - 0.5              # 中心化
    data[0][data[2] == 0] = 0.0
    data[1][data[2] == 0] = 0.0
    return data


# ---------------------------------------------------------------------------
# 读取源（视频 或 图片序列目录）
# ---------------------------------------------------------------------------
def iter_frames(source):
    """返回 (frames生成器, total帧数)。"""
    if os.path.isdir(source):
        fns = sorted(glob.glob(os.path.join(source, "*.jpg")) +
                     glob.glob(os.path.join(source, "*.png")) +
                     glob.glob(os.path.join(source, "*.jpeg")) +
                     glob.glob(os.path.join(source, "*.bmp")))
        if not fns:
            raise SystemExit(f"目录里没有图片: {source}")
        def gen():
            for fn in fns:
                im = cv2.imread(fn)
                if im is not None:
                    yield im
        return gen(), len(fns)
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise SystemExit(f"无法打开视频: {source}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        def gen():
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
            cap.release()
        return gen(), total


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="YOLO pose + ST-GCN 动作识别推理")
    p.add_argument("--source", required=True, help="视频文件 或 图片序列目录")
    p.add_argument("--stgcn_model", default=str(ROOT / "work_dir/recognition/my_dataset/ST_GCN/epoch30_model.pt"),
                   help="ST-GCN 权重 .pt 路径")
    p.add_argument("--yolo_model", default=str(ROOT.parent / "ultralytics-main/pose/yolo26m-pose.pt"),
                   help="YOLO pose 模型路径")
    p.add_argument("--label_map", default=str(ROOT / "data/my_dataset/out/label_map.json"),
                   help="label_map.json 路径")
    p.add_argument("--device", default="auto", help="auto / cpu / 0")
    p.add_argument("--num_person", type=int, default=1, help="每帧保留人数")
    p.add_argument("--select", default="cpr", choices=["cpr", "top_conf"],
                   help="多人筛选策略")
    p.add_argument("--max_frame", type=int, default=300, help="固定帧数 T")
    p.add_argument("--imgsz", type=int, default=640, help="YOLO 推理尺寸")
    p.add_argument("--conf", type=float, default=0.25, help="检测置信度阈值")
    p.add_argument("--topk", type=int, default=3, help="显示前 K 个预测")
    p.add_argument("--annotate", default=None, help="可选：输出标注视频路径 (如 out.mp4)")
    p.add_argument("--use_knn", action="store_true",
                   help="使用 k-NN 模板匹配替代 ST-GCN softmax（针对训练集 1-NN 100%）")
    p.add_argument("--knn_k", type=int, default=5, help="k-NN 投票邻居数")
    p.add_argument("--train_data", default=str(ROOT / "data/my_dataset/out/train_data.npy"),
                   help="训练骨架张量路径（KNN 模板库）")
    p.add_argument("--train_label", default=str(ROOT / "data/my_dataset/out/train_label.pkl"),
                   help="训练标签路径（KNN 模板库）")
    args = p.parse_args()

    yolo_dev, torch_dev = resolve_device(args.device)

    # 1) 加载模型与类别名
    print(f"[1/4] 加载 YOLO pose 模型: {args.yolo_model}")
    yolo = YOLO(args.yolo_model)
    idx2name = load_label_map(args.label_map)
    num_class = len(idx2name)
    print(f"[2/4] 加载 ST-GCN 模型: {args.stgcn_model} (num_class={num_class}, device={torch_dev})")
    stgcn = load_stgcn(args.stgcn_model, num_class, torch_dev)

    # 2) 逐帧提骨架
    print(f"[3/4] 提取骨架: {args.source}")
    gen, total = iter_frames(args.source)
    skels = []
    h = w = None
    for frame in gen:
        if h is None:
            h, w = frame.shape[:2]
        skels.append(extract_one(yolo, frame, w, h, args.num_person,
                                 args.imgsz, args.conf, yolo_dev, args.select))
    n_det = sum(1 for s in skels if s is not None)
    print(f"      共 {total} 帧，有效检测 {n_det} 帧")

    if n_det == 0:
        raise SystemExit("没有检测到任何人体，无法推理。")

    # 3) 组张量 + 前向
    print("[4/4] ST-GCN 前向推理")
    data = build_tensor(skels, args.max_frame, args.num_person)      # (3, T, 18, M)
    x = torch.from_numpy(data).float().unsqueeze(0).to(torch_dev)    # (1, 3, T, 18, M)
    with torch.no_grad():
        out = stgcn(x)
    probs = torch.softmax(out, dim=1)[0].cpu().numpy()

    # 4) k-NN 模板匹配（可选）：把过拟合变成"训练集最近邻"
    if args.use_knn:
        print("[KNN] 模板匹配模式")
        from knn_match import knn_predict
        train_data = np.load(args.train_data)
        _, train_labels = pickle.load(open(args.train_label, "rb"))
        k = args.knn_k
        knn_top, knn_dist, knn_labels = knn_predict(
            data, train_data, np.array(train_labels), k=k)
        # 用加权后的票数作为新的"置信度"：每个邻居按 1/(距离+ε) 投票
        weights = 1.0 / (knn_dist + 1e-6)
        from collections import defaultdict
        vote_scores = defaultdict(float)
        for lab, w in zip(knn_labels, weights):
            vote_scores[int(lab)] += float(w)
        # 全部 14 类的票数（包括没投到的填 0）
        votes = np.zeros(num_class, dtype=np.float32)
        for c, s in vote_scores.items():
            if 0 <= c < num_class:
                votes[c] = s
        # 归一化到 [0,1]
        total_v = votes.sum()
        if total_v > 0:
            knn_probs = votes / total_v
        else:
            knn_probs = votes
        probs = knn_probs
        topk = probs.argsort()[::-1][:args.topk]
        print(f"      k={k} 近邻标签: {[int(l) for l in knn_labels]}")
    else:
        topk = probs.argsort()[::-1][:args.topk]

    print("\n========== 预测结果 ==========")
    for rank, c in enumerate(topk, 1):
        print(f"  Top{rank}: {idx2name[int(c)]:<8s} (label {int(c)})  置信度 {probs[c]*100:5.1f}%")
    print("==============================")

    # 4) 可选标注输出
    if args.annotate:
        annotate(args.source, args.annotate, idx2name[int(topk[0])], float(probs[topk[0]]))
        print(f"\n已保存标注视频: {args.annotate}")


def annotate(source, out_path, label, conf):
    """把预测结果画到每一帧上，输出视频。"""
    if os.path.isdir(source):
        fns = sorted(glob.glob(os.path.join(source, "*.jpg")) +
                     glob.glob(os.path.join(source, "*.png")))
        im0 = cv2.imread(fns[0])
    else:
        cap = cv2.VideoCapture(source)
        ok, im0 = cap.read()
        cap.release()
        if not ok:
            raise SystemExit("标注时无法读取源")
    h, w = im0.shape[:2]
    # 用 H.264 (avc1) 而非 mp4v：后者是 MPEG-4 Visual，Chrome/Edge/Firefox 不支持解码
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"avc1"), 25.0, (w, h))

    text = f"{label}  {conf*100:.1f}%"
    if os.path.isdir(source):
        for fn in fns:
            im = cv2.imread(fn)
            cv2.putText(im, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 3)
            writer.write(im)
    else:
        cap = cv2.VideoCapture(source)
        while True:
            ok, im = cap.read()
            if not ok:
                break
            cv2.putText(im, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 3)
            writer.write(im)
        cap.release()
    writer.release()


if __name__ == "__main__":
    main()
