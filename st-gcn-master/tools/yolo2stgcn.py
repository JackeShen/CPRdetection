"""
用 YOLO pose 从视频/图片序列提取骨架，转成 ST-GCN 训练数据 (npy + pkl)。

作用：把 YOLO 的 COCO 17 点映射成 ST-GCN 的 OpenPose 18 点（补 neck 点），
      再组织成 ST-GCN 认的 (N, 3, T, V, M) 张量 + (样本名, 标签) 的 pkl。

输入目录结构（video_root 下按类别分文件夹）：
    video_root/
        动作A/  a1.mp4              # 视频文件
        动作A/  a1/                 # 或图片序列目录(含 *.jpg)  <- SupC14 整理后就是这个
        动作B/  b1.mp4 ...
类别文件夹名按字母排序，label = 排序索引 (0 .. N-1)。

用法（在 st-gcn 项目根目录，oldshen 环境）：
    python tools/yolo2stgcn.py \
        --video_root ./mydata \
        --model ../ultralytics-main/pose/yolo26m-pose.pt \
        --out ./data/my_dataset \
        --max_frame 300 --num_person 2 --val_ratio 0.2

输出：
    <out>/train_data.npy / train_label.pkl
    <out>/val_data.npy   / val_label.pkl
    <out>/label_map.json  (类别名 -> label 索引)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import random
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# COCO 17 点 (YOLO pose) -> OpenPose 18 点 (ST-GCN) 索引映射
COCO_TO_OPENPOSE = {
    0: 0,    # nose          -> nose
    1: 15,   # left_eye      -> left_eye
    2: 14,   # right_eye     -> right_eye
    3: 17,   # left_ear      -> left_ear
    4: 16,   # right_ear     -> right_ear
    5: 5,    # left_shoulder -> left_shoulder
    6: 2,    # right_shoulder-> right_shoulder
    7: 6,    # left_elbow    -> left_elbow
    8: 3,    # right_elbow   -> right_elbow
    9: 7,    # left_wrist    -> left_wrist
    10: 4,   # right_wrist   -> right_wrist
    11: 11,  # left_hip      -> left_hip
    12: 8,   # right_hip     -> right_hip
    13: 12,  # left_knee     -> left_knee
    14: 9,   # right_knee    -> right_knee
    15: 13,  # left_ankle    -> left_ankle
    16: 10,  # right_ankle   -> right_ankle
}
NUM_JOINTS = 18
NECK_OP = 1
RSHOULDER_OP = 2
LSHOULDER_OP = 5


def coco17_to_openpose18(xy, conf):
    """单个人的 17 点 -> 18 点。

    xy:   [17, 2] 归一化坐标 (0~1)
    conf: [17]    每个关键点置信度
    返回: pose18 [18, 2], score18 [18]
    """
    pose = np.zeros((NUM_JOINTS, 2), dtype=np.float32)
    score = np.zeros((NUM_JOINTS,), dtype=np.float32)
    for coco_idx, op_idx in COCO_TO_OPENPOSE.items():
        pose[op_idx] = xy[coco_idx]
        score[op_idx] = conf[coco_idx]
    # 补 neck：左右肩中点
    pose[NECK_OP] = (pose[RSHOULDER_OP] + pose[LSHOULDER_OP]) / 2.0
    score[NECK_OP] = (score[RSHOULDER_OP] + score[LSHOULDER_OP]) / 2.0
    return pose, score


def select_cpr_person(boxes, img_w, img_h):
    """CPR 场景：多人时排除「躺在地上」的人，返回施救者索引（通常 1 个）。

    躺在地上的人特征（用户描述）：
      - bbox 更宽（人横躺，水平跨度大）
      - 位于画面下方（底边 y 更大）
    判据：lie_score = 归一化框宽 + 归一化底边位置，取最大者为躺着的人并排除；
    排除后若仍有多人，取最靠上（顶部 y 最小）者作为施救者。
    """
    n = len(boxes)
    if n <= 1:
        return list(range(n))
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    bw = (x2 - x1) / max(img_w, 1)        # 归一化框宽
    by2 = y2 / max(img_h, 1)              # 归一化框底边（越靠下越大）
    lie_score = bw + by2                   # 躺着嫌疑分
    lie_idx = int(np.argmax(lie_score))
    keep = [i for i in range(n) if i != lie_idx]
    if len(keep) > 1:
        keep.sort(key=lambda i: y1[i])     # 取最靠上的施救者
        keep = keep[:1]
    return keep


def infer_frame(model, frame, w, h, num_person, imgsz, conf, device, select="cpr"):
    """对单帧做 YOLO，返回 (3, 18, num_person) 的该帧骨骼数据；无人则返回 None。

    select='cpr' 时多人中只保留 CPR 施救者（排除躺在地上的人）；
    否则按置信度排序取前 num_person。
    """
    res = model.predict(source=frame, conf=conf, imgsz=imgsz, device=device, verbose=False)
    kpts = res[0].keypoints
    if kpts is None or kpts.shape[0] == 0:
        return None
    xy = kpts.xy.cpu().numpy()       # [P, 17, 2] 像素坐标
    confs = kpts.conf.cpu().numpy()  # [P, 17]
    xy[:, :, 0] /= w
    xy[:, :, 1] /= h                 # 归一化到 0~1

    order = confs.sum(axis=1).argsort()[::-1].tolist()  # 默认：置信度降序
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


def frames_to_data(frame_getter, total, w, h, max_frame, num_person, imgsz, conf, device, select="cpr"):
    """把一序列帧转成 (3, max_frame, 18, num_person) 张量并中心化。

    frame_getter(i) -> BGR ndarray (第 i 帧)
    """
    if total > max_frame:
        idx = np.linspace(0, total - 1, max_frame).round().astype(int)
    else:
        idx = list(range(total))

    data = np.zeros((3, max_frame, NUM_JOINTS, num_person), dtype=np.float32)
    last_out = None
    for t, fi in enumerate(idx):
        frame = frame_getter(fi)
        if frame is None:
            if last_out is not None:
                data[:, t, :, :] = last_out   # 缺帧沿用上一帧
            continue
        out = infer_frame(model_ref, frame, w, h, num_person, imgsz, conf, device, select)
        if out is not None:
            data[:, t, :, :] = out
            last_out = out
        elif last_out is not None:
            data[:, t, :, :] = last_out       # 偶发空检测沿用上一帧

    # 中心化 + 置信度为 0 的点坐标置 0（与 kinetics_gendata 一致）
    data[0:2] = data[0:2] - 0.5
    data[0][data[2] == 0] = 0.0
    data[1][data[2] == 0] = 0.0
    return data


# 让 infer_frame 能拿到 model：在 process_* 里临时设引用（保持简单）
model_ref = None


def process_video(model, video_path, max_frame, num_person, imgsz, conf, device, select="cpr"):
    global model_ref
    model_ref = model
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"无法读取视频: {video_path}")
    h, w = frames[0].shape[:2]
    return frames_to_data(lambda i: frames[i], len(frames), w, h,
                           max_frame, num_person, imgsz, conf, device, select)


def process_image_seq(model, dir_path, max_frame, num_person, imgsz, conf, device, select="cpr"):
    global model_ref
    model_ref = model
    fns = sorted(glob.glob(os.path.join(dir_path, "*.jpg")) +
                 glob.glob(os.path.join(dir_path, "*.png")) +
                 glob.glob(os.path.join(dir_path, "*.jpeg")) +
                 glob.glob(os.path.join(dir_path, "*.bmp")))
    total = len(fns)
    if total == 0:
        raise RuntimeError(f"目录里没有图片: {dir_path}")
    im0 = cv2.imread(fns[0])
    if im0 is None:
        raise RuntimeError(f"无法读取首帧: {fns[0]}")
    h, w = im0.shape[:2]
    return frames_to_data(lambda i: cv2.imread(fns[i]), total, w, h,
                           max_frame, num_person, imgsz, conf, device, select)


def save_split(samples, out_dir, tag):
    """把 [(name, label, data), ...] 存成 <tag>_data.npy + <tag>_label.pkl。"""
    if not samples:
        return
    names = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    datas = np.stack([s[2] for s in samples])
    np.save(out_dir / f"{tag}_data.npy", datas)
    with open(out_dir / f"{tag}_label.pkl", "wb") as f:
        pickle.dump((names, labels), f)
    print(f"已保存 {tag}: {datas.shape} ({len(labels)} 个样本)")


def parse_args():
    p = argparse.ArgumentParser(description="YOLO pose -> ST-GCN 数据转换")
    p.add_argument("--video_root", required=True, help="按类别分文件夹的数据根目录")
    p.add_argument("--model", required=True, help="YOLO pose 模型路径, 如 yolo26m-pose.pt")
    p.add_argument("--out", default="./data/my_dataset", help="输出目录")
    p.add_argument("--max_frame", type=int, default=300, help="每样本固定帧数 T")
    p.add_argument("--num_person", type=int, default=1, help="每帧保留人数 M（CPR 场景建议 1，仅施救者）")
    p.add_argument("--select", default="cpr", choices=["cpr", "top_conf"],
                   help="多人筛选策略：cpr=排除躺地者保留施救者；top_conf=按置信度取前 num_person")
    p.add_argument("--imgsz", type=int, default=640, help="YOLO 推理尺寸")
    p.add_argument("--conf", type=float, default=0.25, help="检测置信度阈值")
    p.add_argument("--device", default=None, help="推理设备, 如 cpu / 0")
    p.add_argument("--val_ratio", type=float, default=0.2, help="验证集比例(分层抽样)")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    from ultralytics import YOLO

    model = YOLO(args.model)
    video_root = Path(args.video_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    categories = sorted(d.name for d in video_root.iterdir() if d.is_dir())
    if not categories:
        raise SystemExit(f"在 {video_root} 下没找到类别文件夹")
    label_map = {c: i for i, c in enumerate(categories)}
    print("类别 -> label:", json.dumps(label_map, ensure_ascii=False))

    samples = []
    for cat in categories:
        cat_dir = video_root / cat
        for e in sorted(os.listdir(cat_dir)):
            ep = cat_dir / e
            if ep.is_file() and ep.suffix.lower() in VIDEO_EXTS:
                name = f"{cat}/{ep.stem}"
                data = process_video(model, ep, args.max_frame, args.num_person,
                                     args.imgsz, args.conf, args.device, args.select)
            elif ep.is_dir():
                if not (glob.glob(str(ep / "*.jpg")) or glob.glob(str(ep / "*.png"))):
                    continue
                name = f"{cat}/{ep.name}"
                data = process_image_seq(model, ep, args.max_frame, args.num_person,
                                          args.imgsz, args.conf, args.device, args.select)
            else:
                continue
            samples.append((name, label_map[cat], data))
            print(f"  完成 {name} -> shape {data.shape}")

    if not samples:
        raise SystemExit("没有处理任何样本")

    train, val = [], []
    for cat in categories:
        idx = label_map[cat]
        cat_samples = [s for s in samples if s[1] == idx]
        random.shuffle(cat_samples)
        n_val = max(1, int(round(len(cat_samples) * args.val_ratio))) if len(cat_samples) >= 2 else 0
        val += cat_samples[:n_val]
        train += cat_samples[n_val:]

    save_split(train, out_dir, "train")
    save_split(val, out_dir, "val")
    with open(out_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"完成。label 映射见 {out_dir / 'label_map.json'}")


if __name__ == "__main__":
    main()
