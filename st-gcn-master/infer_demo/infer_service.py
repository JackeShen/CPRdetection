"""YOLO+ST-GCN 推理服务层。

设计要点：
- 复用 infer.py 的函数（extract_one / build_tensor / load_stgcn / iter_frames / annotate）
- 后台线程跑推理，stdout 重定向到 log 队列
- 通过 task_id 跟踪任务状态
- 保存全 14 类概率 + 骨架帧 JSON，供前端可视化
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # st-gcn-master
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "infer_demo"))
sys.path.insert(0, str(ROOT / "tools"))    # 为了 yolo2stgcn.coco17_to_openpose18

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402
from infer import (                                            # noqa: E402
    load_stgcn, load_label_map, resolve_device,
    extract_one, build_tensor, iter_frames, annotate,
    VIDEO_EXTS, IMG_EXTS,
)
from ultralytics import YOLO                                    # noqa: E402
from yolo2stgcn import (                                       # noqa: E402
    coco17_to_openpose18, select_cpr_person, NUM_JOINTS,
)

# task_id -> {
#   "status", "log", "result", "error",
#   "source": str,                         # 原始路径
#   "skel_json_path": str,                # 骨架帧 JSON 路径
#   "probs_json_path": str,                # 全 14 类概率
#   "keyframe_paths": [str],               # 关键帧图片（图片序列模式）
# }
_TASKS: dict[str, dict] = {}
_TASKS_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()    # 全局串行锁：YOLO/ST-GCN 共享 CUDA 上下文，并发会 illegal memory access
HISTORY_PATH = Path(__file__).resolve().parent / "outputs" / "history.json"


def _is_video(p: str) -> bool:
    return os.path.splitext(p)[1].lower() in VIDEO_EXTS


def _is_dir_images(p: str) -> bool:
    return os.path.isdir(p) and any(
        f.lower().endswith(tuple(IMG_EXTS))
        for f in os.listdir(p)
    )


def _history_load() -> dict:
    try:
        return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"tasks": []}


def _history_save(h: dict):
    HISTORY_PATH.parent.mkdir(exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(h, ensure_ascii=False, indent=2),
                            encoding="utf-8")


def create_task() -> str:
    tid = uuid.uuid4().hex[:8]
    rec = {
        "status": "pending",
        "log": deque(maxlen=2000),
        "result": None,
        "error": None,
        "started_at": time.time(),
        "source": None,
    }
    with _TASKS_LOCK:
        _TASKS[tid] = rec
    return tid


def get_task(tid: str) -> dict | None:
    with _TASKS_LOCK:
        return _TASKS.get(tid)


def _log(tid: str, msg: str):
    with _TASKS_LOCK:
        t = _TASKS.get(tid)
        if t is None:
            return
        t["log"].append(msg)


def _save_skel_and_probs(tid: str, source_basename: str,
                         data: np.ndarray, probs: np.ndarray,
                         idx2name: dict) -> dict:
    """把骨架张量和全 14 类概率保存成 JSON，返回 URL 信息。"""
    out_dir = HISTORY_PATH.parent
    out_dir.mkdir(exist_ok=True)
    skel_path = out_dir / f"{tid}_skel.json"
    probs_path = out_dir / f"{tid}_probs.json"

    # 骨架：取 T 维 5 个均匀帧 + 每帧 (x,y,score) for each (joint, person)
    T = data.shape[1]
    sample_idx = sorted(set(
        [0, T // 4, T // 2, 3 * T // 4, T - 1] + [T // 2]
    ))
    frames = []
    for ti in sample_idx:
        ti = max(0, min(T - 1, ti))
        x = data[0, ti]                   # (18, M)
        y = data[1, ti]
        s = data[2, ti]
        M = x.shape[1]
        people = []
        for m in range(M):
            points = []
            for j in range(18):
                points.append({
                    "joint": j,
                    "name": OPENPOSE_18_NAMES[j],
                    "x": float(x[j, m]) + 0.5,
                    "y": float(y[j, m]) + 0.5,
                    "score": float(s[j, m]),
                })
            people.append(points)
        frames.append({"t": int(ti), "people": people})

    skel_path.write_text(json.dumps(
        {"task_id": tid, "source": source_basename,
         "frames": frames, "layout": "openpose"}, ensure_ascii=False),
        encoding="utf-8")

    # 14 类全概率
    probs_payload = {
        "task_id": tid,
        "classes": [{"index": int(c), "name": idx2name[c], "prob": float(probs[c])}
                    for c in range(len(probs))],
    }
    probs_path.write_text(json.dumps(probs_payload, ensure_ascii=False),
                          encoding="utf-8")
    return {
        "skel_url": f"/outputs/{tid}_skel.json",
        "probs_url": f"/outputs/{tid}_probs.json",
    }


OPENPOSE_18_NAMES = [
    "nose", "neck", "r_shoulder", "r_elbow", "r_wrist",
    "l_shoulder", "l_elbow", "l_wrist",
    "r_hip", "r_knee", "r_ankle",
    "l_hip", "l_knee", "l_ankle",
    "r_eye", "l_eye", "r_ear", "l_ear",
]
OPENPOSE_18_EDGES = [
    (0, 1), (0, 14), (0, 15), (14, 16), (15, 17),
    (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13),
]


def run_inference_task(
    task_id: str,
    source: str,
    stgcn_model: str,
    yolo_model: str,
    label_map: str,
    device: str = "auto",
    num_person: int = 1,
    select: str = "cpr",
    max_frame: int = 300,
    imgsz: int = 640,
    conf: float = 0.25,
    topk: int = 3,
    method: str = "stgcn",
    knn_k: int = 1,
    train_data: str = "",
    train_label: str = "",
):
    """后台跑推理；所有状态写入 _TASKS[task_id]。
    method: 'stgcn' / 'knn' / 'both'
    """
    try:
        with _TASKS_LOCK:
            _TASKS[task_id]["status"] = "running"
            _TASKS[task_id]["source"] = source

        if not (os.path.isdir(source) or os.path.isfile(source)):
            raise SystemExit(f"路径不存在: {source}")
        if not _is_dir_images(source) and not _is_video(source):
            raise SystemExit(f"路径必须是图片序列目录或视频: {source}")

        yolo_dev, torch_dev = resolve_device(device)

        _log(task_id, f"[1/4] 加载 YOLO pose 模型: {yolo_model}")
        yolo = YOLO(yolo_model)

        idx2name = load_label_map(label_map)
        num_class = len(idx2name)
        _log(task_id, f"[2/4] 加载 ST-GCN 模型: {stgcn_model} (num_class={num_class}, device={torch_dev})")
        stgcn = load_stgcn(stgcn_model, num_class, torch_dev)

        if method != "stgcn":
            try:
                from knn_match import knn_predict
            except Exception as e:
                raise SystemExit(f"KNN 模块加载失败: {e}")
            train_arr = np.load(train_data)
            _, train_lbls = pickle.load(open(train_label, "rb"))
            train_lbls = np.array(train_lbls, dtype=np.int64)
            _log(task_id, f"      KNN 模板就绪 ({train_arr.shape}), method={method} k={knn_k}")

        _log(task_id, f"[3/4] 提取骨架: {source}")
        gen, total = iter_frames(source)
        skels: list = []
        h = w = None
        for i, frame in enumerate(gen):
            if h is None:
                h, w = frame.shape[:2]
            sk = extract_one(yolo, frame, w, h, num_person,
                             imgsz, conf, yolo_dev, select)
            skels.append(sk)
            if (i + 1) % 30 == 0:
                _log(task_id, f"      进度: {i+1}/{total or len(skels)} 帧")
        n_det = sum(1 for s in skels if s is not None)
        _log(task_id, f"      共 {len(skels)} 帧，有效检测 {n_det} 帧")

        if n_det == 0:
            raise SystemExit("没有检测到任何人体，无法推理。")

        _log(task_id, "[4/4] 推理")
        data = build_tensor(skels, max_frame, num_person)
        x = torch.from_numpy(data).float().unsqueeze(0).to(torch_dev)
        with torch.no_grad():
            out = stgcn(x)
        stgcn_probs = torch.softmax(out, dim=1)[0].cpu().numpy()

        # KNN 概率（如果需要）
        knn_probs = None
        if method != "stgcn":
            from knn_match import knn_predict
            _, knn_dist, knn_labels = knn_predict(
                data, train_arr, train_lbls, k=knn_k)
            from collections import defaultdict
            knn_top1_int = int(knn_labels[0])  # 1-NN：最近邻的标签
            votes = np.zeros(num_class, dtype=np.float32)
            for lab in knn_labels.tolist():
                votes[int(lab)] += 1.0
            votes = votes / max(1, len(knn_labels))  # 投票比例
            knn_probs = votes
            _log(task_id,
                 f"      knn={method} k={knn_k}  Top1={idx2name[knn_top1_int]} (dist={float(knn_dist[0]):.2f}) neighbors={knn_labels.tolist()}")

        # 按 method 选最终 probs
        # 关键点：先把两路 Top1 在最顶部定义好（下面 method 分支里要用）
        stgcn_top1 = {"label": int(stgcn_probs.argmax()), "name": idx2name[int(stgcn_probs.argmax())],
                      "confidence": float(stgcn_probs.max())}
        knn_top1 = ({"label": int(knn_probs.argmax()), "name": idx2name[int(knn_probs.argmax())],
                     "confidence": float(knn_probs.max())}
                    if knn_probs is not None else None)

        if method == "stgcn":
            probs = stgcn_probs
            method_label = "ST-GCN"
        elif method == "knn":
            probs = knn_probs
            method_label = f"KNN (k={knn_k})"
        else:  # both：综合两路，按各自置信度高者拿 Top1
            # 各自归一化："置信度"用 max prob
            conf_s = float(stgcn_probs.max())
            # KNN 1-NN 的"置信度"用 1/(1+距离)
            conf_k = float(1.0 / (1.0 + knn_dist[0])) if knn_dist[0] >= 0 else float(knn_probs.max())
            # 取两路 top1 中置信度高者作为最终 Top1；Top3 是双方各 Top3 的并集
            top_idx_s = stgcn_probs.argsort()[::-1]
            top_idx_k = knn_probs.argsort()[::-1]
            if conf_s >= conf_k:
                chosen_src = "stgcn"; top1_label = int(top_idx_s[0])
            else:
                chosen_src = "knn";   top1_label = int(top_idx_k[0])
            # 最终概率：stgcn 加权 0.6 + knn 加权 0.4 拼，再把 Top1 顶到首位
            probs = 0.6 * stgcn_probs + 0.4 * knn_probs
            method_label = f"综合 (ST-GCN {conf_s*100:.1f}% vs KNN {conf_k*100:.1f}%) → 选 {chosen_src}"
            _log(task_id, f"      综合决策: ST-GCN 置信度 {conf_s*100:.1f}%  KNN 置信度 {conf_k*100:.1f}%  => {chosen_src}")

        top_idx = probs.argsort()[::-1][:topk]
        top_results = [
            {"rank": int(r + 1), "label": int(c), "name": idx2name[int(c)],
             "confidence": float(probs[c])}
            for r, c in enumerate(top_idx)
        ]

        _log(task_id, "")
        _log(task_id, f"========== 预测结果 ({method_label}) ==========")
        for r in top_results:
            _log(task_id, f"  Top{r['rank']}: {r['name']}  概率 {r['confidence']*100:5.1f}%")
        if method == "both":
            _log(task_id, f"      ST-GCN: {stgcn_top1['name']} ({stgcn_top1['confidence']*100:.1f}%)")
            if knn_top1: _log(task_id, f"      KNN   : {knn_top1['name']} ({knn_top1['confidence']*100:.1f}%)")
        _log(task_id, "==============================")

        # 标注输出视频
        out_dir = Path(__file__).resolve().parent / "outputs"
        out_dir.mkdir(exist_ok=True)
        annotate_path = str(out_dir / f"{task_id}_annotated.mp4")
        annotate_url = None
        try:
            annotate(source, annotate_path, top_results[0]["name"],
                      top_results[0]["confidence"])
            annotate_url = f"/outputs/{task_id}_annotated.mp4"
            _log(task_id, f"已保存标注视频: {annotate_path}")
        except Exception as e:
            _log(task_id, f"标注视频生成失败: {e}")

        # 骨架 + 全概率 JSON
        try:
            urls = _save_skel_and_probs(task_id, os.path.basename(source.rstrip("/\\")),
                                        data, probs, idx2name)
        except Exception as e:
            urls = {"skel_url": None, "probs_url": None}
            _log(task_id, f"骨架/概率保存失败: {e}")

        # 拼接结果
        all_probs = [
            {"index": int(c), "name": idx2name[c], "prob": float(probs[c])}
            for c in range(num_class)
        ]
        result = {
            "topk": top_results,
            "source": source,
            "annotated_url": annotate_url,
            "probs": all_probs,
            "skel_url": urls["skel_url"],
            "probs_url": urls["probs_url"],
            "frames": len(skels),
            "valid_detections": n_det,
            "image_size": [int(h), int(w)],
            "is_video": _is_video(source),
            "method": method,
            "method_label": method_label,
            "stgcn_top1": stgcn_top1,
            "knn_top1": knn_top1,
        }
        with _TASKS_LOCK:
            _TASKS[task_id]["status"] = "done"
            _TASKS[task_id]["result"] = result

        # 写入历史
        try:
            hist = _history_load()
            hist["tasks"].insert(0, {
                "task_id": task_id,
                "source": source,
                "started_at": time.time(),
                "top1": top_results[0]["name"],
                "top1_conf": top_results[0]["confidence"],
                "annotated_url": annotate_url,
            })
            hist["tasks"] = hist["tasks"][:50]
            _history_save(hist)
        except Exception:
            pass

    except SystemExit as e:
        with _TASKS_LOCK:
            _TASKS[task_id]["status"] = "error"
            _TASKS[task_id]["error"] = str(e)
        _log(task_id, f"错误: {e}")
    except Exception as e:
        with _TASKS_LOCK:
            _TASKS[task_id]["status"] = "error"
            _TASKS[task_id]["error"] = f"{type(e).__name__}: {e}"
        _log(task_id, f"错误: {type(e).__name__}: {e}")


def start(source: str, **kwargs) -> str:
    tid = create_task()
    th = threading.Thread(target=_run_with_lock, args=(tid, source),
                          kwargs=kwargs, daemon=True)
    th.start()
    return tid


def _run_with_lock(tid, source, **kwargs):
    with _INFERENCE_LOCK:
        run_inference_task(tid, source, **kwargs)


def cleanup_old_tasks(ttl_seconds: int = 3600):
    now = time.time()
    with _TASKS_LOCK:
        to_del = [k for k, v in _TASKS.items() if now - v["started_at"] > ttl_seconds]
        for k in to_del:
            _TASKS.pop(k, None)


def get_history() -> list:
    return _history_load().get("tasks", [])


# ============== 实时检测（300 帧累积 + ST-GCN + KNN 滑动窗口） ==============
# 流程：前端每 100~200ms POST 一帧图片过来 → 后端 YOLO 提 18 点 →
# 累加到 skel_seq（最多 300 帧）→ 累积够 300 帧时跑一次 ST-GCN + KNN，出 Top1，返回 →
# 之后每 30 帧滑动一次（用最近 300 帧做 ST-GCN + KNN 取 Top1）
# 累积阶段（< 300 帧）top1 为 None，前端显示"缓冲中 X/300"

_LIVE_BUF: dict = {
    "skel_seq": [],         # list of (3,18,1) 累积滑动窗口
    "frame_count": 0,      # 累计 POST 次数
    "last_top1": None,     # 最近一次 Top1（ST-GCN + KNN 任选）
    "last_topk": [],       # 最近完整 topk
    "yolo": None,           # YOLO lazy
    "stgcn": None,          # ST-GCN 模型 lazy
    "stgcn_path": "",
    "num_class": 14,
    "train_data": None,
    "train_labels": None,
    "train_arr_path": "",
    "current_label_path": "",
    "idx2name": None,
    "label_map_path": "",
    "last_warm_infer": -15,   # 缓冲期预估分类的时间戳（节流用）
    "last_provisional": None, # 缓冲期 KNN 预估的 top1（跨帧保留，直到下次更新/正式推理）
    "last_provisional_topk": [],  # 缓冲期 KNN 预估的 topk（跨帧保留，供框上显示 Top-2）
}
_LIVE_LOCK = threading.Lock()
# 滑动窗口：每 30 帧做一次推理
_SLIDE_STEP = 30
# 必须达到该帧数才允许推理
_MIN_BUFFER = 300
# 滑动窗口最大长度
_MAX_SEQ = 300
# 施救者判定：bbox 高宽比 (高/宽) 低于该值视为「躺姿患者」→ 不作为施救者。
# 跪/站姿施救者 bbox 通常高>宽（aspect > 1），躺姿患者 bbox 宽>高（aspect < 0.5）
_ASPECT_MIN = 0.5
# ============================================================
# 场景切换检测（硬切自愈）：拼接视频 / 换片段时，相邻帧所有关节会瞬间大幅
# 位移，真实 CPR 动作不可能在 33ms 内整体平移这么多。检测到即清空旧缓冲，
# 重新累积，避免上一段骨架污染 ST-GCN 的 300 帧时间窗口。
# ============================================================
_SCENE_JUMP_THRESH = 0.16   # 归一化坐标下，可靠关节的平均位移超过此值 → 疑似硬切
_SCENE_PER_JOINT = 0.12     # 单个关节判定为「大幅位移」的阈值
_SCENE_JOINT_FRAC = 0.60    # 至少 60% 的可靠关节同步大幅位移才算硬切（防止单关节抖动误判）
_SCENE_MIN_JOINTS = 5       # 可用于比较的可靠关节数下限
_SCENE_DEBOUNCE = 24        # 切完后前 24 帧不再检测（避免新片段起始检测抖动误触发）


def _live_setup(train_data_path: str, train_label_path: str,
                label_map_path: str, stgcn_model_path: str = "") -> str:
    """lazy 加载 YOLO + 训练模板 + ST-GCN（可选）。"""
    s = ""
    with _LIVE_LOCK:
        if _LIVE_BUF["yolo"] is None:
            from ultralytics import YOLO
            yolo_path = str(ROOT.parent / "ultralytics-main/pose/yolo26m-pose.pt")
            _LIVE_BUF["yolo"] = YOLO(yolo_path)
            s += f"YOLO 加载: {yolo_path}\n"
        if _LIVE_BUF.get("train_data") is None or _LIVE_BUF.get("train_arr_path") != train_data_path:
            _LIVE_BUF["train_data"] = np.load(train_data_path)
            _LIVE_BUF["train_arr_path"] = train_data_path
            s += f"训练模板加载: {train_data_path} ({_LIVE_BUF['train_data'].shape})\n"
        if _LIVE_BUF.get("train_labels") is None or _LIVE_BUF.get("current_label_path") != train_label_path:
            _, lbl = pickle.load(open(train_label_path, "rb"))
            _LIVE_BUF["train_labels"] = np.array(lbl, dtype=np.int64)
            _LIVE_BUF["current_label_path"] = train_label_path
        if _LIVE_BUF.get("idx2name") is None or _LIVE_BUF.get("label_map_path") != label_map_path:
            _LIVE_BUF["idx2name"] = load_label_map(label_map_path)
            _LIVE_BUF["label_map_path"] = label_map_path
        # ST-GCN 模型按需加载
        if stgcn_model_path and _LIVE_BUF.get("stgcn_path") != stgcn_model_path:
            model = load_stgcn(stgcn_model_path, num_class=len(_LIVE_BUF["idx2name"]), device="cuda:0")
            _LIVE_BUF["stgcn"] = model
            _LIVE_BUF["stgcn_path"] = stgcn_model_path
            _LIVE_BUF["num_class"] = len(_LIVE_BUF["idx2name"])
            s += f"ST-GCN 加载: {stgcn_model_path}\n"
    return s


def live_predict_image(image_bgr, train_data_path="",
                       train_label_path="", label_map_path="",
                       stgcn_model_path: str = "",
                       conf=0.25, imgsz=640, num_person=1,
                       method: str = "stgcn",
                       topk: int = 3):
    """实时推理：累积 300 帧做 ST-GCN + KNN，每 30 帧滑动更新。
    前 300 帧（帧累积阶段）top1 为 None，前端应显示"缓冲中 X/300"。

    Parameters
    ----------
    method : 'stgcn' / 'knn' / 'both'
        - 'stgcn': 满 300 帧后跑 ST-GCN softmax
        - 'knn'  : 同时跑 KNN k=1，取 KNN Top1
        - 'both' : 同时跑，取置信度高者
    其余路径：模板 / 标签 / ST-GCN 模型
    """
    # 路径 fallback
    if not train_data_path:
        train_data_path = str(ROOT / "data/my_dataset/out/train_data.npy")
    if not train_label_path:
        train_label_path = str(ROOT / "data/my_dataset/out/train_label.pkl")
    if not label_map_path:
        label_map_path = str(ROOT / "data/my_dataset/out/label_map.json")
    if not stgcn_model_path:
        stgcn_model_path = str(ROOT / "work_dir/recognition/my_dataset/ST_GCN/epoch30_model.pt")
    _live_setup(train_data_path, train_label_path, label_map_path, stgcn_model_path)

    with _LIVE_LOCK:
        yolo = _LIVE_BUF["yolo"]
        train_arr = _LIVE_BUF["train_data"]
        train_lbls = _LIVE_BUF["train_labels"]
        idx2name = _LIVE_BUF["idx2name"]

    # YOLO 单帧推理
    h, w = image_bgr.shape[:2]
    res = yolo.predict(source=image_bgr, conf=conf, imgsz=imgsz, device="0", verbose=False)
    kpts_obj = res[0].keypoints

    # 构造单帧骨架 (3, 18, M)
    skel = np.zeros((3, 18, num_person), dtype=np.float32)
    people_kpts = []
    box_norm = None
    if kpts_obj is not None and kpts_obj.shape[0] > 0:
        xy = kpts_obj.xy.cpu().numpy()
        confs = kpts_obj.conf.cpu().numpy()
        xy[:, :, 0] /= w
        xy[:, :, 1] /= h
        # 只保留施救者：CPR 场景排除「躺在地上」的患者（bbox 宽扁 + 靠下）
        boxes = None
        if res[0].boxes is not None:
            boxes = res[0].boxes.xyxy.cpu().numpy()
        person_idx = None
        chosen_box = None
        if boxes is not None and len(boxes) == confs.shape[0] and len(boxes) > 0:
            keep = select_cpr_person(boxes, w, h)
            if keep:
                p = int(keep[0])
                bw = float(boxes[p, 2] - boxes[p, 0])
                bh = float(boxes[p, 3] - boxes[p, 1])
                aspect = bh / max(bw, 1e-6)
                # 保护：施救者 bbox 应明显「竖」（高 > 宽）；只有躺姿患者时视为无人
                if aspect >= _ASPECT_MIN:
                    person_idx = p
                    chosen_box = boxes[p]
        if person_idx is not None:
            pose, sc = coco17_to_openpose18(xy[person_idx], confs[person_idx])
            skel[0, :, 0] = pose[:, 0]
            skel[1, :, 0] = pose[:, 1]
            skel[2, :, 0] = sc
            people_kpts.append([
                {"x": float(pose[j, 0]), "y": float(pose[j, 1]),
                 "score": float(sc[j]), "name": OPENPOSE_18_NAMES[j]}
                for j in range(18)
            ])
            if chosen_box is not None:
                box_norm = {
                    "x1": float(chosen_box[0]) / w,
                    "y1": float(chosen_box[1]) / h,
                    "x2": float(chosen_box[2]) / w,
                    "y2": float(chosen_box[3]) / h,
                }
        # person_idx 为 None（无人 / 只有躺姿患者）→ kpts 空、box None、skel 全 0

    # 累积骨架到滑动窗口 + 触发推理
    top1 = None
    topk_list = []
    phase = "warming"
    frame_count_after = 0
    buffer_len = 0
    last_inference_frame = -1
    last_top1 = None
    provisional_flag = False    # 缓冲期 KNN 预估（非 300 帧正式推理）

    with _LIVE_LOCK:
        # 场景切换检测：硬切会让所有关节在相邻帧间瞬间大幅位移 → 清空旧缓冲
        # （必须在 append 之前比较上一帧，检测成功后缓冲已清空，本帧作为新片段首帧保留）
        _maybe_scene_reset(_LIVE_BUF, skel)
        _LIVE_BUF["skel_seq"].append(skel.copy())
        if len(_LIVE_BUF["skel_seq"]) > _MAX_SEQ:
            # 滑动窗口
            _LIVE_BUF["skel_seq"] = _LIVE_BUF["skel_seq"][-_MAX_SEQ:]
        _LIVE_BUF["frame_count"] += 1
        frame_count_after = _LIVE_BUF["frame_count"]
        buffer_len = len(_LIVE_BUF["skel_seq"])
        last_top1 = _LIVE_BUF.get("last_top1")
        last_inference_frame = _LIVE_BUF.get("last_inference_frame", -1)

        ready = buffer_len >= _MIN_BUFFER
        if ready:
            phase = "ready"
            # 每 _SLIDE_STEP 帧触发一次推理（或第一次）
            should_infer = (frame_count_after - last_inference_frame >= _SLIDE_STEP) or (last_inference_frame < 0)
            if should_infer:
                try:
                    # 拼成 (3, T, 18, 1)，与 train (N, 3, 300, 18, 1) 对齐
                    seq = np.stack(_LIVE_BUF["skel_seq"], axis=0)        # (T, 3, 18, 1)
                    seq = seq.transpose(1, 0, 2, 3)                      # (3, T, 18, 1)
                    if seq.shape[1] != 300:
                        # 重采样到 300 帧
                        Ts = seq.shape[1]
                        idx = np.linspace(0, Ts - 1, 300).round().astype(int)
                        seq = seq[:, idx]
                    # 中心化（与离线一致）
                    seq_proc = seq.copy()
                    seq_proc[0:2] -= 0.5
                    seq_proc[0][seq_proc[2] == 0] = 0.0
                    seq_proc[1][seq_proc[2] == 0] = 0.0

                    method_chosen = method
                    stgcn_probs = None
                    knn_probs = None
                    if method in ("stgcn", "both"):
                        stgcn = _LIVE_BUF["stgcn"]
                        x = torch.from_numpy(seq_proc).float().unsqueeze(0)
                        if torch.cuda.is_available(): x = x.cuda()
                        with torch.no_grad():
                            out = stgcn(x)
                        stgcn_probs = torch.softmax(out, dim=1)[0].cpu().numpy()
                    if method in ("knn", "both"):
                        from knn_match import knn_predict
                        _, knn_dist, knn_lab = knn_predict(seq_proc, train_arr, train_lbls, k=1)
                        # KNN 1-NN：构造按距离倒数的全类别概率
                        knn_probs = np.zeros(len(idx2name), dtype=np.float32)
                        knn_probs[int(knn_lab[0])] = 1.0

                    if method == "stgcn" and stgcn_probs is not None:
                        probs = stgcn_probs
                    elif method == "knn" and knn_probs is not None:
                        probs = knn_probs
                    elif method == "both" and stgcn_probs is not None:
                        probs = stgcn_probs
                    else:
                        probs = stgcn_probs if stgcn_probs is not None else knn_probs

                    # top-K
                    if probs is not None:
                        order = probs.argsort()[::-1]
                        for r, c in enumerate(order[:topk]):
                            topk_list.append({
                                "rank": int(r + 1),
                                "label": int(c),
                                "name": idx2name[int(c)],
                                "confidence": float(probs[c]),
                            })
                        if topk_list:
                            top1 = topk_list[0]
                            _LIVE_BUF["last_top1"] = top1
                            _LIVE_BUF["last_topk"] = topk_list
                except Exception as e:
                    # 推理失败兜底：保持 last_top1 不变（不丢），并打印 traceback 便于排查
                    import traceback as _tb
                    print(f"[live] sliding infer failed: {type(e).__name__}: {e}",
                          file=sys.stderr)
                    _tb.print_exc(file=sys.stderr)
                _LIVE_BUF["last_inference_frame"] = frame_count_after
            # 未触发推理的帧（每 30 帧才推一次）：回填上次结果，避免框上类别闪回「施救者」
            top1 = last_top1 or top1
            topk_list = topk_list or _LIVE_BUF.get("last_topk", [])
        else:
            # 没累积够 300 帧：top1 优先用正式推理结果，否则用缓冲期 KNN 预估（跨帧保留）
            top1 = last_top1 or _LIVE_BUF.get("last_provisional")
            topk_list = _LIVE_BUF.get("last_topk", []) or \
                _LIVE_BUF.get("last_provisional_topk", []) or \
                ([top1] if top1 else [])
            # 缓冲期预估：累积 >=30 帧且距上次预估 >=15 帧时，用 KNN 对当前部分
            # 缓冲（重采样到 300）做初步分类，让框上提前显示动作类别（标"估"）
            if last_top1 is None and buffer_len >= 30:
                if frame_count_after - _LIVE_BUF.get("last_warm_infer", -15) >= 15:
                    try:
                        from knn_match import knn_predict
                        seq = np.stack(_LIVE_BUF["skel_seq"], axis=0)   # (T,3,18,1)
                        seq = seq.transpose(1, 0, 2, 3)                 # (3,T,18,1)
                        Ts = seq.shape[1]
                        ridx = np.linspace(0, Ts - 1, 300).round().astype(int)
                        seq_r = seq[:, ridx]
                        seq_proc = seq_r.copy()
                        seq_proc[0:2] -= 0.5
                        seq_proc[0][seq_proc[2] == 0] = 0.0
                        seq_proc[1][seq_proc[2] == 0] = 0.0
                        # k=5 邻居 + 距离倒数加权投票 → 构造 Top-2 类别
                        _, knn_dist, knn_lab = knn_predict(
                            seq_proc, train_arr, train_lbls, k=5)
                        weights = 1.0 / (1.0 + np.asarray(knn_dist, dtype=np.float32))
                        scores = np.zeros(len(idx2name), dtype=np.float32)
                        for lab, wt in zip(knn_lab.tolist(), weights.tolist()):
                            scores[int(lab)] += wt
                        s_sum = float(scores.sum()) or 1.0
                        order = scores.argsort()[::-1]
                        topk_list = []
                        for r, c in enumerate(order[:2]):
                            conf = float(scores[int(c)]) / s_sum
                            if conf <= 0:
                                break
                            topk_list.append({
                                "rank": int(r + 1),
                                "label": int(c),
                                "name": idx2name[int(c)],
                                "confidence": conf,
                            })
                        if topk_list:
                            top1 = topk_list[0]
                            provisional_flag = True
                            _LIVE_BUF["last_warm_infer"] = frame_count_after
                            _LIVE_BUF["last_provisional"] = top1
                            _LIVE_BUF["last_provisional_topk"] = topk_list
                    except Exception:
                        pass

    return {
        "kpts": people_kpts,
        "box": box_norm,                                      # 施救者 bbox（归一化，无人时为 None）
        "top1": top1,
        "topk": topk_list,
        "phase": phase,                          # "warming" / "ready"
        "buffer_len": buffer_len,                # 当前缓冲长度
        "buffer_need": _MIN_BUFFER,              # 需要累积的帧数
        "frame_count": frame_count_after,
        "slide_step": _SLIDE_STEP,
        "next_infer_in": max(0, _SLIDE_STEP - (frame_count_after - last_inference_frame)) if phase == "ready" and last_inference_frame >= 0 else None,
        "image_size": [int(h), int(w)],
        "method": method,
        "provisional": provisional_flag,     # True=缓冲期 KNN 预估（非正式 300 帧推理）
    }


def _reset_buffer_state(buf):
    """清空实时骨架缓冲与推理状态（保留已加载的模型/模板）。"""
    buf["skel_seq"] = []
    buf["frame_count"] = 0
    buf["last_top1"] = None
    buf["last_topk"] = []
    buf["last_inference_frame"] = -1
    buf["last_warm_infer"] = -15
    buf["last_provisional"] = None
    buf["last_provisional_topk"] = []


def _maybe_scene_reset(buf, cur_skel):
    """检测硬切：相邻帧整体骨架瞬间大幅位移 → 清空旧缓冲，重新累积。

    返回 True 表示已触发重置。需在持有 _LIVE_LOCK 时调用（buf 即 _LIVE_BUF）。
    """
    seq = buf["skel_seq"]
    if len(seq) < 2:
        return False
    # 切完后去抖窗口：前 _SCENE_DEBOUNCE 帧不检测（避免新片段起始抖动误判）
    if buf["frame_count"] < _SCENE_DEBOUNCE:
        return False
    prev = seq[-1]
    pc = prev[2].reshape(-1)
    cc = cur_skel[2].reshape(-1)
    mask = (pc > 0.3) & (cc > 0.3)            # 两帧都可靠的关节
    if mask.sum() < _SCENE_MIN_JOINTS:
        return False
    px, py = prev[0].reshape(-1), prev[1].reshape(-1)
    cx, cy = cur_skel[0].reshape(-1), cur_skel[1].reshape(-1)
    dists = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
    valid = dists[mask]
    mean_d = float(valid.mean())
    frac_big = float((valid > _SCENE_PER_JOINT).mean())
    if mean_d > _SCENE_JUMP_THRESH and frac_big > _SCENE_JOINT_FRAC:
        _reset_buffer_state(buf)
        print(f"[live] 检测到场景切换 (mean_d={mean_d:.3f}, "
              f"frac_big={frac_big:.2f})，已重置缓冲重新累积", file=sys.stderr)
        return True
    return False


def live_reset():
    """清空实时 buffer，重新开始累积。"""
    with _LIVE_LOCK:
        _reset_buffer_state(_LIVE_BUF)
