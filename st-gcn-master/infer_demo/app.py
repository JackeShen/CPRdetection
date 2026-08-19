"""YOLO+ST-GCN 可视化 Web 推理界面。

启动（st-gcn-master 目录下、oldshen 环境）：
    python infer_demo/app.py
默认监听 0.0.0.0:5000，浏览器打开 http://127.0.0.1:5000
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from flask import Flask, request, jsonify, render_template, send_from_directory, make_response

import infer_service

ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

DEFAULTS = {
    "stgcn_model": str(ROOT / "work_dir/recognition/my_dataset/ST_GCN/epoch30_model.pt"),
    "yolo_model":  str(ROOT.parent / "ultralytics-main/pose/yolo26m-pose.pt"),
    "label_map":   str(ROOT / "data/my_dataset/out/label_map.json"),
    "train_data":  str(ROOT / "data/my_dataset/out/train_data.npy"),
    "train_label": str(ROOT / "data/my_dataset/out/train_label.pkl"),
    "device":      "auto",
    "num_person":  1,
    "select":      "cpr",
    "max_frame":   300,
    "imgsz":       640,
    "conf":        0.25,
    "topk":        3,
    "method":      "knn",   # 默认 KNN k=1（val 集 50%，比 ST-GCN 高 14%）
    "knn_k":       1,
}

# OpenPose 18 关节顺序（与 graph.py 一致）
EDGE_LINKS = getattr(infer_service, "OPENPOSE_18_EDGES", None)
if EDGE_LINKS is None:
    EDGE_LINKS = [(0, 1), (0, 14), (0, 15), (14, 16), (15, 17),
                  (1, 2), (2, 3), (3, 4),
                  (1, 5), (5, 6), (6, 7),
                  (1, 8), (8, 9), (9, 10),
                  (1, 11), (11, 12), (12, 13)]

JOINT_NAMES = getattr(infer_service, "OPENPOSE_18_NAMES", [
    "nose", "neck", "r_shoulder", "r_elbow", "r_wrist",
    "l_shoulder", "l_elbow", "l_wrist",
    "r_hip", "r_knee", "r_ankle",
    "l_hip", "l_knee", "l_ankle",
    "r_eye", "l_eye", "r_ear", "l_ear",
])

app = Flask(__name__,
            template_folder=str(Path(__file__).parent / "templates"),
            static_folder=str(Path(__file__).parent / "static"))


# 强制每次都从服务器拿，避免开发时缓存问题
@app.after_request
def no_cache(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.route("/")
def index():
    return _with_no_cache(make_response(render_template("index.html", defaults=DEFAULTS)))


@app.route("/live")
def live_page():
    return _with_no_cache(make_response(render_template("live.html", defaults=DEFAULTS)))


@app.route("/vlm")
def vlm_page():
    return _with_no_cache(make_response(render_template("vlm.html", defaults=DEFAULTS)))


def _with_no_cache(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.route("/api/run", methods=["POST"])
def api_run():
    form = request.form
    file = request.files.get("file")
    if file and file.filename:
        save_path = UPLOAD_DIR / file.filename
        file.save(save_path)
        source = str(save_path)
    else:
        source = (form.get("source_path") or "").strip()
        if not source:
            return jsonify({"error": "请选择文件 或 输入服务器路径"}), 400
        source = os.path.normpath(source)

    cfg = dict(DEFAULTS)
    for k in ("device", "num_person", "select", "max_frame", "imgsz", "conf", "topk", "knn_k"):
        if k in form and form.get(k) not in (None, ""):
            try:
                cfg[k] = type(DEFAULTS[k])(form.get(k))
            except (TypeError, ValueError):
                pass
    for k in ("stgcn_model", "yolo_model", "label_map", "train_data", "train_label", "method"):
        if form.get(k):
            cfg[k] = form.get(k).strip()

    tid = infer_service.start(source, **cfg)
    return jsonify({"task_id": tid, "source": source})


@app.route("/api/poll/<tid>")
def api_poll(tid):
    t = infer_service.get_task(tid)
    if t is None:
        return jsonify({"error": "task not found"}), 404
    return jsonify({
        "status": t["status"],
        "log": list(t["log"]),
        "result": t.get("result"),
        "error": t.get("error"),
        "started_at": t["started_at"],
    })


@app.route("/api/history")
def api_history():
    return jsonify({"tasks": infer_service.get_history()})


@app.route("/api/live_predict", methods=["POST"])
def api_live_predict():
    """实时检测：上传一帧，返回关键点 + 当前 KNN Top1。"""
    file = request.files.get("frame")
    if not file or not file.filename:
        return jsonify({"error": "missing 'frame' file"}), 400
    import numpy as _np
    raw = file.read()
    img = _np.frombuffer(raw, dtype=_np.uint8)
    import cv2
    img_bgr = cv2.imdecode(img, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({"error": "decode failed"}), 400

    # 关键参数（可从前端 form 覆盖）
    cfg = dict(DEFAULTS)
    for k in ("imgsz", "conf", "topk"):
        if k in request.form and request.form.get(k) not in (None, ""):
            try: cfg[k] = type(DEFAULTS[k])(request.form.get(k))
            except (TypeError, ValueError): pass
    for k in ("train_data", "train_label"):
        v = request.form.get(k)
        if v: cfg[k] = v.strip()

    result = infer_service.live_predict_image(
        img_bgr,
        train_data_path=cfg["train_data"],
        train_label_path=cfg["train_label"],
        label_map_path=cfg["label_map"],
        conf=cfg["conf"],
        imgsz=cfg["imgsz"],
        topk=cfg["topk"],
    )
    return jsonify(result)


@app.route("/api/live_reset", methods=["POST"])
def api_live_reset():
    """重置实时累积 buffer。"""
    infer_service.live_reset()
    return jsonify({"status": "reset"})


@app.route("/api/vlm_classify", methods=["POST"])
def api_vlm_classify():
    """VLM 图片分类：上传一张图片，返回 qwen3-vl:2b 的 9 类分类结果。"""
    file = request.files.get("image")
    if not file or not file.filename:
        return jsonify({"error": "missing 'image' file"}), 400
    image_bytes = file.read()
    if not image_bytes:
        return jsonify({"error": "image is empty"}), 400
    import vlm_service
    result = vlm_service.classify_image_bytes(image_bytes)
    return jsonify(result)


@app.route("/class_names.json")
def class_names():
    return send_from_directory(str(ROOT / "data/my_dataset/out"), "class_names.json")


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)


# 后台清理
def _cleanup_loop():
    while True:
        time.sleep(600)
        try:
            infer_service.cleanup_old_tasks()
        except Exception:
            pass


threading.Thread(target=_cleanup_loop, daemon=True).start()

# 也提供给前端：边的连接 + 关节名
@app.route("/api/skeleton_meta")
def api_skeleton_meta():
    return jsonify({
        "layout": "openpose",
        "joints": JOINT_NAMES,
        "edges": EDGE_LINKS,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
