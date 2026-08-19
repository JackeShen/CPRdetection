"""从 mydata 图片序列生成 1080p 网页可播放的 demo 视频。

与 convert_demovideos.py 的输出参数保持一致：
    1920x1080, H.264 Constrained Baseline Level 3.1, yuv420p, faststart, 30fps

只为 demo_videos 里还没有 `_1080p` 的类别生成（已有 SupC04/05/07/08/10）。

用法：
    python generate_demo_videos_from_mydata.py
"""
from __future__ import annotations

import subprocess
from pathlib import Path

# imageio-ffmpeg 自带 ffmpeg（已确认支持 libx264，现有 _1080p 即由此生成）
FFMPEG = r"C:\Users\11137\miniconda3\envs\oldshen\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe"

MYDATA = Path(r"C:\Users\11137\Desktop\shixi\CPROldshen\st-gcn-master\mydata")
OUT = Path(r"C:\Users\11137\Desktop\shixi\CPR-Coach-main\demo_videos")

# demo_videos 还没有 _1080p 的类别 → 规范中文名（来自 data/my_dataset/out/class_names.json）
CATS = {
    "SupC00": "正确动作",
    "SupC01": "双手重叠",
    "SupC02": "握拳按压",
    "SupC03": "单手按压",
    "SupC06": "跳跃按压",
    "SupC09": "位置偏移",
    "SupC11": "频率过慢",
    "SupC12": "按压过度",
    "SupC13": "随机位置按压",
}

# 每个类别选 r00_ch0 这一路视角（所有视角帧数一致，ch0 即可），≥350 帧
REP_VIEW = "r00_ch0"


def generate(cat: str, name: str) -> None:
    src_dir = MYDATA / cat / f"{cat}_{REP_VIEW}"
    dst = OUT / f"{cat}_{name}_1080p.mp4"

    if not src_dir.is_dir():
        print(f"[跳过] 源序列不存在: {src_dir}")
        return
    n_frames = len(list(src_dir.glob("*.jpg")))
    if n_frames < 350:
        print(f"[警告] {cat} 仅 {n_frames} 帧 (<350)，仍生成: {dst.name}")

    # 若目标已存在则直接覆盖（ffmpeg -y），便于像素格式修正后重生成
    cmd = [
        FFMPEG, "-y",
        "-framerate", "30",
        "-start_number", "0",
        "-i", str(src_dir / "%06d.jpg"),
        "-vf", "scale=1920:1080,format=yuv420p,setrange=limited",
        "-c:v", "libx264",
        "-profile:v", "baseline",
        "-level", "3.1",
        "-pix_fmt", "yuv420p",
        "-color_range", "1",
        "-crf", "23",
        "-preset", "medium",
        "-movflags", "+faststart",
        str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        sz = dst.stat().st_size / 1024 / 1024
        print(f"[OK]   {dst.name}  ({sz:.1f} MB, 源 {n_frames} 帧)")
    else:
        print(f"[FAIL] {dst.name}\n{r.stderr[-1000:]}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"ffmpeg: {FFMPEG}")
    print(f"源:     {MYDATA}")
    print(f"输出:   {OUT}")
    print(f"待生成 {len(CATS)} 个类别：{', '.join(CATS.keys())}\n")
    for cat, name in CATS.items():
        generate(cat, name)
    print("\n=== 全部完成 ===")


if __name__ == "__main__":
    main()
