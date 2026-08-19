"""把多个 demo 视频拼接成一个 1080p 网页可播放视频。

用法：
    python combine_demo_videos.py <输出名.mp4> <输入1.mp4> <输入2.mp4> ...

输出参数与现有 _1080p 一致：H.264 Baseline Level 3.1, yuv420p(限幅), 1920x1080, 30fps, faststart。
用 concat 滤波器拼接后统一重编码，避免各片段色彩范围/参数微小差异导致的问题。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

FFMPEG = r"C:\Users\11137\miniconda3\envs\oldshen\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe"


def combine(out_path: str, inputs: list[str]) -> None:
    if len(inputs) < 2:
        raise SystemExit("至少需要 2 个输入视频")
    for p in inputs:
        if not Path(p).is_file():
            raise SystemExit(f"输入不存在: {p}")

    n = len(inputs)
    in_args = []
    for p in inputs:
        in_args += ["-i", p]

    # 拼接所有视频流，再统一缩放/像素格式/色彩范围
    fc = (
        "".join(f"[{i}:v]" for i in range(n))
        + f"concat=n={n}:v=1:a=0[v];"
        + "[v]scale=1920:1080,format=yuv420p,setrange=limited[v2]"
    )

    cmd = [
        FFMPEG, "-y",
        *in_args,
        "-filter_complex", fc,
        "-map", "[v2]",
        "-c:v", "libx264",
        "-profile:v", "baseline",
        "-level", "3.1",
        "-pix_fmt", "yuv420p",
        "-color_range", "1",
        "-crf", "23",
        "-preset", "medium",
        "-movflags", "+faststart",
        out_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("[FAIL]\n" + r.stderr[-1200:])
        raise SystemExit(1)
    sz = Path(out_path).stat().st_size / 1024 / 1024
    print(f"[OK] {out_path}  ({sz:.1f} MB)")


def main() -> None:
    if len(sys.argv) < 4:
        print("用法: python combine_demo_videos.py <输出.mp4> <输入1> <输入2> ...")
        raise SystemExit(1)
    out = sys.argv[1]
    ins = sys.argv[2:]
    print(f"拼接 {len(ins)} 个视频 → {out}\n")
    combine(out, ins)


if __name__ == "__main__":
    main()
