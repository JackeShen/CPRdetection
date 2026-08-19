"""批量把目录下的 4K / 不兼容的 mp4 转码成浏览器友好的 1080p H.264 Baseline。

用法：
    python convert_demovideos.py <dir> [--delete-src]

会扫描目录下所有 .mp4，对每个文件：
- 如果分辨率 <= 1080p：跳过
- 否则：用 ffmpeg 转成 1920x1080 H.264 Baseline Level 3.1 + AAC audio
- 输出文件名：原名 + "_1080p.mp4"
- 加 --delete-src 时会删除原文件

依赖：imageio-ffmpeg（自带 ffmpeg.exe，无需系统装）
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def get_ffmpeg():
    """优先用 imageio-ffmpeg 自带的 ffmpeg，回退到系统 ffmpeg。"""
    try:
        import imageio_ffmpeg
        p = imageio_ffmpeg.get_ffmpeg_exe()
        if os.path.isfile(p):
            return p
    except ImportError:
        pass
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg
    raise RuntimeError("未找到 ffmpeg，请安装 imageio-ffmpeg（pip install imageio-ffmpeg）")


def probe_resolution(ffmpeg, mp4_path):
    """用 ffprobe-like 解析（直接调 ffmpeg -i）。"""
    r = subprocess.run([ffmpeg, "-i", str(mp4_path)], capture_output=True, text=True)
    # 解析 "Stream #0:0 ..." 行里有 "3840x2160"
    for line in (r.stdout + r.stderr).splitlines():
        if "Stream" in line and "Video" in line:
            import re
            m = re.search(r"(\d{2,5})x(\d{2,5})", line)
            if m:
                return int(m.group(1)), int(m.group(2))
    return None, None


def convert_one(ffmpeg, src, dst):
    cmd = [
        ffmpeg, "-y", "-i", str(src),
        "-vf", "scale=1920:1080",
        "-c:v", "libx264",
        "-profile:v", "baseline",
        "-level", "3.1",
        "-pix_fmt", "yuv420p",
        "-crf", "23",
        "-preset", "medium",
        "-c:a", "aac",
        "-movflags", "+faststart",
        str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0, r.stderr[-500:] if r.returncode != 0 else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="目录路径，扫描所有 .mp4")
    ap.add_argument("--delete-src", action="store_true",
                    help="转码成功后删除原文件")
    ap.add_argument("--force", action="store_true",
                    help="即使源已经 ≤1080p 也强制转码")
    args = ap.parse_args()

    root = Path(args.dir)
    if not root.is_dir():
        print(f"❌ {root} 不是目录")
        sys.exit(1)
    ffmpeg = get_ffmpeg()
    print(f"使用 ffmpeg: {ffmpeg}\n")

    mp4s = sorted(root.glob("*.mp4"))
    print(f"扫描到 {len(mp4s)} 个 mp4，开始处理：\n")

    converted, skipped, failed = [], [], []
    for src in mp4s:
        name = src.name
        if "_1080p" in src.stem or "_web" in src.stem:
            print(f"⏭ 跳过（已转码）: {name}")
            continue
        w, h = probe_resolution(ffmpeg, src)
        if w is None:
            print(f"⚠️ 无法探测分辨率，跳过: {name}")
            continue
        if (w <= 1920 and h <= 1080) and not args.force:
            print(f"⏭ {name}: {w}x{h} 已经 ≤1080p，跳过")
            skipped.append(name)
            continue

        dst = src.parent / (src.stem + "_1080p.mp4")
        if dst.exists():
            print(f"⏭ {name}: 输出已存在 ({dst.name})，跳过")
            skipped.append(name)
            continue

        print(f"🔄 {name}: {w}x{h} → 1920x1080 H.264 baseline ...")
        ok, err = convert_one(ffmpeg, src, dst)
        if ok:
            sz_src = src.stat().st_size / 1024 / 1024
            sz_dst = dst.stat().st_size / 1024 / 1024
            print(f"   ✅ 完成: {dst.name} ({sz_dst:.1f} MB, 源 {sz_src:.1f} MB)")
            converted.append((src, dst))
            if args.delete_src:
                src.unlink()
                print(f"   🗑 已删除原文件: {name}")
        else:
            print(f"   ❌ 失败: {err[:200]}")
            failed.append(name)

    print(f"\n=== 总结 ===")
    print(f"转换: {len(converted)}")
    print(f"跳过: {len(skipped)}")
    print(f"失败: {len(failed)}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
