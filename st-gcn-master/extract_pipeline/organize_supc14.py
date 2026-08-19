"""
把服务器上的原始 SupC14_Dataset 整理成 ST-GCN 用的 mydata 格式（图片序列目录）。

原始结构: SupC14_Dataset/SupCxx/rYY/chZ(+ofchZ)/一大堆 4K jpg
  - SupCxx : 14 个动作类别 (SupC00..SupC13)
  - rYY    : 每个动作 2 次重复 (r00, r01)
  - chZ    : 每个重复 4 个 RGB 视角 (ch0..ch3)，另有 ofchZ 是光流(本任务用不到)
  - 图片是 4K (3840x2160) jpg

输出结构: mydata/SupCxx/SupCxx_rYY_chZ/ (1080p jpg 图片序列)
  - 类别 = mydata 下的直接子目录名 (SupCxx)
  - 每个样本 = 一个图片序列目录
  - 只取 RGB (ch0..ch3)，排除光流 (ofch*)

用法（服务器）:
    python organize_supc14.py \
        --src /data/SupC14_Dataset \
        --dst ./mydata \
        --out_w 1920 --out_h 1080 --quality 85
"""
from __future__ import annotations

import argparse
import glob
import os

import cv2


def organize(src: str, dst: str, out_w: int, out_h: int, quality: int) -> int:
    categories = [f"SupC{i:02d}" for i in range(14)]   # SupC00 .. SupC13
    reps = ["r00", "r01"]
    views = ["ch0", "ch1", "ch2", "ch3"]                # 仅 RGB，排除 ofch 光流

    done = 0
    for cat in categories:
        for rep in reps:
            for view in views:
                src_dir = os.path.join(src, cat, rep, view)
                if not os.path.isdir(src_dir):
                    print(f"[跳过] 不存在: {src_dir}")
                    continue
                fns = sorted(glob.glob(os.path.join(src_dir, "*.jpg")))
                if not fns:
                    print(f"[跳过] 无 jpg: {src_dir}")
                    continue

                out_dir = os.path.join(dst, cat, f"{cat}_{rep}_{view}")
                os.makedirs(out_dir, exist_ok=True)

                for i, fn in enumerate(fns):
                    im = cv2.imread(fn)
                    if im is None:
                        continue
                    im = cv2.resize(im, (out_w, out_h))
                    cv2.imwrite(
                        os.path.join(out_dir, f"{i:06d}.jpg"),
                        im,
                        [cv2.IMWRITE_JPEG_QUALITY, quality],
                    )

                done += 1
                print(f"[完成] {cat}_{rep}_{view}: {len(fns)} 帧 -> {out_dir}")
    return done


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="整理 SupC14 原始数据为 mydata 图片序列格式")
    p.add_argument("--src", required=True, help="原始 SupC14_Dataset 根目录")
    p.add_argument("--dst", required=True, help="输出 mydata 根目录")
    p.add_argument("--out_w", type=int, default=1920, help="输出宽度 (默认 1920)")
    p.add_argument("--out_h", type=int, default=1080, help="输出高度 (默认 1080)")
    p.add_argument("--quality", type=int, default=85, help="jpg 质量 (默认 85)")
    args = p.parse_args()

    total = organize(args.src, args.dst, args.out_w, args.out_h, args.quality)
    print(f"\n全部完成，共整理 {total} 个样本序列 (14类 x 2重复 x 4视角)。")
    print(f"输出目录: {args.dst}")
