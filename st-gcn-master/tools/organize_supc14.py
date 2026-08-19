"""
把 SupC14_Dataset 整理成 ST-GCN 用的 mydata 格式（图片序列目录）。

原始结构: SupC14_Dataset/SupCxx/rYY/chZ(+ofchZ)/一大堆 4K jpg
  - SupCxx : 14 个动作类别 (SupC00..SupC13)
  - rYY    : 每个动作 2 次重复 (r00, r01)
  - chZ    : 每个重复 4 个 RGB 视角 (ch0..ch3)，另有 ofchZ 是光流(本任务用不到)

输出结构: mydata/SupCxx/SupCxx_rYY_chZ/ (1080p jpg 图片序列)
  - 类别 = mydata 下的直接子目录名 (SupCxx)
  - 每个样本 = 一个图片序列目录
  - 只取 RGB (ch0..ch3)，排除光流 (ofch*)

用法:
    python tools/organize_supc14.py
"""
from __future__ import annotations

import os
import glob

import cv2

SRC = r"E:/BaiduNetdiskDownload/SupC14_Dataset"
DST = r"C:/Users/11137/Desktop/shixi/CPROldshen/st-gcn-master/mydata"

CATEGORIES = [f"SupC{i:02d}" for i in range(14)]   # SupC00 .. SupC13
REPS = ["r00", "r01"]
VIEWS = ["ch0", "ch1", "ch2", "ch3"]               # 仅 RGB，排除 ofch 光流
OUT_W, OUT_H = 1920, 1080                          # 降到 1080p
JPEG_QUALITY = 85


def organize() -> int:
    done = 0
    for cat in CATEGORIES:
        for rep in REPS:
            for view in VIEWS:
                src = os.path.join(SRC, cat, rep, view)
                if not os.path.isdir(src):
                    print(f"[跳过] 不存在: {src}")
                    continue
                fns = sorted(glob.glob(os.path.join(src, "*.jpg")))
                if not fns:
                    print(f"[跳过] 无 jpg: {src}")
                    continue

                out_dir = os.path.join(DST, cat, f"{cat}_{rep}_{view}")
                os.makedirs(out_dir, exist_ok=True)

                for i, fn in enumerate(fns):
                    im = cv2.imread(fn)
                    if im is None:
                        continue
                    im = cv2.resize(im, (OUT_W, OUT_H))
                    cv2.imwrite(
                        os.path.join(out_dir, f"{i:06d}.jpg"),
                        im,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
                    )

                done += 1
                print(f"[完成] {cat}_{rep}_{view}: {len(fns)} 帧 -> {out_dir}")
    return done


if __name__ == "__main__":
    total = organize()
    print(f"\n全部完成，共整理 {total} 个样本序列 (14类 x 2重复 x 4视角)。")
    print(f"输出目录: {DST}")
