#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CPR 动作 VLM 分类器 —— 命令行独立版本

用法:
    python cpr_vlm_classify.py <图片路径>
    python cpr_vlm_classify.py C:\\path\\to\\cpr_frame.jpg

功能:
    选择一张 CPR 按压图片 → Ollama qwen3-vl:2b 分析 → 输出 9 类分类结果

依赖:
    pip install ollama
    ollama serve  (确保 Ollama 服务在跑)
    ollama pull qwen3-vl:2b

注意:
    本脚本可独立运行，也可通过 Web UI 访问 (http://127.0.0.1:5000/vlm)
    核心逻辑复用 st-gcn-master/infer_demo/vlm_service.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# 尝试导入 infer_demo 下的 vlm_service；如果不在路径里则内联
def _import_vlm_service():
    """优先从 infer_demo 导入，找不到则用内联逻辑。"""
    infer_demo = Path(__file__).resolve().parent.parent / "st-gcn-master" / "infer_demo"
    if infer_demo.exists():
        sys.path.insert(0, str(infer_demo))
        try:
            import vlm_service
            return vlm_service
        except ImportError:
            pass
    return None


def main():
    if len(sys.argv) < 2:
        print("=" * 55)
        print("  CPR 动作 VLM 分类器 (Ollama qwen3-vl:2b)")
        print("=" * 55)
        print()
        print("用法:")
        print("  python cpr_vlm_classify.py <图片路径>")
        print()
        print("示例:")
        print('  python cpr_vlm_classify.py C:\\demo\\cpr_frame.jpg')
        print()
        print("9 类分类标准:")
        classes = [
            "0. 正确动作", "1. 双手重叠", "2. 握拳按压", "3. 单手按压",
            "4. 手臂弯曲", "5. 手臂倾斜", "6. 蹲姿按压", "7. 站立按压",
            "8. 位置偏移",
        ]
        for c in classes:
            print(f"    {c}")
        print()
        print("前提:")
        print("  1. ollama serve  (Ollama 服务在跑)")
        print("  2. ollama pull qwen3-vl:2b  (模型已拉取)")
        print("  3. pip install ollama  (Python 包已安装)")
        sys.exit(0)

    image_path = sys.argv[1]

    if not Path(image_path).exists():
        print(f"错误: 文件不存在 → {image_path}")
        sys.exit(1)

    # 导入核心模块
    vlm = _import_vlm_service()
    if vlm is None:
        print("错误: 找不到 vlm_service.py，请确认项目结构完整")
        print("      期望路径: st-gcn-master/infer_demo/vlm_service.py")
        sys.exit(1)

    print(f"模型: {vlm.MODEL}")
    print(f"图片: {image_path}")
    print(f"Ollama: {vlm.OLLAMA_HOST}")
    print()
    print("正在分析（qwen3-vl:2b 推理中，请稍候）...")
    print()

    result = vlm.classify_image_file(image_path)

    if result.get("error"):
        print(f"错误: {result['error']}")
        sys.exit(1)

    # 格式化输出
    print("=" * 55)
    print("  VLM 分类结果")
    print("=" * 55)
    print()

    idx = result.get("class_index")
    orig = result.get("original_class_index")
    name = result.get("predicted_class", "未知")
    conf = result.get("confidence", "low")

    # 类别行
    conf_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(conf, "⚪")
    print(f"  预测类别: {name}")
    print(f"  VLM 索引: {idx}")
    if orig is not None:
        print(f"  原始14类索引: {orig}")
    print(f"  置信度:   {conf_emoji} {conf}")
    print()

    # 推理理由
    reasoning = result.get("reasoning", "")
    if reasoning:
        print("  ── 判断依据 ──")
        print(f"  {reasoning}")
        print()

    # 可观察特征
    features = result.get("observable_features", [])
    if features:
        print("  ── 可观察特征 ──")
        for f in features:
            print(f"  ▸ {f}")
        print()

    print("=" * 55)

    # 原始输出（debug）
    if "--raw" in sys.argv:
        print()
        print("── 模型原始输出 ──")
        print(result.get("raw_response", ""))
        print()


if __name__ == "__main__":
    main()
