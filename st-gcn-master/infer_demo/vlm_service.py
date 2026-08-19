"""VLM 图片分类服务 —— 基于 Ollama qwen3-vl:2b。

将一张 CPR 按压图片送入视觉语言模型，按 9 类空间姿态标准分类。
9 类 = 原始 14 类中可从单帧判断的子集（排除时序类 6/10/11/12/13）。

用法（Web）：由 app.py 的 /api/vlm_classify 调用 classify_image_bytes()
用法（CLI）：python Ollamaproject/cpr_vlm_classify.py <图片路径>
"""
from __future__ import annotations

import base64
import json
import re
import sys
from pathlib import Path

# ── 常量 ──────────────────────────────────────────────────────────────
MODEL = "qwen3-vl:2b"
OLLAMA_HOST = "http://localhost:11434"
# 上下文窗口：qwen3-vl:2b 默认 4096，但 9 类提示词 + 图片 token 常超 4600+。
# 显式放大到 16384，避免 exceed_context_size_error。2B 模型显存占用不大，可放心。
NUM_CTX = 16384

# VLM 9 类 → 原始 14 类索引映射
# VLM 0..5 = 原始 0..5；VLM 6 = 原始 7(蹲姿)；VLM 7 = 原始 8(站立)；VLM 8 = 原始 9(位置偏移)
VLM_TO_ORIG = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 7, 7: 8, 8: 9}

VLM_CLASSES = [
    "正确动作", "双手重叠", "握拳按压", "单手按压",
    "手臂弯曲", "手臂倾斜", "蹲姿按压", "站立按压", "位置偏移",
]

SYSTEM_PROMPT = """你是一名心肺复苏（CPR）胸外按压动作质量评估专家。
请仔细观察图片中正在进行CPR胸外按压的施救者，从以下9个类别中选出最匹配的一个，并给出判断理由。

【9类动作定义】
0. 正确动作：施救者跪在患者身侧，双手掌根重叠、十指交叉扣紧，双臂完全伸直与患者胸壁垂直，肩-腕-胸成一条直线，按压位置在胸骨中下段（两乳头连线中点）。
1. 双手重叠：双手有叠放但未交叉扣紧，手指松散未锁扣，或两手平贴无交叉。
2. 握拳按压：用握拳的方式（拳头）而非掌根接触胸壁进行按压。
3. 单手按压：仅用一只手掌按压胸壁，另一只手未参与或悬空。
4. 手臂弯曲：按压时肘关节弯曲，双臂未保持伸直状态。
5. 手臂倾斜：双臂未与患者胸壁垂直，从侧面看肩-腕连线与垂直方向有明显夹角。
6. 蹲姿按压：施救者蹲着而非跪着按压，臀部贴近脚跟，身体重心过低。
7. 站立按压：施救者站立弯腰按压，双膝未着地，身体重心过高。
8. 位置偏移：按压位置偏离胸骨中下段，偏左/偏右/偏上/偏下明显。

【判断要点】
- 优先观察手臂姿态（伸直/弯曲/倾斜）和手部接触方式（掌根/拳头/单手/双手）
- 其次观察身体姿势（跪姿/蹲姿/站姿）和按压位置
- 如果图片中看不清或无法判断，confidence 设为 low 并在 reasoning 中说明

【输出要求】
请严格按以下JSON格式输出，不要输出任何其他内容：
{
  "predicted_class": "类别名称",
  "class_index": 0到8的整数,
  "confidence": "high或medium或low",
  "reasoning": "简要说明判断依据",
  "observable_features": ["可观察到的关键特征1", "特征2"]
}"""


# ── 核心函数 ──────────────────────────────────────────────────────────
def _encode_image(image_bytes: bytes) -> str:
    """bytes → base64 字符串"""
    return base64.b64encode(image_bytes).decode("utf-8")


def _parse_response(text: str) -> dict:
    """从模型输出中提取 JSON，兜底正则提取。"""
    # 尝试直接解析
    text = text.strip()
    # 去掉可能的 ```json ... ``` 包裹
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # 兜底：从文本中提取 {...}
        m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
            except json.JSONDecodeError:
                data = _fallback_parse(text)
        else:
            data = _fallback_parse(text)

    # 规范化字段
    class_idx = data.get("class_index")
    if class_idx is not None:
        try:
            class_idx = int(class_idx)
            if class_idx < 0 or class_idx > 8:
                class_idx = None
        except (TypeError, ValueError):
            class_idx = None

    if class_idx is None:
        # 尝试从 predicted_class 名称反查
        pred_name = data.get("predicted_class", "")
        for i, name in enumerate(VLM_CLASSES):
            if name in pred_name or pred_name in name:
                class_idx = i
                break

    data["class_index"] = class_idx
    pred_name = data.get("predicted_class", "")
    # 去掉 "0. 正确动作" 这类编号前缀
    pred_name = re.sub(r"^\s*\d+[.、)．\s]+", "", pred_name).strip()
    # 模型有时只回数字索引（如 "0"）而不回类别名 → 用索引反查名称
    if pred_name.isdigit() and class_idx is not None:
        pred_name = VLM_CLASSES[class_idx]
    data["predicted_class"] = pred_name or (VLM_CLASSES[class_idx] if class_idx is not None else "未知")
    data["confidence"] = data.get("confidence", "low")
    data["reasoning"] = data.get("reasoning", "")
    data["observable_features"] = data.get("observable_features", [])

    # 映射回原始 14 类索引
    if class_idx is not None:
        data["original_class_index"] = VLM_TO_ORIG[class_idx]
    else:
        data["original_class_index"] = None

    return data


def _fallback_parse(text: str) -> dict:
    """正则兜底：从纯文本中提取分类信息。"""
    result = {"predicted_class": "未知", "class_index": None,
              "confidence": "low", "reasoning": text[:200],
              "observable_features": []}
    for i, name in enumerate(VLM_CLASSES):
        if name in text:
            result["predicted_class"] = name
            result["class_index"] = i
            break
    return result


def classify_image_bytes(image_bytes: bytes, model: str = MODEL) -> dict:
    """对图片 bytes 做 VLM 分类，返回解析后的 dict。

    Parameters
    ----------
    image_bytes : bytes
        图片原始字节（jpg/png/bmp 均可）
    model : str
        Ollama 模型名，默认 qwen3-vl:2b

    Returns
    -------
    dict with keys:
        predicted_class, class_index, confidence, reasoning,
        observable_features, original_class_index, raw_response
    """
    try:
        from ollama import Client
    except ImportError as e:
        return {"error": f"ollama 包未安装: {e}", "predicted_class": None}

    img_b64 = _encode_image(image_bytes)

    try:
        client = Client(host=OLLAMA_HOST)
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "请对这张CPR按压图片进行分类评估。",
                 "images": [img_b64]},
            ],
            options={"num_ctx": NUM_CTX},
        )
    except Exception as e:
        return {"error": f"Ollama 调用失败: {type(e).__name__}: {e}",
                "predicted_class": None}

    raw_text = response.get("message", {}).get("content", "") if isinstance(response, dict) else response["message"]["content"]
    result = _parse_response(raw_text)
    result["raw_response"] = raw_text
    result["model"] = model
    return result


def classify_image_file(image_path: str, model: str = MODEL) -> dict:
    """对图片文件做 VLM 分类。"""
    path = Path(image_path)
    if not path.exists():
        return {"error": f"文件不存在: {image_path}", "predicted_class": None}
    return classify_image_bytes(path.read_bytes(), model=model)


# ── CLI 入口 ──────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print("用法: python vlm_service.py <图片路径>")
        print("示例: python vlm_service.py C:\\path\\to\\cpr_frame.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    print(f"模型: {MODEL}")
    print(f"图片: {image_path}")
    print("正在分析...\n")

    result = classify_image_file(image_path)
    if result.get("error"):
        print(f"错误: {result['error']}")
        sys.exit(1)

    print("=" * 50)
    print(f"  预测类别: {result['predicted_class']}  (VLM索引: {result['class_index']}, 原始索引: {result.get('original_class_index')})")
    print(f"  置信度:   {result['confidence']}")
    print(f"  理由:     {result['reasoning']}")
    if result.get("observable_features"):
        print(f"  可观察特征:")
        for feat in result["observable_features"]:
            print(f"    - {feat}")
    print("=" * 50)


if __name__ == "__main__":
    main()
