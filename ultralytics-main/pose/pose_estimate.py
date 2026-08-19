from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LOCAL_CONFIG_DIR = SCRIPT_DIR / ".ultralytics_config"

LOCAL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(LOCAL_CONFIG_DIR))
sys.path.insert(0, str(PROJECT_ROOT))


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".mpeg", ".mpg", ".m4v"}


def parse_args() -> argparse.Namespace:
    default_model = SCRIPT_DIR / "yolo26m-pose.pt"
    default_output = SCRIPT_DIR / "outputs"

    parser = argparse.ArgumentParser(
        description="Run YOLO pose estimation on an image, video, directory, webcam, or stream."
    )
    parser.add_argument(
        "--source",
        "-s",
        required=True,
        help="Input image/video path, directory path, webcam index such as 0, or stream URL.",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=str(default_model),
        help=f"Pose model path. Default: {default_model}",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=str(default_output),
        help=f"Directory for annotated outputs. Default: {default_output}",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size. Default: 640")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold. Default: 0.25")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold. Default: 0.7")
    parser.add_argument(
        "--device",
        default=None,
        help="Device, for example 0 for first GPU or cpu. Default: Ultralytics auto-selects.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display prediction window while running. Useful for webcam/video.",
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Save YOLO-format detection/keypoint txt results next to rendered outputs.",
    )
    parser.add_argument(
        "--save-conf",
        action="store_true",
        help="Include confidence values in saved txt labels. Only used with --save-txt.",
    )
    parser.add_argument(
        "--hide-labels",
        action="store_true",
        help="Hide class labels on the rendered result.",
    )
    parser.add_argument(
        "--hide-conf",
        action="store_true",
        help="Hide confidence text on the rendered result.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming inference. Recommended for long videos, webcams, and RTSP/HTTP streams.",
    )
    return parser.parse_args()


def normalize_source(source: str) -> str | int:
    return int(source) if source.isdigit() else source


def validate_paths(model_path: Path, source: str) -> None:
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if source.isdigit() or "://" in source:
        return

    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source_path}")

    if source_path.is_file() and source_path.suffix.lower() not in IMAGE_EXTS | VIDEO_EXTS:
        raise ValueError(
            f"Unsupported source file type: {source_path.suffix}. "
            f"Supported images: {sorted(IMAGE_EXTS)}; videos: {sorted(VIDEO_EXTS)}"
        )


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    validate_paths(model_path, args.source)
    output_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    print(f"Running pose estimation on: {args.source}")
    results = model.predict(
        source=normalize_source(args.source),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        show=args.show,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        project=str(output_dir),
        name="predict",
        exist_ok=True,
        stream=args.stream,
        show_labels=not args.hide_labels,
        show_conf=not args.hide_conf,
    )

    if args.stream:
        frame_count = 0
        for _ in results:
            frame_count += 1
        print(f"Finished. Processed {frame_count} streamed frames.")

    print(f"Annotated results saved under: {output_dir / 'predict'}")


if __name__ == "__main__":
    main()
