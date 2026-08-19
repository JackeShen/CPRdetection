from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


UI_DIR = Path(__file__).resolve().parent
POSE_DIR = UI_DIR.parent
PROJECT_ROOT = POSE_DIR.parent
DEFAULT_MODEL = POSE_DIR / "yolo26m-pose.pt"
DEFAULT_OUTPUT = POSE_DIR / "outputs" / "ui_predict"
LOCAL_CONFIG_DIR = POSE_DIR / ".ultralytics_config"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".mpeg", ".mpg", ".m4v"}


class PoseWorker(QThread):
    frame_ready = pyqtSignal(object)
    message = pyqtSignal(str)
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(
        self,
        source: str,
        model_path: str,
        output_dir: str,
        conf: float,
        iou: float,
        imgsz: int,
        device: str | None,
        save_output: bool,
        save_txt: bool,
        save_conf: bool,
    ) -> None:
        super().__init__()
        self.source = source
        self.model_path = model_path
        self.output_dir = output_dir
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.save_output = save_output
        self.save_txt = save_txt
        self.save_conf = save_conf
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            LOCAL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("YOLO_CONFIG_DIR", str(LOCAL_CONFIG_DIR))
            sys.path.insert(0, str(PROJECT_ROOT))

            from ultralytics import YOLO

            source_obj = int(self.source) if self.source.isdigit() else self.source
            output_dir = Path(self.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            self.message.emit(f"Loading model: {self.model_path}")
            model = YOLO(self.model_path)

            if self._is_image_source(self.source):
                self._run_image(model, output_dir)
            else:
                self._run_video(model, source_obj, output_dir)
        except Exception as exc:
            self.failed.emit(str(exc))

    def _is_image_source(self, source: str) -> bool:
        return Path(source).suffix.lower() in IMAGE_EXTS if not source.isdigit() and "://" not in source else False

    def _predict_frame(self, model, frame):
        results = model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        return results[0], results[0].plot()

    def _run_image(self, model, output_dir: Path) -> None:
        frame = cv2.imread(self.source)
        if frame is None:
            raise RuntimeError(f"Cannot read image: {self.source}")

        result, annotated = self._predict_frame(model, frame)
        self.frame_ready.emit(annotated)

        if self.save_output:
            out_path = output_dir / f"{Path(self.source).stem}_pose.jpg"
            cv2.imwrite(str(out_path), annotated)
        if self.save_txt:
            label_path = output_dir / "labels" / f"{Path(self.source).stem}.txt"
            result.save_txt(label_path, save_conf=self.save_conf)
            self.message.emit(f"Saved labels: {label_path}")
        if self.save_output:
            self.finished_ok.emit(f"Saved: {out_path}")
        else:
            self.finished_ok.emit("Image inference finished.")

    def _run_video(self, model, source_obj, output_dir: Path) -> None:
        cap = cv2.VideoCapture(source_obj)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {self.source}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1:
            fps = 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if self.save_output and width > 0 and height > 0:
            out_name = "webcam_pose.mp4" if str(self.source).isdigit() else f"{Path(str(self.source)).stem}_pose.mp4"
            out_path = output_dir / out_name
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        else:
            out_path = None

        frame_idx = 0
        delay = 1.0 / fps
        self.message.emit("Running realtime pose estimation...")

        while not self._stop:
            ok, frame = cap.read()
            if not ok:
                break

            start = time.time()
            result, annotated = self._predict_frame(model, frame)
            self.frame_ready.emit(annotated)
            if writer is not None:
                writer.write(annotated)
            if self.save_txt:
                stem = "webcam" if str(self.source).isdigit() else Path(str(self.source)).stem
                label_path = output_dir / "labels" / f"{stem}_{frame_idx:06d}.txt"
                result.save_txt(label_path, save_conf=self.save_conf)

            frame_idx += 1
            if frame_idx % 10 == 0:
                self.message.emit(f"Processed frames: {frame_idx}")

            elapsed = time.time() - start
            if elapsed < delay:
                time.sleep(delay - elapsed)

        cap.release()
        if writer is not None:
            writer.release()
            self.finished_ok.emit(f"Saved: {out_path}")
        else:
            self.finished_ok.emit("Video inference stopped." if self._stop else "Video inference finished.")


class PoseUI(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.worker: PoseWorker | None = None
        self.current_frame = None
        self.setWindowTitle("YOLO Pose Estimation UI")
        self.resize(1120, 760)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        main = QHBoxLayout(root)
        main.setContentsMargins(14, 14, 14, 14)
        main.setSpacing(12)

        left = QVBoxLayout()
        title = QLabel("YOLO Pose Estimation")
        font = QFont()
        font.setPointSize(17)
        font.setBold(True)
        title.setFont(font)
        left.addWidget(title)

        input_box = QGroupBox("Input")
        form = QFormLayout(input_box)
        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("Image/video path, folder, or camera index 0")
        form.addRow("Source:", self._path_row(self.source_edit, self.choose_source))
        self.model_edit = QLineEdit(str(DEFAULT_MODEL))
        form.addRow("Model:", self._path_row(self.model_edit, self.choose_model))
        self.output_edit = QLineEdit(str(DEFAULT_OUTPUT))
        form.addRow("Output:", self._path_row(self.output_edit, self.choose_output_dir))
        left.addWidget(input_box)

        param_box = QGroupBox("Parameters")
        params = QFormLayout(param_box)
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.25)
        params.addRow("Conf:", self.conf_spin)
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.01, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.70)
        params.addRow("IoU:", self.iou_spin)
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(128, 2048)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)
        params.addRow("Image size:", self.imgsz_spin)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["Auto", "CPU", "GPU 0"])
        params.addRow("Device:", self.device_combo)
        self.save_combo = QComboBox()
        self.save_combo.addItems(["Save output", "Preview only"])
        params.addRow("Save:", self.save_combo)
        self.txt_combo = QComboBox()
        self.txt_combo.addItems(["No TXT", "Save TXT"])
        params.addRow("Labels:", self.txt_combo)
        self.conf_combo = QComboBox()
        self.conf_combo.addItems(["No label conf", "Save label conf"])
        params.addRow("Label conf:", self.conf_combo)
        left.addWidget(param_box)

        buttons = QHBoxLayout()
        self.run_btn = QPushButton("Start")
        self.run_btn.clicked.connect(self.run_pose)
        buttons.addWidget(self.run_btn)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_pose)
        buttons.addWidget(self.stop_btn)
        left.addLayout(buttons)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(160)
        left.addWidget(self.log, 1)
        main.addLayout(left, 0)

        right = QVBoxLayout()
        self.preview = QLabel("Preview")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(640, 480)
        self.preview.setStyleSheet("background:#111; color:#ddd; border:1px solid #444;")
        right.addWidget(self.preview, 1)
        main.addLayout(right, 1)

        self.setCentralWidget(root)

    def _path_row(self, edit: QLineEdit, chooser) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit, 1)
        button = QPushButton("Browse")
        button.clicked.connect(chooser)
        layout.addWidget(button)
        return row

    def choose_source(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose image or video",
            str(POSE_DIR),
            "Media (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff *.mp4 *.avi *.mov *.mkv *.wmv);;All Files (*)",
        )
        if path:
            self.source_edit.setText(path)

    def choose_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose model", str(POSE_DIR), "Model (*.pt);;All Files (*)")
        if path:
            self.model_edit.setText(path)

    def choose_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose output directory", str(DEFAULT_OUTPUT))
        if path:
            self.output_edit.setText(path)

    def run_pose(self) -> None:
        if self.worker is not None:
            QMessageBox.information(self, "Running", "A pose estimation task is already running.")
            return

        source = self.source_edit.text().strip()
        model = self.model_edit.text().strip()
        output = self.output_edit.text().strip()
        if not source:
            QMessageBox.warning(self, "Missing source", "Please choose a source file or enter camera index 0.")
            return
        if not model:
            QMessageBox.warning(self, "Missing model", "Please choose a model file.")
            return

        device_text = self.device_combo.currentText()
        device = None
        if device_text == "CPU":
            device = "cpu"
        elif device_text == "GPU 0":
            device = "0"

        self.worker = PoseWorker(
            source=source,
            model_path=model,
            output_dir=output,
            conf=self.conf_spin.value(),
            iou=self.iou_spin.value(),
            imgsz=self.imgsz_spin.value(),
            device=device,
            save_output=self.save_combo.currentText() == "Save output",
            save_txt=self.txt_combo.currentText() == "Save TXT",
            save_conf=self.conf_combo.currentText() == "Save label conf",
        )
        self.worker.frame_ready.connect(self.show_frame)
        self.worker.message.connect(self.append_log)
        self.worker.finished_ok.connect(self.worker_finished)
        self.worker.failed.connect(self.worker_failed)
        self.worker.start()

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.append_log("Started.")

    def stop_pose(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.append_log("Stopping...")

    def show_frame(self, frame) -> None:
        self.current_frame = frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg).scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview.setPixmap(pixmap)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.current_frame is not None:
            self.show_frame(self.current_frame)

    def append_log(self, text: str) -> None:
        self.log.append(text.rstrip())
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def worker_finished(self, message: str) -> None:
        self.append_log(message)
        self.worker = None
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def worker_failed(self, message: str) -> None:
        self.append_log(f"Error: {message}")
        self.worker = None
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)


def main() -> None:
    app = QApplication(sys.argv)
    window = PoseUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
