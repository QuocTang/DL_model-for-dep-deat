"""
PySide6 desktop UI for single-image inference.

Run:
  python v2/06_inference/ui_pyside.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core import (
    DEFAULT_ML_MODEL,
    DEFAULT_RESULTS_JSON,
    DEFAULT_YOLO_WEIGHTS,
    DeepFeatureMLPredictor,
)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Durian Disease Inference - PySide6")
        self.resize(1200, 760)

        self.predictor: DeepFeatureMLPredictor | None = None
        self.current_image_rgb: np.ndarray | None = None
        self.current_image_path: Path | None = None

        self._build_ui()
        self._load_model()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QHBoxLayout(root)

        left_panel = QVBoxLayout()
        right_panel = QVBoxLayout()
        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(right_panel, 1)

        self.image_label = QLabel("No image selected")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(520, 520)
        self.image_label.setStyleSheet("border: 1px solid #888; background: #f3f3f3;")
        left_panel.addWidget(self.image_label)

        image_btn_row = QHBoxLayout()
        self.select_image_btn = QPushButton("Select Image")
        self.select_image_btn.clicked.connect(self._select_image)
        self.run_infer_btn = QPushButton("Run Inference")
        self.run_infer_btn.clicked.connect(self._run_inference)
        image_btn_row.addWidget(self.select_image_btn)
        image_btn_row.addWidget(self.run_infer_btn)
        left_panel.addLayout(image_btn_row)

        path_group = QGroupBox("Model Paths")
        path_form = QFormLayout(path_group)

        self.yolo_path_edit = QLineEdit(str(DEFAULT_YOLO_WEIGHTS))
        self.ml_model_path_edit = QLineEdit(str(DEFAULT_ML_MODEL))
        self.results_json_path_edit = QLineEdit(str(DEFAULT_RESULTS_JSON))

        path_form.addRow("YOLO weights", self.yolo_path_edit)
        path_form.addRow("ML model", self.ml_model_path_edit)
        path_form.addRow("results.json", self.results_json_path_edit)

        path_btn_row = QGridLayout()
        self.load_model_btn = QPushButton("Load/Reload Models")
        self.load_model_btn.clicked.connect(self._load_model)
        path_btn_row.addWidget(self.load_model_btn, 0, 0)
        path_form.addRow(path_btn_row)

        right_panel.addWidget(path_group)

        setting_group = QGroupBox("Inference Settings")
        setting_form = QFormLayout(setting_group)

        self.layers_edit = QLineEdit("4,6,9")
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(320, 1280)
        self.imgsz_spin.setValue(640)
        self.topk_spin = QSpinBox()
        self.topk_spin.setRange(1, 10)
        self.topk_spin.setValue(5)

        setting_form.addRow("Feature layers", self.layers_edit)
        setting_form.addRow("Input size", self.imgsz_spin)
        setting_form.addRow("Top-K", self.topk_spin)

        right_panel.addWidget(setting_group)

        self.pred_label = QLabel("Predicted class: -")
        self.pred_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        right_panel.addWidget(self.pred_label)

        self.result_table = QTableWidget(0, 3)
        self.result_table.setHorizontalHeaderLabels(["Model Label", "Class", "Probability"])
        self.result_table.horizontalHeader().setStretchLastSection(True)
        right_panel.addWidget(self.result_table)

        self.raw_text = QTextEdit()
        self.raw_text.setReadOnly(True)
        self.raw_text.setPlaceholderText("Raw prediction output")
        right_panel.addWidget(self.raw_text)

    def _select_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp)",
        )
        if not file_path:
            return

        self.current_image_path = Path(file_path)
        image = Image.open(file_path).convert("RGB")
        self.current_image_rgb = np.array(image, dtype=np.uint8)
        self._show_image(self.current_image_rgb)

    def _show_image(self, image_rgb: np.ndarray) -> None:
        h, w, c = image_rgb.shape
        qimage = QImage(image_rgb.data, w, h, c * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        scaled = pixmap.scaled(
            self.image_label.width(),
            self.image_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def _load_model(self) -> None:
        try:
            layers = [int(x.strip()) for x in self.layers_edit.text().split(",") if x.strip()]
            self.predictor = DeepFeatureMLPredictor(
                yolo_weights=Path(self.yolo_path_edit.text().strip()),
                ml_model_path=Path(self.ml_model_path_edit.text().strip()),
                results_json_path=Path(self.results_json_path_edit.text().strip()),
                layers=layers,
                imgsz=int(self.imgsz_spin.value()),
            )
            QMessageBox.information(self, "Success", "Models loaded successfully")
        except Exception as exc:
            self.predictor = None
            QMessageBox.critical(self, "Load Error", str(exc))

    def _run_inference(self) -> None:
        if self.predictor is None:
            QMessageBox.warning(self, "Warning", "Model is not loaded")
            return
        if self.current_image_rgb is None:
            QMessageBox.warning(self, "Warning", "Please select an image first")
            return

        try:
            top_k = int(self.topk_spin.value())
            pred = self.predictor.predict_from_rgb(self.current_image_rgb, top_k=top_k)
        except Exception as exc:
            QMessageBox.critical(self, "Inference Error", str(exc))
            return

        self.pred_label.setText(f"Predicted class: {pred['predicted_class']}")
        self.raw_text.setPlainText(str(pred))

        rows = pred.get("top_k", [])
        self.result_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            self.result_table.setItem(r, 0, QTableWidgetItem(str(row.get("model_label", ""))))
            self.result_table.setItem(r, 1, QTableWidgetItem(str(row.get("class_name", ""))))
            self.result_table.setItem(r, 2, QTableWidgetItem(f"{row.get('probability', 0.0):.6f}"))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
