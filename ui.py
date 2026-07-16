import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton, QFileDialog,
                             QHBoxLayout, QVBoxLayout, QFrame, QProgressBar, QMessageBox, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QPixmap, QImage
from ultralytics import YOLO
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class VideoThread(QThread):
    change_pixmap = pyqtSignal(np.ndarray)
    finished = pyqtSignal()

    def __init__(self, model, source=0):
        super().__init__()
        self.model = model
        self.source = source
        self.running = False

    def run(self):
        self.running = True
        cap = cv2.VideoCapture(self.source)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model(frame)
            annotated_frame = results[0].plot()
            self.change_pixmap.emit(annotated_frame)
        cap.release()
        self.finished.emit()

    def stop(self):
        self.running = False

class YOLOv8DetectorQt(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能水果缺陷检测系统")
        self.setGeometry(100, 100, 1200, 900)
        self.model = None
        self.video_thread = None
        self.init_ui()

    def init_ui(self):
        # 主窗口布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        # 左侧控制面板
        control_panel = QFrame()
        control_panel.setFrameShape(QFrame.StyledPanel)
        control_panel.setFixedWidth(350)
        control_layout = QVBoxLayout(control_panel)
        # 右侧显示区域
        display_panel = QWidget()
        display_layout = QVBoxLayout(display_panel)
        # 图像显示区域
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: white;")
        display_layout.addWidget(self.image_label)
        # 图表显示区域
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        display_layout.addWidget(self.canvas)
        # 将面板添加到主布局
        main_layout.addWidget(control_panel)
        main_layout.addWidget(display_panel)
        # 控制面板组件
        self.create_model_section(control_layout)
        self.create_data_section(control_layout)
        self.create_control_section(control_layout)
        self.create_progress_section(control_layout)
        # 状态栏
        self.status_bar = self.statusBar()

    def create_model_section(self, layout):
        group = QFrame()
        group.setFrameShape(QFrame.StyledPanel)
        group_layout = QVBoxLayout(group)

        # 模型路径
        model_label = QLabel("模型路径:")
        self.model_entry = QLineEdit()
        browse_model_btn = QPushButton("浏览")
        browse_model_btn.clicked.connect(self.browse_model)
        load_model_btn = QPushButton("加载模型")
        load_model_btn.clicked.connect(self.load_model)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(browse_model_btn)
        btn_layout.addWidget(load_model_btn)

        group_layout.addWidget(model_label)
        group_layout.addWidget(self.model_entry)
        group_layout.addLayout(btn_layout)
        layout.addWidget(group)

    def create_data_section(self, layout):
        group = QFrame()
        group.setFrameShape(QFrame.StyledPanel)
        group_layout = QVBoxLayout(group)

        # 验证集路径
        valset_label = QLabel("验证集路径:")
        self.valset_entry = QLineEdit()
        browse_valset_btn = QPushButton("浏览")
        browse_valset_btn.clicked.connect(self.browse_valset)

        # 图片路径
        image_label = QLabel("图片路径:")
        self.image_entry = QLineEdit()
        browse_image_btn = QPushButton("浏览")
        browse_image_btn.clicked.connect(self.browse_image)

        group_layout.addWidget(valset_label)
        group_layout.addWidget(self.valset_entry)
        group_layout.addWidget(browse_valset_btn)
        group_layout.addWidget(image_label)
        group_layout.addWidget(self.image_entry)
        group_layout.addWidget(browse_image_btn)
        layout.addWidget(group)

    def create_control_section(self, layout):
        group = QFrame()
        group.setFrameShape(QFrame.StyledPanel)
        group_layout = QVBoxLayout(group)

        # 功能按钮
        detect_image_btn = QPushButton("图片检测")
        detect_image_btn.clicked.connect(self.detect_image)
        self.video_btn = QPushButton("视频检测")
        self.video_btn.clicked.connect(self.toggle_video)
        validate_btn = QPushButton("验证评估")
        validate_btn.clicked.connect(self.validate_model)

        group_layout.addWidget(detect_image_btn)
        group_layout.addWidget(self.video_btn)
        group_layout.addWidget(validate_btn)
        layout.addWidget(group)

    def create_progress_section(self, layout):
        self.progress = QProgressBar()
        layout.addWidget(self.progress)

    def browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", os.path.expanduser("~"),
            "Model Files (*.pt *.onnx);;All Files (*)"
        )
        if path:
            self.model_entry.setText(path)

    def load_model(self):
        model_path = self.model_entry.text()
        if not model_path:
            QMessageBox.critical(self, "错误", "请先选择模型文件")
            return

        try:
            self.status_bar.showMessage("正在加载模型...")
            self.model = YOLO(model_path)
            QMessageBox.information(self, "成功", "模型加载成功!")
            self.status_bar.showMessage("模型加载成功")

            # 自动查找验证集
            model_dir = os.path.dirname(model_path)
            val_path = os.path.join(model_dir, "val")
            if os.path.exists(val_path):
                self.valset_entry.setText(val_path)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载模型失败: {str(e)}")
            self.status_bar.showMessage("模型加载失败")

    def browse_valset(self):
        path = QFileDialog.getExistingDirectory(
            self, "选择验证集目录", os.path.expanduser("~")
        )
        if path:
            self.valset_entry.setText(path)

    def browse_image(self):
        if not self.model:
            QMessageBox.critical(self, "错误", "请先加载模型")
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", os.path.expanduser("~"),
            "Image Files (*.jpg *.jpeg *.png);;All Files (*)"
        )
        if path:
            self.image_entry.setText(path)
            self.show_image(path)

    def show_image(self, path):
        pixmap = QPixmap(path)
        if pixmap.isNull():
            QMessageBox.critical(self, "错误", "无法加载图片")
            return

        # 缩放图片以适应标签
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

    def detect_image(self):
        if not self.model:
            QMessageBox.critical(self, "错误", "请先加载模型")
            return

        image_path = self.image_entry.text()
        if not image_path:
            QMessageBox.critical(self, "错误", "请先选择图片")
            return

        try:
            self.status_bar.showMessage("正在检测...")
            results = self.model(image_path)
            self.show_detection_result(results[0])
            self.status_bar.showMessage("检测完成")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"检测失败: {str(e)}")
            self.status_bar.showMessage("检测失败")

    def show_detection_result(self, result):
        im_array = result.plot()
        height, width, channel = im_array.shape
        bytes_per_line = 3 * width
        q_img = QImage(im_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

    def toggle_video(self):
        if self.video_thread and self.video_thread.isRunning():
            self.stop_video()
        else:
            self.start_video()

    def start_video(self):
        if not self.model:
            QMessageBox.critical(self, "错误", "请先加载模型")
            return

        reply = QMessageBox.question(
            self, "选择输入源", "使用摄像头吗?", 
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            source = 0
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "选择视频文件", os.path.expanduser("~"),
                "Video Files (*.mp4 *.avi *.mov);;All Files (*)"
            )
            if not path:
                return
            source = path

        self.video_thread = VideoThread(self.model, source)
        self.video_thread.change_pixmap.connect(self.update_video_frame)
        self.video_thread.finished.connect(self.video_finished)
        self.video_btn.setText("停止检测")
        self.video_thread.start()

    @pyqtSlot(np.ndarray)
    def update_video_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(convert_to_qt_format)
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

    def video_finished(self):
        self.video_btn.setText("视频检测")
        self.status_bar.showMessage("视频检测已停止")

    def stop_video(self):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread.quit()
            self.video_thread.wait()

    def validate_model(self):
        if not self.model:
            QMessageBox.critical(self, "错误", "请先加载模型")
            return

        val_path = self.valset_entry.text()
        if not val_path:
            QMessageBox.critical(self, "错误", "请选择验证集路径")
            return

        try:
            self.status_bar.showMessage("正在评估验证集...")
            self.progress.setValue(0)

            metrics = self.model.val(
                data=os.path.join(val_path, "data.yaml"),
                split='val',
                plots=False
            )

            self.progress.setValue(100)
            self.visualize_metrics(metrics)
            self.status_bar.showMessage("验证集评估完成")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"验证失败: {str(e)}")
            self.status_bar.showMessage("验证失败")

    def visualize_metrics(self, metrics):
        self.ax.clear()
        precision = metrics.box.mp
        recall = metrics.box.mr
        map50 = metrics.box.map50
        map = metrics.box.map

        labels = ['Precision', 'Recall', 'mAP50', 'mAP']
        values = [precision, recall, map50, map]
        colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']

        bars = self.ax.bar(labels, values, color=colors)
        self.ax.set_ylim(0, 1)
        self.ax.set_title('模型性能指标')
        self.ax.set_ylabel('得分')

        for bar in bars:
            height = bar.get_height()
            self.ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')

        self.canvas.draw()

    def resizeEvent(self, event):
        # 当窗口大小改变时更新图片显示
        if hasattr(self, 'current_pixmap'):
            scaled_pixmap = self.current_pixmap.scaled(
                self.image_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
        super().resizeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = YOLOv8DetectorQt()
    window.show()
    sys.exit(app.exec_())