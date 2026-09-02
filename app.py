import sys
import os
import json
import time
import logging
from typing import List, Dict, Any, Optional

from PySide6.QtCore import Qt, QThread, Signal, Slot, QUrl
from PySide6.QtGui import QIcon, QFont, QPixmap, QImage, QColor, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QSpinBox, QComboBox, QCheckBox,
    QProgressBar, QTextEdit, QTableWidget, QTableWidgetItem, QTabWidget,
    QFileDialog, QGroupBox, QSplitter, QFrame, QHeaderView, QMessageBox
)

from video_processor import VideoProcessor
from vision_engine import AdvancedVisionAnalyzer
from summarizer import TextSummarizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dontknow.gui")


class ProcessingWorker(QThread):
    """
    Background worker thread running the Advanced SOTA Neural Vision Analyzer Engine
    (Microsoft GIT-Large + Salesforce BLIP-Large with 10-Beam Search).
    """
    progress_signal = Signal(int, int, str)  # current_step, total_steps, status_text
    log_signal = Signal(str)                  # log text message
    frame_preview_signal = Signal(str)        # frame image path currently being processed
    finished_signal = Signal(dict)            # dict with final results
    error_signal = Signal(str)               # error message

    def __init__(
        self,
        video_path: str,
        output_subfolder: str,
        frame_step: int = 1,
        detailed_prompts: bool = True,
        sentences_count: int = 5
    ):
        super().__init__()
        self.video_path = video_path
        self.output_subfolder = output_subfolder
        self.frame_step = max(1, frame_step)
        self.detailed_prompts = detailed_prompts
        self.sentences_count = sentences_count
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            start_time = time.time()
            self.log_signal.emit(f"🚀 Starting Advanced Neural Vision Pipeline for: {os.path.basename(self.video_path)}")
            
            # Step 1: Video Analysis & Frame Extraction
            vp = VideoProcessor(self.video_path)
            meta = vp.get_metadata()
            total_video_frames = meta["total_frames"]
            self.log_signal.emit(f"📹 Video Metadata: {meta['width']}x{meta['height']} @ {meta['fps']:.2f} FPS | Total Frames: {total_video_frames} | Duration: {meta['duration_sec']:.1f}s")
            
            abs_subfolder = os.path.abspath(self.output_subfolder)
            os.makedirs(abs_subfolder, exist_ok=True)
            self.log_signal.emit(f"📁 Output subfolder ready: {abs_subfolder}")

            # 1.1 Frame Extraction
            self.log_signal.emit(f"🖼️ Extracting frames (sampling step: {self.frame_step})...")

            def on_frame_extracted(idx, total, path, ts):
                if self._is_cancelled:
                    return
                percent = int((idx / total) * 35)
                self.progress_signal.emit(percent, 100, f"Extracting Frame {idx}/{total} ({ts:.1f}s)")
                self.frame_preview_signal.emit(path)

            extracted_frames = vp.extract_frames(
                output_dir=abs_subfolder,
                frame_step=self.frame_step,
                progress_callback=on_frame_extracted,
                stop_checker=lambda: self._is_cancelled
            )

            if self._is_cancelled:
                self.log_signal.emit("🛑 Task cancelled during frame extraction.")
                return

            self.log_signal.emit(f"✅ Extracted {len(extracted_frames)} frame PNG files into subfolder '{abs_subfolder}'.")

            # Step 2: Advanced SOTA Vision AI Analysis on all frames from frame 1 to last frame
            self.log_signal.emit("🧠 Initializing Advanced Neural Vision Engine (Microsoft GIT-Large + Salesforce BLIP-Large)...")
            vision_engine = AdvancedVisionAnalyzer()
            
            self.log_signal.emit(f"🎨 Generating High-Accuracy Image-to-Text for all {len(extracted_frames)} frames...")

            def on_prompt_generated(current_idx, total_count, record):
                if self._is_cancelled:
                    return
                percent = 35 + int((current_idx / total_count) * 55)
                prompt_snippet = record['image_prompt'][:70] + "..." if len(record['image_prompt']) > 70 else record['image_prompt']
                self.progress_signal.emit(percent, 100, f"Analyzing Frame {current_idx}/{total_count}")
                self.frame_preview_signal.emit(record.get("image_path", ""))
                self.log_signal.emit(f"  [Frame {record['frame_index']} | {record['timestamp_sec']:.1f}s]: Text -> \"{prompt_snippet}\"")

            raw_prompts_array = vision_engine.process_frame_list(
                frames_info=extracted_frames,
                detailed_prompts=self.detailed_prompts,
                progress_callback=on_prompt_generated,
                stop_checker=lambda: self._is_cancelled
            )

            if self._is_cancelled:
                self.log_signal.emit("🛑 Task cancelled during visual text generation.")
                return

            self.log_signal.emit(f"✅ Generated advanced neural descriptions across all {len(raw_prompts_array)} frames.")

            # Step 3: Deduplication & Unified Story Summarization
            self.log_signal.emit("📝 Synthesizing all frame text entries into ONE unified compact story summary...")
            self.progress_signal.emit(92, 100, "Synthesizing Unified Story...")
            
            summarizer = TextSummarizer()
            summary_result = summarizer.generate_story_summary(
                raw_prompts_array=raw_prompts_array,
                sentences_count=self.sentences_count
            )

            elapsed = time.time() - start_time
            self.log_signal.emit(f"🎉 Pipeline completed in {elapsed:.2f} seconds!")
            self.progress_signal.emit(100, 100, "Completed!")

            final_payload = {
                "metadata": meta,
                "output_subfolder": abs_subfolder,
                "extracted_frames_count": len(extracted_frames),
                "raw_prompts_array": raw_prompts_array,
                "summary_result": summary_result,
                "elapsed_sec": elapsed
            }
            self.finished_signal.emit(final_payload)

        except Exception as e:
            logger.exception("Worker exception")
            self.error_signal.emit(str(e))


class DropAreaWidget(QFrame):
    """Interactive drag & drop widget for video selection."""
    file_dropped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setObjectName("dropArea")
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        self.label_icon = QLabel("🎬")
        self.label_icon.setStyleSheet("font-size: 44px; background: transparent;")
        self.label_icon.setAlignment(Qt.AlignCenter)

        self.label_text = QLabel("Drag & Drop Video File Here\nor Click to Browse")
        self.label_text.setStyleSheet("font-size: 14px; color: #a0aec0; font-weight: 500; background: transparent;")
        self.label_text.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.label_icon)
        layout.addWidget(self.label_text)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            filePath, _ = QFileDialog.getOpenFileName(
                self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm);;All Files (*)"
            )
            if filePath:
                self.file_dropped.emit(filePath)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            filepath = urls[0].toLocalFile()
            if filepath:
                self.file_dropped.emit(filepath)


class DontKnowApp(QMainWindow):
    """Main Window for 'dontknow' Advanced Neural Vision AI App."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("dontknow - Advanced SOTA Neural Vision AI & Story Generator")
        self.resize(1180, 840)
        self.setMinimumSize(950, 680)

        self.selected_video_path = ""
        self.current_worker = None
        self.last_results = None

        self._setup_stylesheet()
        self._init_ui()

    def _setup_stylesheet(self):
        """Apply modern dark-themed glassmorphism aesthetic."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0f172a;
                color: #f8fafc;
            }
            QWidget {
                color: #e2e8f0;
                font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
            }
            QGroupBox {
                border: 1px solid #334155;
                border-radius: 10px;
                margin-top: 12px;
                padding-top: 14px;
                font-weight: 600;
                color: #38bdf8;
                background-color: #1e293b;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                background-color: #0f172a;
                border-radius: 4px;
            }
            #dropArea {
                border: 2px dashed #475569;
                border-radius: 12px;
                background-color: #1e293b;
                min-height: 110px;
            }
            #dropArea:hover {
                border-color: #38bdf8;
                background-color: #334155;
            }
            QLineEdit, QComboBox, QSpinBox {
                background-color: #0f172a;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 6px 10px;
                color: #f8fafc;
            }
            QLineEdit:focus, QComboBox:focus {
                border-color: #38bdf8;
            }
            QPushButton {
                background: linear-gradient(135deg, #0284c7, #2563eb);
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                color: #ffffff;
                font-weight: 600;
            }
            QPushButton:hover {
                background: linear-gradient(135deg, #38bdf8, #3b82f6);
            }
            #btnPrimary {
                background: linear-gradient(135deg, #059669, #10b981);
                font-size: 15px;
                padding: 10px 22px;
            }
            #btnPrimary:hover {
                background: linear-gradient(135deg, #10b981, #34d399);
            }
            #btnPrimary:disabled {
                background: #334155;
                color: #64748b;
            }
            #btnCancel {
                background: #ef4444;
            }
            #btnCancel:hover {
                background: #f87171;
            }
            QProgressBar {
                border: 1px solid #334155;
                border-radius: 6px;
                text-align: center;
                background-color: #0f172a;
                color: #ffffff;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background: linear-gradient(90deg, #38bdf8, #818cf8);
                border-radius: 5px;
            }
            QTextEdit, QTableWidget {
                background-color: #0f172a;
                border: 1px solid #334155;
                border-radius: 8px;
                color: #e2e8f0;
                gridline-color: #334155;
            }
            QHeaderView::section {
                background-color: #1e293b;
                color: #38bdf8;
                padding: 6px;
                border: 1px solid #334155;
                font-weight: 600;
            }
            QTabWidget::pane {
                border: 1px solid #334155;
                border-radius: 8px;
                background-color: #1e293b;
            }
            QTabBar::tab {
                background-color: #0f172a;
                color: #94a3b8;
                padding: 8px 18px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                margin-right: 2px;
                font-weight: 600;
            }
            QTabBar::tab:selected {
                background-color: #1e293b;
                color: #38bdf8;
                border-bottom: 2px solid #38bdf8;
            }
        """)

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(16, 16, 16, 16)

        # Header Bar
        header = QHBoxLayout()
        title_label = QLabel("dontknow ✦ Advanced Neural Vision AI & Story Generator")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #f8fafc;")

        self.badge_status = QLabel("Ready")
        self.badge_status.setStyleSheet("background-color: #0369a1; color: #e0f2fe; padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: 600;")

        header.addWidget(title_label)
        header.addStretch()
        header.addWidget(self.badge_status)
        main_layout.addLayout(header)

        # Main Splitter
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter, 1)

        # Left Panel Widget
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(10)

        # 1. Video Selection Card
        box_input = QGroupBox("Video File Input")
        lay_input = QVBoxLayout(box_input)
        self.drop_area = DropAreaWidget()
        self.drop_area.file_dropped.connect(self.on_file_selected)
        lay_input.addWidget(self.drop_area)

        self.label_video_info = QLabel("No video selected")
        self.label_video_info.setWordWrap(True)
        self.label_video_info.setStyleSheet("color: #94a3b8; font-size: 12px;")
        lay_input.addWidget(self.label_video_info)
        left_layout.addWidget(box_input)

        # 2. Options Card
        box_options = QGroupBox("Advanced Neural Vision Options")
        lay_options = QVBoxLayout(box_options)

        # Output Subfolder field
        lay_subfolder = QHBoxLayout()
        lay_subfolder.addWidget(QLabel("PNG Subfolder:"))
        self.input_subfolder = QLineEdit("./frames_output")
        lay_subfolder.addWidget(self.input_subfolder)
        lay_options.addLayout(lay_subfolder)

        # Frame Sampling Selector
        lay_sampling = QHBoxLayout()
        lay_sampling.addWidget(QLabel("Frame Sampling:"))
        self.combo_sampling = QComboBox()
        self.combo_sampling.addItems([
            "Every Frame (1:1 - Full Accuracy)",
            "Every 2nd Frame",
            "Every 5th Frame",
            "Every 10th Frame",
            "1 Frame per Second (Recommended)"
        ])
        self.combo_sampling.setCurrentIndex(0)
        lay_sampling.addWidget(self.combo_sampling)
        lay_options.addLayout(lay_sampling)

        # Prompt Detail Checkbox
        self.chk_detailed_prompts = QCheckBox("Include Visual Style Descriptors (Lighting, Palette, Framing)")
        self.chk_detailed_prompts.setChecked(True)
        lay_options.addWidget(self.chk_detailed_prompts)

        # Story Length Selector
        lay_summary = QHBoxLayout()
        lay_summary.addWidget(QLabel("Story Length:"))
        self.combo_story_len = QComboBox()
        self.combo_story_len.addItems(["Compact (3 sentences)", "Standard (5 sentences)", "Detailed (10 sentences)"])
        self.combo_story_len.setCurrentIndex(1)
        lay_summary.addWidget(self.combo_story_len)
        lay_options.addLayout(lay_summary)

        left_layout.addWidget(box_options)

        # 3. Execution Control Buttons
        lay_btns = QHBoxLayout()
        self.btn_start = QPushButton("▶ Run Advanced Neural Vision & Summarize")
        self.btn_start.setObjectName("btnPrimary")
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self.start_processing)

        self.btn_cancel = QPushButton("✖ Stop")
        self.btn_cancel.setObjectName("btnCancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_processing)

        lay_btns.addWidget(self.btn_start, 2)
        lay_btns.addWidget(self.btn_cancel, 1)
        left_layout.addLayout(lay_btns)

        # 4. Progress & Frame Preview Card
        box_progress = QGroupBox("Real-time Progress & Frame Preview")
        lay_progress = QVBoxLayout(box_progress)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        lay_progress.addWidget(self.progress_bar)

        self.label_progress_status = QLabel("Idle")
        self.label_progress_status.setStyleSheet("color: #38bdf8; font-weight: 500;")
        lay_progress.addWidget(self.label_progress_status)

        # Live frame thumbnail preview
        self.preview_image = QLabel("No preview")
        self.preview_image.setFixedHeight(140)
        self.preview_image.setAlignment(Qt.AlignCenter)
        self.preview_image.setStyleSheet("background-color: #0f172a; border-radius: 6px; border: 1px solid #334155; color: #64748b;")
        lay_progress.addWidget(self.preview_image)

        left_layout.addWidget(box_progress)
        left_layout.addStretch()

        splitter.addWidget(left_widget)

        # Right Panel Widget: Output Views
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(8, 0, 0, 0)

        self.tabs = QTabWidget()

        # Tab 1: Detailed Story & Frame Listing
        tab_story = QWidget()
        lay_tab1 = QVBoxLayout(tab_story)
        self.txt_story = QTextEdit()
        self.txt_story.setReadOnly(True)
        self.txt_story.setPlaceholderText("Detailed frame-by-frame visual descriptions will appear here...")
        lay_tab1.addWidget(self.txt_story)

        lay_story_btns = QHBoxLayout()
        self.btn_copy_story = QPushButton("📋 Copy Text")
        self.btn_copy_story.clicked.connect(self.copy_story_to_clipboard)
        self.btn_save_story = QPushButton("💾 Export Document (.md)")
        self.btn_save_story.clicked.connect(self.save_story_file)
        lay_story_btns.addWidget(self.btn_copy_story)
        lay_story_btns.addWidget(self.btn_save_story)
        lay_story_btns.addStretch()
        lay_tab1.addLayout(lay_story_btns)

        self.tabs.addTab(tab_story, "📖 Detailed Frame Descriptions Document")

        # Tab 2: Frame Descriptions Array Table
        tab_array = QWidget()
        lay_tab2 = QVBoxLayout(tab_array)
        self.table_array = QTableWidget()
        self.table_array.setColumnCount(5)
        self.table_array.setHorizontalHeaderLabels(["Frame #", "Time (s)", "Detailed Frame Description", "Style Tags", "Image File"])
        self.table_array.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        lay_tab2.addWidget(self.table_array)

        lay_array_btns = QHBoxLayout()
        self.btn_export_json = QPushButton("💾 Export Frame Array JSON")
        self.btn_export_json.clicked.connect(self.export_json_array)
        self.btn_open_folder = QPushButton("📁 Open PNG Subfolder")
        self.btn_open_folder.clicked.connect(self.open_png_subfolder)
        lay_array_btns.addWidget(self.btn_export_json)
        lay_array_btns.addWidget(self.btn_open_folder)
        lay_array_btns.addStretch()
        lay_tab2.addLayout(lay_array_btns)

        self.tabs.addTab(tab_array, "📊 Frame Descriptions Table")

        # Tab 3: Live Execution Log Stream
        tab_log = QWidget()
        lay_tab3 = QVBoxLayout(tab_log)
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setPlaceholderText("Execution log messages will stream here live...")
        lay_tab3.addWidget(self.txt_log)

        self.tabs.addTab(tab_log, "📜 Live Execution Log")

        right_layout.addWidget(self.tabs)
        splitter.addWidget(right_widget)

        splitter.setSizes([450, 730])

    def on_file_selected(self, filepath: str):
        if not os.path.exists(filepath):
            return
        self.selected_video_path = filepath
        filename = os.path.basename(filepath)
        folder_name = os.path.splitext(filename)[0]
        
        video_dir = os.path.dirname(filepath)
        default_subfolder = os.path.join(video_dir, f"frames_{folder_name}")
        self.input_subfolder.setText(default_subfolder)

        try:
            vp = VideoProcessor(filepath)
            meta = vp.get_metadata()
            self.label_video_info.setText(
                f"<b>File:</b> {filename}<br>"
                f"<b>Resolution:</b> {meta['width']}x{meta['height']} | <b>FPS:</b> {meta['fps']:.2f}<br>"
                f"<b>Total Frames:</b> {meta['total_frames']} | <b>Duration:</b> {meta['duration_sec']:.1f}s"
            )
            self.btn_start.setEnabled(True)
            self.drop_area.label_text.setText(f"Selected: {filename}\n(Click/Drag to change)")
        except Exception as e:
            self.label_video_info.setText(f"<font color='#ef4444'>Error loading video: {e}</font>")
            self.btn_start.setEnabled(False)

    def start_processing(self):
        if not self.selected_video_path or not os.path.exists(self.selected_video_path):
            QMessageBox.warning(self, "No Video", "Please select a valid video file first.")
            return

        subfolder = self.input_subfolder.text().strip()
        if not subfolder:
            QMessageBox.warning(self, "Invalid Path", "Please specify a subfolder path for PNG frames.")
            return

        sampling_idx = self.combo_sampling.currentIndex()
        if sampling_idx == 0:
            frame_step = 1
        elif sampling_idx == 1:
            frame_step = 2
        elif sampling_idx == 2:
            frame_step = 5
        elif sampling_idx == 3:
            frame_step = 10
        elif sampling_idx == 4:
            try:
                vp = VideoProcessor(self.selected_video_path)
                fps = vp.get_metadata().get("fps", 30.0)
                frame_step = int(round(fps))
            except Exception:
                frame_step = 30
        else:
            frame_step = 1

        detailed = self.chk_detailed_prompts.isChecked()
        
        story_len_idx = self.combo_story_len.currentIndex()
        story_sentences = 3 if story_len_idx == 0 else (5 if story_len_idx == 1 else 10)

        # UI state updates
        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.badge_status.setText("Processing")
        self.badge_status.setStyleSheet("background-color: #d97706; color: #fef3c7; padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: 600;")
        self.txt_log.clear()
        self.txt_story.clear()
        self.table_array.setRowCount(0)
        self.progress_bar.setValue(0)

        # Start Worker Thread
        self.current_worker = ProcessingWorker(
            video_path=self.selected_video_path,
            output_subfolder=subfolder,
            frame_step=frame_step,
            detailed_prompts=detailed,
            sentences_count=story_sentences
        )

        self.current_worker.progress_signal.connect(self.on_worker_progress)
        self.current_worker.log_signal.connect(self.on_worker_log)
        self.current_worker.frame_preview_signal.connect(self.on_worker_preview)
        self.current_worker.finished_signal.connect(self.on_worker_finished)
        self.current_worker.error_signal.connect(self.on_worker_error)

        self.current_worker.start()

    def cancel_processing(self):
        if self.current_worker and self.current_worker.isRunning():
            self.current_worker.cancel()
            self.badge_status.setText("Stopping...")
            self.badge_status.setStyleSheet("background-color: #dc2626; color: #fee2e2; padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: 600;")

    @Slot(int, int, str)
    def on_worker_progress(self, current, total, text):
        self.progress_bar.setValue(current)
        self.label_progress_status.setText(text)

    @Slot(str)
    def on_worker_log(self, text):
        self.txt_log.append(text)

    @Slot(str)
    def on_worker_preview(self, path):
        if os.path.exists(path):
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                scaled = pixmap.scaled(self.preview_image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.preview_image.setPixmap(scaled)

    @Slot(dict)
    def on_worker_finished(self, results):
        self.last_results = results
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.badge_status.setText("Completed")
        self.badge_status.setStyleSheet("background-color: #15803d; color: #dcfce7; padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: 600;")

        # Populate Document Tab
        summary_result = results.get("summary_result", {})
        story_md = summary_result.get("story_markdown", "No content generated.")
        self.txt_story.setMarkdown(story_md)

        # Populate Frame Descriptions Table Tab
        raw_prompts_array = results.get("raw_prompts_array", [])
        self.table_array.setRowCount(len(raw_prompts_array))
        for row, item in enumerate(raw_prompts_array):
            self.table_array.setItem(row, 0, QTableWidgetItem(str(item.get("frame_index", ""))))
            self.table_array.setItem(row, 1, QTableWidgetItem(f"{item.get('timestamp_sec', 0.0):.2f}s"))
            self.table_array.setItem(row, 2, QTableWidgetItem(item.get("image_prompt", "")))
            tags = ", ".join(item.get("style_tags", []))
            self.table_array.setItem(row, 3, QTableWidgetItem(tags))
            self.table_array.setItem(row, 4, QTableWidgetItem(item.get("filename", "")))

        # Switch to Document Tab
        self.tabs.setCurrentIndex(0)
        QMessageBox.information(self, "Success", f"Advanced neural frame analysis completed!\nProcessed {results['extracted_frames_count']} frame PNGs into:\n{results['output_subfolder']}")

    @Slot(str)
    def on_worker_error(self, err_msg):
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.badge_status.setText("Error")
        self.badge_status.setStyleSheet("background-color: #dc2626; color: #fee2e2; padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: 600;")
        QMessageBox.critical(self, "Processing Error", f"An error occurred:\n{err_msg}")

    def copy_story_to_clipboard(self):
        text = self.txt_story.toPlainText()
        if text:
            QApplication.clipboard().setText(text)
            QMessageBox.information(self, "Copied", "Text copied to clipboard!")

    def save_story_file(self):
        text = self.txt_story.toPlainText()
        if not text:
            return
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Detailed Descriptions", "frame_descriptions.md", "Markdown (*.md);;Text (*.txt)")
        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(text)
            QMessageBox.information(self, "Saved", f"Document saved to:\n{save_path}")

    def export_json_array(self):
        if not self.last_results:
            return
        save_path, _ = QFileDialog.getSaveFileName(self, "Export Frame Descriptions JSON", "frame_descriptions_array.json", "JSON (*.json)")
        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(self.last_results.get("raw_prompts_array", []), f, indent=2, ensure_ascii=False)
            QMessageBox.information(self, "Saved", f"JSON array saved to:\n{save_path}")

    def open_png_subfolder(self):
        if self.last_results and "output_subfolder" in self.last_results:
            folder = self.last_results["output_subfolder"]
            if os.path.exists(folder):
                os.startfile(folder)


def main():
    app = QApplication(sys.argv)
    window = DontKnowApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
