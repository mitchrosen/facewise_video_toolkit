from __future__ import annotations

import argparse
from cProfile import label
import json
import sys
from pathlib import Path

import av
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)

class VideoReader:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.container = av.open(str(self.path))
        self.stream = self.container.streams.video[0]
        self.fps = float(self.stream.average_rate or 30.0)
        self.frame_count = int(self.stream.frames or 0)
        self.current_frame_idx = 0
        self._iter = self.container.decode(video=0)

    def close(self) -> None:
        self.container.close()

    def seek(self, frame_idx: int) -> None:
        frame_idx = max(0, int(frame_idx))
        timestamp = int(frame_idx / self.fps / self.stream.time_base)
        self.container.seek(timestamp, stream=self.stream)
        self._iter = self.container.decode(video=0)
        self.current_frame_idx = frame_idx

    def frame_at_exact(self, frame_idx: int):
        # Reliable, not optimized: reopen and decode forward to exact frame.
        # Good enough for demo stepping/scrubbing.
        frame_idx = max(0, int(frame_idx))
        self.container.close()
        self.container = av.open(str(self.path))
        self.stream = self.container.streams.video[0]
        self._iter = self.container.decode(video=0)

        img = None
        for i, frame in enumerate(self._iter):
            if i == frame_idx:
                img = frame.to_ndarray(format="rgb24")
                self.current_frame_idx = frame_idx + 1
                break

        return img

    def next_frame(self):
        try:
            frame = next(self._iter)
        except StopIteration:
            return None

        img = frame.to_ndarray(format="rgb24")
        self.current_frame_idx += 1
        return img

class FacewiseJsonIndex:
    def __init__(self, payload: dict):
        self.by_frame: dict[int, list[dict]] = {}

        video = payload.get("video", {})
        video_size = video.get("size") or [100, 100]
        video_w = float(video_size[0])
        video_h = float(video_size[1])

        for shot in payload.get("shots", []):
            shot_number = shot.get("shot_number")

            for track in shot.get("face_tracks", []):
                global_id = track.get("global_id")
                segment_id = track.get("segment_id")
                track_id = track.get("track_id")
                face_label = track.get("face_label")

                observations = track.get("observations") or []

                for obs in observations:
                    frame_idx = int(obs.get("frame_idx", obs.get("f", -1)))
                    bbox = obs.get("bbox") or obs.get("bbox_xyxy")
                    if frame_idx < 0 or not bbox or len(bbox) != 4:
                        continue

                    self.by_frame.setdefault(frame_idx, []).append(
                        {
                            "bbox": [float(v) for v in bbox],
                            "global_id": global_id,
                            "segment_id": segment_id,
                            "track_id": track_id,
                            "shot_number": shot_number,
                            "face_label": face_label,
                        }
                    )

                if observations:
                    continue

                first_frame = int(track.get("first_frame", -1))
                last_frame = int(track.get("last_frame", -1))

                cx = track.get("avg_center_x")
                cy = track.get("avg_center_y")
                bw = track.get("avg_face_width")
                bh = track.get("avg_face_height")

                if (
                    first_frame < 0
                    or last_frame < first_frame
                    or cx is None
                    or cy is None
                    or bw is None
                    or bh is None
                ):
                    continue

                # These Facewise summary fields are normalized percentages.
                x1 = (float(cx) - float(bw) / 2.0) / 100.0 * video_w
                y1 = (float(cy) - float(bh) / 2.0) / 100.0 * video_h
                x2 = (float(cx) + float(bw) / 2.0) / 100.0 * video_w
                y2 = (float(cy) + float(bh) / 2.0) / 100.0 * video_h

                for frame_idx in range(first_frame, last_frame + 1):
                    self.by_frame.setdefault(frame_idx, []).append(
                        {
                            "bbox": [x1, y1, x2, y2],
                            "global_id": global_id,
                            "segment_id": segment_id,
                            "track_id": track_id,
                            "shot_number": shot_number,
                            "face_label": face_label,
                        }
                    )

    def faces_at(self, frame_idx: int) -> list[dict]:
        return self.by_frame.get(int(frame_idx), [])

class Viewer(QMainWindow):
    def __init__(self, video_path: Path | None):
        super().__init__()

        self.reader: VideoReader | None = None
        self.current_pixmap = None
        self.last_open_dir = Path.home()
        self._slider_dragging = False
        self.phone_display_long_side = 650
        self.is_portrait = True

        self.video_label = QLabel("Open a video to begin")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedSize(760, 760)

        self.drawer_visible = False
        self.drawer_button = QPushButton("⌃", self.video_label)
        self.drawer_button.setFlat(True)
        self.drawer_button.setStyleSheet(
            "QPushButton { font-size: 28px; color: white; "
            "background: rgba(0, 0, 0, 80); border: none; }"
        )
        self.drawer_button.clicked.connect(self.toggle_drawer)
        self.drawer_button.resize(72, 36)
        self.drawer_button.hide()

        self.drawer_panel = QFrame(self.video_label)
        self.drawer_panel.setStyleSheet(
            "QFrame { background: rgba(0, 0, 0, 130); border: none; }"
        )
        self.drawer_panel.hide()

        self.drawer_face_labels = [
            QLabel("Face 1", self.drawer_panel),
            QLabel("Face 2", self.drawer_panel),
        ]
        for label in self.drawer_face_labels:
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet(
                "QLabel { color: white; background: rgba(255, 255, 255, 30); }"
            )

        self.drawer_layout = None

        self.open_button = QPushButton("Open…")
        self.open_button.clicked.connect(self.open_video)

        self.open_json_button = QPushButton("Open JSON…")
        self.open_json_button.clicked.connect(self.open_json)
        self.json_path: Path | None = None
        self.facewise_json: dict | None = None

        self.start_button = QPushButton("|<<")
        self.start_button.clicked.connect(self.go_to_start)

        self.orientation_button = QPushButton("Landscape ▭")
        self.orientation_button.clicked.connect(self.toggle_orientation)

        # self.prev_button = QPushButton("<<")
        # self.prev_button.clicked.connect(self.prev_frame)
        # self.prev_button.setAutoRepeat(True)
        # self.prev_button.setAutoRepeatDelay(300)
        # self.prev_button.setAutoRepeatInterval(60)

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_play)

        self.next_button = QPushButton(">>")
        self.next_button.clicked.connect(self.step_once)
        self.next_button.setAutoRepeat(True)
        self.next_button.setAutoRepeatDelay(300)
        self.next_button.setAutoRepeatInterval(60)

        self.end_button = QPushButton(">>|")
        self.end_button.clicked.connect(self.go_to_end)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.sliderPressed.connect(self.begin_slider_drag)
        self.slider.sliderReleased.connect(self.end_slider_drag)

        self.auto_framing = False
        self.framing_button = QPushButton("Auto Framing")
        self.framing_button.clicked.connect(self.toggle_framing)

        self.show_boxes = True
        self.boxes_button = QPushButton("Hide Boxes")
        self.boxes_button.clicked.connect(self.toggle_boxes)

        self.zoom_out_button = QPushButton("Zoom -")
        self.zoom_out_button.clicked.connect(self.zoom_out)

        self.zoom_in_button = QPushButton("Zoom +")
        self.zoom_in_button.clicked.connect(self.zoom_in)

        self.pan_up_button = QPushButton("↑")
        self.pan_up_button.clicked.connect(lambda: self.pan(0.0, -0.1))

        self.pan_left_button = QPushButton("←")
        self.pan_left_button.clicked.connect(lambda: self.pan(-0.1, 0.0))

        self.pan_right_button = QPushButton("→")
        self.pan_right_button.clicked.connect(lambda: self.pan(0.1, 0.0))

        self.pan_down_button = QPushButton("↓")
        self.pan_down_button.clicked.connect(lambda: self.pan(0.0, 0.1))

        pan_layout = QGridLayout()
        pan_layout.addWidget(self.pan_up_button, 0, 1)
        pan_layout.addWidget(self.pan_left_button, 1, 0)
        pan_layout.addWidget(self.pan_right_button, 1, 2)
        pan_layout.addWidget(self.pan_down_button, 2, 1)

        pan_widget = QWidget()
        pan_widget.setLayout(pan_layout)

        self.manual_mode = False
        self.manual_zoom = 1.0
        self.manual_crop: tuple[float, float, float, float] | None = None
        self.manual_center: tuple[float, float] | None = None
        self.last_visible_center: tuple[float, float] | None = None
        self.manual_pan_x = 0.0
        self.manual_pan_y = 0.0

        self.face_index: FacewiseJsonIndex | None = None
        self.displayed_frame_idx = 0

        playback_controls = QHBoxLayout()
        playback_controls.addWidget(self.open_button)
        playback_controls.addWidget(self.open_json_button)
        playback_controls.addWidget(self.start_button)
        # playback_controls.addWidget(self.prev_button)
        playback_controls.addWidget(self.play_button)
        playback_controls.addWidget(self.next_button)
        playback_controls.addWidget(self.end_button)
        playback_controls.addWidget(self.slider, stretch=1)

        framing_controls = QHBoxLayout()
        framing_controls.addStretch(1)
        framing_controls.addWidget(self.orientation_button)
        framing_controls.addWidget(self.framing_button)
        framing_controls.addWidget(self.boxes_button)
        framing_controls.addWidget(self.zoom_out_button)
        framing_controls.addWidget(self.zoom_in_button)
        framing_controls.addWidget(pan_widget)
        framing_controls.addStretch(1)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label, alignment=Qt.AlignCenter)
        layout.addLayout(playback_controls)
        layout.addLayout(framing_controls)

        root = QWidget()
        root.setLayout(layout)
        self.setCentralWidget(root)

        self.timer = QTimer()
        self.timer.timeout.connect(self.step_playback)

        self.setWindowTitle("Facewise Demo Viewer")
        self.set_controls_enabled(False)

        if video_path is not None:
            self.load_video(video_path)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.render_current_pixmap()

    def showEvent(self, event):
        super().showEvent(event)
        self.update_drawer_faces()

    def set_controls_enabled(self, enabled: bool):
        self.start_button.setEnabled(enabled)
        # self.prev_button.setEnabled(enabled)
        self.play_button.setEnabled(enabled)
        self.next_button.setEnabled(enabled)
        self.end_button.setEnabled(enabled)
        self.slider.setEnabled(enabled)

    def set_stopped(self):
        self.timer.stop()
        self.play_button.setText("Play")
        # self.prev_button.setEnabled(self.reader is not None)
        self.next_button.setEnabled(self.reader is not None)

    def toggle_play(self):
        if self.reader is None:
            return

        if self.timer.isActive():
            self.set_stopped()
        else:
            self.timer.start()
            self.play_button.setText("Pause")
            # self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video",
            str(self.last_open_dir),
            "Video Files (*.mp4 *.mov *.m4v *.avi);;All Files (*)",
        )
        if path:
            self.load_video(Path(path))

    def load_video(self, path: Path):
        self.set_stopped()

        if self.reader is not None:
            self.reader.close()

        self.reader = VideoReader(path)
        self.last_open_dir = path.parent
        self.current_pixmap = None

        self.timer.setInterval(max(1, int(1000 / self.reader.fps)))
        self.slider.setMaximum(max(0, self.reader.frame_count - 1))
        self.slider.setValue(0)

        self.set_controls_enabled(True)
        self.setWindowTitle(f"Facewise Demo Viewer — {path.name}")

        QTimer.singleShot(0, self.go_to_start)

    def open_json(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Facewise output_global_json",
            str(self.last_open_dir),
            "JSON Files (*.json);;All Files (*)",
        )
        if path:
            self.load_json(Path(path))

    def load_json(self, path: Path):
        self.json_path = path
        self.facewise_json = json.loads(path.read_text())
        self.face_index = FacewiseJsonIndex(self.facewise_json)

        title = self.windowTitle()
        self.setWindowTitle(f"{title} — JSON: {path.name}")

        self.render_current_pixmap()

    def go_to_start(self):
        if self.reader is None:
            return
        self.set_stopped()
        img = self.reader.frame_at_exact(0)
        if img is not None:
            self.show_frame(img)

    def go_to_end(self):
        if self.reader is None:
            return
        self.set_stopped()
        last = max(0, self.reader.frame_count - 1)
        self.reader.seek(last)
        img = self.reader.next_frame()
        if img is not None:
            self.show_frame(img)

    # def prev_frame(self):
    #     if self.reader is None:
    #         return
    #     self.set_stopped()
    #     target = max(0, self.reader.current_frame_idx - 2)
    #     self.reader.seek(target)
    #     img = self.reader.next_frame()
    #     if img is not None:
    #         self.show_frame(img)

    def step_once(self):
        if self.reader is None:
            return
        self.set_stopped()
        self.step_playback()

    def step_playback(self):
        if self.reader is None:
            return

        img = self.reader.next_frame()
        if img is None:
            self.set_stopped()
            return

        self.show_frame(img)

    def begin_slider_drag(self):
        self._slider_dragging = True

    def end_slider_drag(self):
        if self.reader is None:
            return

        target = self.slider.value()
        self.reader.seek(target)
        img = self.reader.next_frame()
        if img is not None:
            self.show_frame(img)

        self._slider_dragging = False

    def show_frame(self, img):
        self.displayed_frame_idx = max(0, int(self.reader.current_frame_idx) - 1)

        h, w, ch = img.shape
        qimg = QImage(img.data, w, h, ch * w, QImage.Format_RGB888).copy()
        self.current_pixmap = QPixmap.fromImage(qimg)
        self.render_current_pixmap()
        self.update_drawer_faces()

        if not self._slider_dragging:
            self.slider.blockSignals(True)
            self.slider.setValue(min(self.reader.current_frame_idx - 1, self.slider.maximum()))
            self.slider.blockSignals(False)

    def render_current_pixmap(self):
        if self.current_pixmap is None:
            return

        viewport_w, viewport_h = self.viewport_size()

        canvas = QPixmap(viewport_w, viewport_h)
        canvas.fill(Qt.black)

        src = self.current_pixmap.copy()

        painter = QPainter(src)
        if self.show_boxes and self.face_index is not None:
            faces = self.face_index.faces_at(self.displayed_frame_idx)

            pen = QPen(Qt.red)
            pen.setWidth(3)
            painter.setPen(pen)

            for face in faces:
                x1, y1, x2, y2 = face["bbox"]
                painter.drawRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                label = face.get("face_label") or f"g={face.get('global_id')} s={face.get('segment_id')}"
                painter.drawText(int(x1), max(12, int(y1) - 6), label)
        painter.end()

        target_face = self.auto_frame() if self.auto_framing and not self.manual_mode else None

        manual_zoom = self.manual_zoom if self.manual_mode else 1.0

        crop = None

        if self.manual_mode and self.manual_crop is not None:
            crop = self.manual_crop
        elif target_face is not None:
            crop = self.crop_for_face(target_face, self.viewport_aspect())

        if crop is None:
            scaled = src.scaled(
                viewport_w,
                viewport_h,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )

            if manual_zoom != 1.0:
                scaled = scaled.scaled(
                    int(scaled.width() * manual_zoom),
                    int(scaled.height() * manual_zoom),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )

            x = (viewport_w - scaled.width()) // 2
            y = (viewport_h - scaled.height()) // 2

            if self.manual_mode and manual_zoom != 1.0:
                overflow_x = max(0, scaled.width() - viewport_w)
                overflow_y = max(0, scaled.height() - viewport_h)
                x -= int(self.manual_pan_x * overflow_x)
                y -= int(self.manual_pan_y * overflow_y)

            painter = QPainter(canvas)
            painter.drawPixmap(x, y, scaled)
            painter.end()
        else:
            crop_x1, crop_y1, crop_x2, crop_y2 = crop

            source_w = src.width()
            source_h = src.height()

            base_crop_w = max(1.0, crop_x2 - crop_x1)
            base_crop_h = max(1.0, crop_y2 - crop_y1)

            if self.manual_mode and self.manual_center is not None:
                crop_center_x, crop_center_y = self.manual_center
            else:
                crop_center_x = (crop_x1 + crop_x2) / 2.0
                crop_center_y = (crop_y1 + crop_y2) / 2.0

            effective_crop_w = base_crop_w / manual_zoom
            effective_crop_h = base_crop_h / manual_zoom

            effective_aspect = effective_crop_w / effective_crop_h
            viewport_aspect = self.viewport_aspect()
            if effective_aspect < viewport_aspect:
                effective_crop_w = effective_crop_h * viewport_aspect
            else:
                effective_crop_h = effective_crop_w / viewport_aspect

            crop_x1 = crop_center_x - effective_crop_w / 2.0
            crop_y1 = crop_center_y - effective_crop_h / 2.0
            crop_x2 = crop_center_x + effective_crop_w / 2.0
            crop_y2 = crop_center_y + effective_crop_h / 2.0

            if crop_x1 < 0.0:
                crop_x2 -= crop_x1
                crop_x1 = 0.0
            if crop_y1 < 0.0:
                crop_y2 -= crop_y1
                crop_y1 = 0.0
            if crop_x2 > float(source_w):
                shift = crop_x2 - float(source_w)
                crop_x1 -= shift
                crop_x2 = float(source_w)
            if crop_y2 > float(source_h):
                shift = crop_y2 - float(source_h)
                crop_y1 -= shift
                crop_y2 = float(source_h)

            crop_x1 = max(0.0, crop_x1)
            crop_y1 = max(0.0, crop_y1)
            crop_x2 = min(float(source_w), crop_x2)
            crop_y2 = min(float(source_h), crop_y2)

            self.last_visible_center = (
                (crop_x1 + crop_x2) / 2.0,
                (crop_y1 + crop_y2) / 2.0,
            )

            crop_w = max(1, int(crop_x2 - crop_x1))
            crop_h = max(1, int(crop_y2 - crop_y1))

            cropped = src.copy(
                int(crop_x1),
                int(crop_y1),
                crop_w,
                crop_h,
            )

            scaled = cropped.scaled(
                viewport_w,
                viewport_h,
                Qt.KeepAspectRatioByExpanding,
                Qt.SmoothTransformation,
            )

            draw_x = (viewport_w - scaled.width()) // 2
            draw_y = (viewport_h - scaled.height()) // 2

            painter = QPainter(canvas)
            painter.drawPixmap(draw_x, draw_y, scaled)
            painter.end()

        display_long_side = self.phone_display_long_side

        if canvas.width() >= canvas.height():
            display_w = display_long_side
            display_h = round(display_w * canvas.height() / canvas.width())
        else:
            display_h = display_long_side
            display_w = round(display_h * canvas.width() / canvas.height())

        self.video_label.setPixmap(
            canvas.scaled(
                display_w,
                display_h,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )
        self.position_drawer_button(display_w, display_h)
        self.update_drawer_faces()
    
    def toggle_orientation(self):
        """
        Switch the orientation of the viewer between portrait and landscape.

        Maintain, where possible, the center of the currently visible source region
        to be the center of the switched-to orientation when toggling.

        Boundary corrections may cause the displayed center to shift.
        """
        if self.manual_mode and self.last_visible_center is not None:
            self.manual_center = self.last_visible_center

        self.is_portrait = not self.is_portrait
        self.orientation_button.setText("Landscape ▭" if self.is_portrait else "Portrait ▯")
        self.render_current_pixmap()
        self.update_drawer_faces()

    def toggle_drawer(self):
        self.drawer_visible = not self.drawer_visible
        self.position_drawer_button_for_current_orientation()
        self.update_drawer_faces()

    def rebuild_drawer_layout(self):
        # Drawer preview labels are explicitly positioned in
        # position_drawer_button(). A Qt layout here can leave both labels at
        # (0, 0) during rapid orientation/frame updates, causing thumbnails to
        # render on top of each other.
        return

    def position_drawer_button_for_current_orientation(self):
        if self.current_pixmap is None:
            return

        viewport_w, viewport_h = self.viewport_size()
        if viewport_w >= viewport_h:
            display_w = self.phone_display_long_side
            display_h = round(display_w * viewport_h / viewport_w)
        else:
            display_h = self.phone_display_long_side
            display_w = round(display_h * viewport_w / viewport_h)

        self.position_drawer_button(display_w, display_h)

    def position_drawer_button(self, display_w: int, display_h: int):
        label_w = self.video_label.width()
        label_h = self.video_label.height()

        phone_x = (label_w - display_w) // 2
        phone_y = (label_h - display_h) // 2

        if self.is_portrait:
            drawer_h = 150
            self.drawer_panel.setGeometry(
                phone_x,
                phone_y + display_h - drawer_h,
                display_w,
                drawer_h,
            )
            self.drawer_button.setText("⌃" if not self.drawer_visible else "⌄")
            self.drawer_button.resize(72, 36)
            self.drawer_button.move(
                phone_x + (display_w - self.drawer_button.width()) // 2,
                phone_y + display_h - self.drawer_button.height(),
            )

            margin = 8
            spacing = 8
            label_w = max(1, (display_w - 2 * margin - spacing) // 2)
            label_h = max(1, drawer_h - 2 * margin)
            for idx, label in enumerate(self.drawer_face_labels):
                label.setGeometry(
                    margin + idx * (label_w + spacing),
                    margin,
                    label_w,
                    label_h,
                )
        else:
            drawer_w = 180
            self.drawer_panel.setGeometry(
                phone_x + display_w - drawer_w,
                phone_y,
                drawer_w,
                display_h,
            )
            self.drawer_button.setText("‹" if not self.drawer_visible else "›")
            self.drawer_button.resize(36, 72)
            self.drawer_button.move(
                phone_x + display_w - self.drawer_button.width(),
                phone_y + (display_h - self.drawer_button.height()) // 2,
            )

            margin = 8
            spacing = 8
            label_w = max(1, drawer_w - 2 * margin)
            label_h = max(1, (display_h - 2 * margin - spacing) // 2)
            for idx, label in enumerate(self.drawer_face_labels):
                label.setGeometry(
                    margin,
                    margin + idx * (label_h + spacing),
                    label_w,
                    label_h,
                )

        for label in self.drawer_face_labels:
            label.setFixedSize(label_w, label_h)

        self.drawer_panel.setVisible(self.drawer_visible)
        self.drawer_panel.raise_()
        self.drawer_button.raise_()
        self.drawer_button.show()
        self.update_drawer_faces()

    def viewport_size(self) -> tuple[int, int]:
        return (750, 1334) if self.is_portrait else (1334, 750)
    
    def viewport_aspect(self) -> float:
        viewport_w, viewport_h = self.viewport_size()
        return float(viewport_w) / float(viewport_h)
    
    def toggle_boxes(self):
        self.show_boxes = not self.show_boxes
        self.boxes_button.setText("Hide Boxes" if self.show_boxes else "Show Boxes")
        self.render_current_pixmap()

    def toggle_framing(self):
        if self.auto_framing or self.manual_mode:
            self.auto_framing = False
            self.reset_manual_mode()
            self.framing_button.setText("Auto Frame")
        else:
            self.auto_framing = True
            self.reset_manual_mode()
            self.framing_button.setText("Fit Video")

        self.render_current_pixmap()

    def reset_manual_mode(self):
        self.manual_mode = False
        self.manual_zoom = 1.0
        self.manual_pan_x = 0.0
        self.manual_pan_y = 0.0
        self.manual_crop = None
        self.manual_center = None

    def crop_for_face(
        self,
        face,
        viewport_aspect: float | None = None,
        buffer: float = 2.0,
    ):
        x1, y1, x2, y2 = face["bbox"]

        face_w = max(1.0, float(x2) - float(x1))
        face_h = max(1.0, float(y2) - float(y1))

        crop_w = face_w * buffer
        crop_h = face_h * buffer

        if viewport_aspect is not None:
            crop_aspect = crop_w / crop_h
            if crop_aspect < viewport_aspect:
                crop_w = crop_h * viewport_aspect
            else:
                crop_h = crop_w / viewport_aspect

        cx = (float(x1) + float(x2)) / 2.0
        cy = (float(y1) + float(y2)) / 2.0

        return (
            cx - crop_w / 2.0,
            cy - crop_h / 2.0,
            cx + crop_w / 2.0,
            cy + crop_h / 2.0,
        )

    def auto_frame(self):
        if self.face_index is None:
            return None

        faces = self.face_index.faces_at(self.displayed_frame_idx)
        if not faces:
            return None
        
        return max(
            faces,
            key=lambda face: (
                float(face["bbox"][2]) - float(face["bbox"][0])
            )
            * (
                float(face["bbox"][3]) - float(face["bbox"][1])
            ),
        )
    
    def largest_faces_for_current_frame(self, limit: int = 2) -> list[dict]:
        """
        Return up to `limit` faces for the current frame, sorted by
        descending face area.

        The returned list may contain fewer than `limit` faces when there are fewer
        faces for this framethan the limit.
        """
        if self.face_index is None:
            return []

        faces = self.face_index.faces_at(self.displayed_frame_idx)
        return sorted(
            faces,
            key=lambda face: (
                float(face["bbox"][2]) - float(face["bbox"][0])
            )
            * (
                float(face["bbox"][3]) - float(face["bbox"][1])
            ),
            reverse=True,
        )[:limit]

    def update_drawer_faces(self):
        """
        Populate drawer preview widgets with the largest currently
        visible faces, up to a maximum of two.

        Face crops intentionally include substantial padding around
        the detected face rectangle so previews appear as head-and-
        shoulders views rather than tightly cropped facial features.
        """
        if self.current_pixmap is None:
            return

        faces = self.largest_faces_for_current_frame(limit=2)

        for idx, label in enumerate(self.drawer_face_labels):
            if idx >= len(faces):
                label.clear()
                label.hide()
                continue

            face = faces[idx]
            x1, y1, x2, y2 = face["bbox"]

            face_w = max(1.0, float(x2) - float(x1))
            face_h = max(1.0, float(y2) - float(y1))

            pad_x = face_w * 0.75
            pad_y = face_h * 0.75

            crop_x1 = max(0, int(float(x1) - pad_x))
            crop_y1 = max(0, int(float(y1) - pad_y))
            crop_x2 = min(
                self.current_pixmap.width(),
                int(float(x2) + pad_x),
            )
            crop_y2 = min(
                self.current_pixmap.height(),
                int(float(y2) + pad_y),
            )

            crop_w = max(1, crop_x2 - crop_x1)
            crop_h = max(1, crop_y2 - crop_y1)

            face_pix = self.current_pixmap.copy(
                crop_x1,
                crop_y1,
                crop_w,
                crop_h,
            )

            label.show()
            label.setPixmap(
                face_pix.scaled(
                    max(1, label.width()),
                    max(1, label.height()),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )

            for label in self.drawer_face_labels:
                label.setAlignment(Qt.AlignCenter)

    def zoom_in(self):
        self.enter_manual_mode()
        self.manual_zoom = min(6.0, self.manual_zoom * 1.15)
        self.clamp_manual_center()
        self.render_current_pixmap()

    def zoom_out(self):
        self.enter_manual_mode()
        self.manual_zoom = max(self.minimum_manual_zoom(), self.manual_zoom / 1.15)
        self.clamp_manual_center()
        self.render_current_pixmap()

    def enter_manual_mode(self):
        
        if self.manual_crop is None:
            if self.auto_framing:
                face = self.auto_frame()
                if face is not None:
                    self.manual_crop = self.crop_for_face(face, self.viewport_aspect())

        if self.manual_crop is not None and self.manual_center is None:
            x1, y1, x2, y2 = self.manual_crop
            self.manual_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

        self.manual_mode = True
        self.framing_button.setText("Fit Video")

    def minimum_manual_zoom(self) -> float:
        if self.manual_crop is None or self.current_pixmap is None:
            return 1.0

        crop_x1, crop_y1, crop_x2, crop_y2 = self.manual_crop
        crop_w = max(1.0, crop_x2 - crop_x1)
        crop_h = max(1.0, crop_y2 - crop_y1)

        source_w = max(1.0, float(self.current_pixmap.width()))
        source_h = max(1.0, float(self.current_pixmap.height()))

        viewport_aspect = self.viewport_aspect()
        fit_crop_w = source_w
        fit_crop_h = source_w / viewport_aspect

        if fit_crop_h > source_h:
            fit_crop_h = source_h
            fit_crop_w = source_h * viewport_aspect

        return min(crop_w / fit_crop_w, crop_h / fit_crop_h)
    
    def clamp_manual_center(self):
        if self.current_pixmap is None:
            return

        if self.manual_crop is None:
            self.manual_pan_x = max(-0.5, min(0.5, self.manual_pan_x))
            self.manual_pan_y = max(-0.5, min(0.5, self.manual_pan_y))
            return

        if self.manual_center is None:
            return

        x1, y1, x2, y2 = self.manual_crop
        crop_w = max(1.0, x2 - x1)
        crop_h = max(1.0, y2 - y1)

        visible_w = crop_w / max(0.001, self.manual_zoom)
        visible_h = crop_h / max(0.001, self.manual_zoom)

        source_w = float(self.current_pixmap.width())
        source_h = float(self.current_pixmap.height())

        cx, cy = self.manual_center

        if visible_w >= source_w:
            cx = source_w / 2.0
        else:
            cx = max(visible_w / 2.0, min(source_w - visible_w / 2.0, cx))

        if visible_h >= source_h:
            cy = source_h / 2.0
        else:
            cy = max(visible_h / 2.0, min(source_h - visible_h / 2.0, cy))

        self.manual_center = (cx, cy)

    def pan(self, dx: float, dy: float):
        if not self.auto_framing and not self.manual_mode:
            return
        self.enter_manual_mode()
        if self.manual_crop is None:
            self.manual_pan_x += dx
            self.manual_pan_y += dy
            self.clamp_manual_center()
            self.render_current_pixmap()
            return

        if self.manual_center is None:
            return

        x1, y1, x2, y2 = self.manual_crop
        crop_w = max(1.0, x2 - x1)
        crop_h = max(1.0, y2 - y1)

        cx, cy = self.manual_center
        cx += dx * crop_w / max(1.0, self.manual_zoom)
        cy += dy * crop_h / max(1.0, self.manual_zoom)
        self.manual_center = (cx, cy)
        self.clamp_manual_center()

        self.render_current_pixmap()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    app = QApplication(sys.argv)

    video_path = Path(args.video) if args.video else None
    json_path = Path(args.json) if args.json else None

    viewer = Viewer(video_path)

    if json_path is not None:
        viewer.load_json(json_path)

    viewer.resize(1000, 700)
    viewer.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()