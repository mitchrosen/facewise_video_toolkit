from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass

import av
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QComboBox,
    QMainWindow,
    QPushButton,
    QSlider,
    QFrame,
    QScrollArea,
    QGridLayout,
    QHBoxLayout,
    QVBoxLayout,
    QBoxLayout,
    QWidget,
)

REFERENCE_DIAGONAL = 6.1
REFERENCE_LONG_SIDE = 650

@dataclass(frozen=True)
class DevicePreset:
    name: str
    viewport_width: int
    viewport_height: int
    display_inches: float

IPHONE_PRESETS = [
    DevicePreset("iPhone SE (3rd)",   750, 1334, 4.7),

    DevicePreset("iPhone 15",         1179, 2556, 6.1),
    DevicePreset("iPhone 15 Plus",    1290, 2796, 6.7),
    DevicePreset("iPhone 15 Pro",     1179, 2556, 6.1),
    DevicePreset("iPhone 15 Pro Max", 1290, 2796, 6.7),

    DevicePreset("iPhone 16",         1179, 2556, 6.1),
    DevicePreset("iPhone 16 Plus",    1290, 2796, 6.7),
    DevicePreset("iPhone 16 Pro",     1206, 2622, 6.3),
    DevicePreset("iPhone 16 Pro Max", 1320, 2868, 6.9),

    DevicePreset("iPhone 17",         1206, 2622, 6.3),
    DevicePreset("iPhone Air",        1260, 2736, 6.6),
    DevicePreset("iPhone 17 Pro",     1206, 2622, 6.3),
    DevicePreset("iPhone 17 Pro Max", 1320, 2868, 6.9),
]

class VideoReader:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.container = av.open(str(self.path))
        self.stream = self.container.streams.video[0]
        self.fps = float(self.stream.average_rate or 30.0)
        self.frame_count = int(self.stream.frames or 0)
        self.current_frame_idx = 0
        self._iter = self.container.decode(video=0)
        self._decoder_aligned = True
        self._cache: dict[int, object] = {}
        self._cache_order: list[int] = []
        self._cache_limit = 12

    def close(self) -> None:
        self.container.close()

    def _remember_frame(self, frame_idx: int, img) -> None:
        self._cache[int(frame_idx)] = img
        if frame_idx in self._cache_order:
            self._cache_order.remove(frame_idx)
        self._cache_order.append(frame_idx)

        while len(self._cache_order) > self._cache_limit:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)

    def _cached_frame(self, frame_idx: int):
        img = self._cache.get(int(frame_idx))
        if img is None:
            return None

        if frame_idx in self._cache_order:
            self._cache_order.remove(frame_idx)
        self._cache_order.append(frame_idx)
        return img

    def _frame_index_from_pts(self, frame) -> int | None:
        if frame.pts is None:
            return None

        seconds = float(frame.pts * self.stream.time_base)
        return int(round(seconds * self.fps))

    def seek(self, frame_idx: int) -> None:
        frame_idx = max(0, int(frame_idx))
        timestamp = int(frame_idx / self.fps / self.stream.time_base)
        self.container.seek(timestamp, stream=self.stream)
        self._iter = self.container.decode(video=0)
        self._decoder_aligned = False
        self.current_frame_idx = frame_idx

    def frame_at_exact(self, frame_idx: int, *, use_cache: bool = True):
        frame_idx = max(0, int(frame_idx))
        if self.frame_count:
            frame_idx = min(frame_idx, self.frame_count - 1)

        if use_cache:
            cached = self._cached_frame(frame_idx)
            if cached is not None:
                self.current_frame_idx = frame_idx + 1
                self._decoder_aligned = False
                return cached

        # Seek to the nearest prior keyframe and decode forward.
        timestamp = int(frame_idx / self.fps / self.stream.time_base)
        self.container.seek(
            timestamp,
            stream=self.stream,
            backward=True,
            any_frame=False,
        )
        self._iter = self.container.decode(video=0)

        decoded_idx = None
        fallback_idx = max(0, frame_idx - 60)

        for offset, frame in enumerate(self._iter):
            pts_idx = self._frame_index_from_pts(frame)
            if pts_idx is None:
                if decoded_idx is None:
                    decoded_idx = fallback_idx
                else:
                    decoded_idx += 1
            else:
                decoded_idx = pts_idx

            if decoded_idx < frame_idx:
                continue

            img = frame.to_ndarray(format="rgb24")
            self._remember_frame(frame_idx, img)
            self.current_frame_idx = frame_idx + 1
            self._decoder_aligned = True
            return img

        self._decoder_aligned = False
        return None

    def next_frame(self):
        if self.frame_count and self.current_frame_idx >= self.frame_count:
            return None

        if not self._decoder_aligned:
            return self.frame_at_exact(self.current_frame_idx, use_cache=False)

        try:
            frame = next(self._iter)
        except StopIteration:
            return None

        img = frame.to_ndarray(format="rgb24")
        self.current_frame_idx += 1
        self._remember_frame(self.current_frame_idx - 1, img)
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
        self.device = IPHONE_PRESETS[0]
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

        self.drawer_scroll_area = QScrollArea(self.drawer_panel)
        self.drawer_scroll_area.setWidgetResizable(True)
        self.drawer_scroll_area.setFrameShape(QFrame.NoFrame)
        self.drawer_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.drawer_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.drawer_scroll_area.setStyleSheet(
            """
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollArea > QWidget > QWidget {
                background: transparent;
            }
            """
        )

        self.drawer_contents = QWidget()
        self.drawer_contents.setAttribute(Qt.WA_TranslucentBackground)
        self.drawer_layout = QBoxLayout(QBoxLayout.LeftToRight)
        self.drawer_layout.setContentsMargins(8, 8, 8, 8)
        self.drawer_layout.setSpacing(8)
        self.drawer_contents.setLayout(self.drawer_layout)
        self.drawer_scroll_area.setWidget(self.drawer_contents)

        self.drawer_face_labels: list[QLabel] = []
        self.drawer_thumb_w = 1
        self.drawer_thumb_h = 1

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

        self.device_combo = QComboBox()
        self.device_combo.setMinimumContentsLength(20)
        self.device_combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon
        )

        self.device_combo.addItem(IPHONE_PRESETS[0].name, IPHONE_PRESETS[0])

        self.device_combo.insertSeparator(self.device_combo.count())
        for preset in IPHONE_PRESETS[1:5]:
            self.device_combo.addItem(preset.name, preset)

        self.device_combo.insertSeparator(self.device_combo.count())
        for preset in IPHONE_PRESETS[5:9]:
            self.device_combo.addItem(preset.name, preset)

        self.device_combo.insertSeparator(self.device_combo.count())
        for preset in IPHONE_PRESETS[9:]:
            self.device_combo.addItem(preset.name, preset)

        self.device_combo.currentIndexChanged[int].connect(
            self.change_device_preset
        )

        self.prev_button = QPushButton("<<")
        self.prev_button.clicked.connect(self.prev_frame)
        self.prev_button.setAutoRepeat(True)
        self.prev_button.setAutoRepeatDelay(300)
        self.prev_button.setAutoRepeatInterval(60)

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

        self.frame_label = QLabel("Frame: 0 / 0")
        self.frame_label.setMinimumWidth(140)
        self.frame_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.auto_framing = True
        self.auto_frame_button = QPushButton("Auto Frame")
        self.auto_frame_button.clicked.connect(self.set_auto_frame_mode)

        self.fit_video_button = QPushButton("Fit Video")
        self.fit_video_button.clicked.connect(self.set_fit_video_mode)

        framing_mode_layout = QVBoxLayout()
        framing_mode_layout.addWidget(self.auto_frame_button)
        framing_mode_layout.addWidget(self.fit_video_button)

        framing_mode_frame = QFrame()
        framing_mode_frame.setFrameShape(QFrame.StyledPanel)
        framing_mode_frame.setLayout(framing_mode_layout)
        framing_mode_frame.setStyleSheet(
            "QFrame { border: 1px solid gray; border-radius: 4px; padding: 4px; }"
        )

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
        self.manual_center: tuple[float, float] | None = None
        self.manual_base_crop_size: tuple[float, float] | None = None
        self.manual_anchor_kind: str | None = None
        self.manual_anchor_face: dict | None = None
        self.manual_zoom_factor = 1.0
        self.manual_pan_x = 0.0
        self.manual_pan_y = 0.0
        self.last_visible_center: tuple[float, float] | None = None

        self.face_index: FacewiseJsonIndex | None = None
        self.displayed_frame_idx = 0
        self.current_shot_number = None

        playback_controls = QHBoxLayout()
        playback_controls.addWidget(self.open_button)
        playback_controls.addWidget(self.open_json_button)
        playback_controls.addWidget(self.start_button)
        playback_controls.addWidget(self.prev_button)
        playback_controls.addWidget(self.play_button)
        playback_controls.addWidget(self.next_button)
        playback_controls.addWidget(self.end_button)
        playback_controls.addWidget(self.slider, stretch=1)
        playback_controls.addWidget(self.frame_label)

        framing_controls = QHBoxLayout()
        framing_controls.addStretch(1)
        framing_controls.addWidget(self.orientation_button)
        framing_controls.addWidget(self.device_combo)
        framing_controls.addWidget(framing_mode_frame)
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
        self.update_framing_buttons()

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
        self.prev_button.setEnabled(enabled)
        self.play_button.setEnabled(enabled)
        self.next_button.setEnabled(enabled)
        self.end_button.setEnabled(enabled)
        self.slider.setEnabled(enabled)

    def set_stopped(self):
        self.timer.stop()
        self.play_button.setText("Play")
        self.prev_button.setEnabled(self.reader is not None)
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
        self.auto_framing = True
        self.reset_manual_mode()
        self.update_framing_buttons()

        self.timer.setInterval(max(1, int(1000 / self.reader.fps)))
        self.slider.setMaximum(max(0, self.reader.frame_count - 1))
        self.slider.setValue(0)
        self.frame_label.setText(
            f"Frame: 0 / {max(0, self.reader.frame_count - 1)}")

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
        self.current_shot_number = None
        self.auto_framing = True
        self.reset_manual_mode()
        self.update_framing_buttons()

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
        img = self.reader.frame_at_exact(last)
        if img is not None:
            self.show_frame(img)

    def prev_frame(self):
        if self.reader is None:
            return

        self.set_stopped()
        target = max(0, self.reader.current_frame_idx - 2)
        img = self.reader.frame_at_exact(target)
        if img is not None:
            self.show_frame(img)

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
        img = self.reader.frame_at_exact(target)
        if img is not None:
            self.show_frame(img)

        self._slider_dragging = False

    def show_frame(self, img):
        self.displayed_frame_idx = max(0, int(self.reader.current_frame_idx) - 1)
        self.maybe_reset_manual_mode_for_scene_change()

        h, w, ch = img.shape
        qimg = QImage(img.data, w, h, ch * w, QImage.Format_RGB888).copy()
        self.current_pixmap = QPixmap.fromImage(qimg)
        self.render_current_pixmap()
        self.update_drawer_faces()

        if not self._slider_dragging:
            self.slider.blockSignals(True)
            self.slider.setValue(min(self.reader.current_frame_idx - 1, self.slider.maximum()))
            self.slider.blockSignals(False)

        self.frame_label.setText(
            f"Frame: {self.displayed_frame_idx:,} / "
            f"{self.slider.maximum():,}"
        )

    def shot_number_for_frame(self, frame_idx: int):
        if self.face_index is None:
            return None

        faces = self.face_index.faces_at(frame_idx)
        if not faces:
            return None

        return faces[0].get("shot_number")

    def maybe_reset_manual_mode_for_scene_change(self):
        shot_number = self.shot_number_for_frame(self.displayed_frame_idx)
        if shot_number is None:
            return

        if self.current_shot_number is None:
            self.current_shot_number = shot_number
            return

        if shot_number == self.current_shot_number:
            return

        self.current_shot_number = shot_number

        if self.manual_mode:
            self.auto_framing = True
            self.reset_manual_mode()
            self.update_framing_buttons()

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

        crop = self.current_source_crop()
        self.paint_source_crop(src, canvas, crop)
        self.last_visible_center = self.crop_center(crop)

        display_long_side = round(REFERENCE_LONG_SIDE * 
                                  self.device.display_inches / 
                                  REFERENCE_DIAGONAL)

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
        display_long_side = round(
            REFERENCE_LONG_SIDE * self.device.display_inches / REFERENCE_DIAGONAL
        )
        if viewport_w >= viewport_h:
            display_w = display_long_side
            display_h = round(display_w * viewport_h / viewport_w)
        else:
            display_h = display_long_side
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
            self.drawer_scroll_area.setGeometry(0, 0, display_w, drawer_h)
            self.drawer_button.setText("⌃" if not self.drawer_visible else "⌄")
            self.drawer_button.resize(72, 36)
            self.drawer_button.move(
                phone_x + (display_w - self.drawer_button.width()) // 2,
                phone_y + display_h - self.drawer_button.height(),
            )

            self.drawer_layout.setDirection(QBoxLayout.LeftToRight)
            self.drawer_thumb_w = 120
            self.drawer_thumb_h = max(1, drawer_h - 24)
        else:
            drawer_w = 180
            self.drawer_panel.setGeometry(
                phone_x + display_w - drawer_w,
                phone_y,
                drawer_w,
                display_h,
            )
            self.drawer_scroll_area.setGeometry(0, 0, drawer_w, display_h)
            self.drawer_button.setText("‹" if not self.drawer_visible else "›")
            self.drawer_button.resize(36, 72)
            self.drawer_button.move(
                phone_x + display_w - self.drawer_button.width(),
                phone_y + (display_h - self.drawer_button.height()) // 2,
            )

            self.drawer_layout.setDirection(QBoxLayout.TopToBottom)
            self.drawer_thumb_w = max(1, drawer_w - 24)
            self.drawer_thumb_h = 110
        self.drawer_panel.setVisible(self.drawer_visible)
        self.drawer_panel.raise_()
        self.drawer_scroll_area.raise_()
        self.drawer_button.raise_()
        self.drawer_button.show()
        self.update_drawer_faces()

    def viewport_size(self) -> tuple[int, int]:
        w = self.device.viewport_width
        h = self.device.viewport_height
        return (w, h) if self.is_portrait else (h, w)

    def change_device_preset(self, index: int):
        preset = self.device_combo.itemData(index)
        if preset is None:
            return

        self.device = preset
        self.render_current_pixmap()
        self.update_drawer_faces()
    
    def viewport_aspect(self) -> float:
        viewport_w, viewport_h = self.viewport_size()
        return float(viewport_w) / float(viewport_h)
    
    def toggle_boxes(self):
        self.show_boxes = not self.show_boxes
        self.boxes_button.setText("Hide Boxes" if self.show_boxes else "Show Boxes")
        self.render_current_pixmap()

    def set_auto_frame_mode(self):
        self.auto_framing = True
        self.reset_manual_mode()
        self.update_framing_buttons()
        self.render_current_pixmap()

    def set_fit_video_mode(self):
        self.auto_framing = False
        self.reset_manual_mode()
        self.update_framing_buttons()
        self.render_current_pixmap()

    def update_framing_buttons(self):
        """
        Keep Auto Frame and Fit Video as independent actions.

        Pure Fit Video:
            Auto Frame enabled, Fit Video disabled.

        Pure Auto Frame:
            Auto Frame disabled, Fit Video enabled.

        Manual zoom/pan/face-selection:
            both buttons enabled.
        """
        if self.manual_mode:
            self.auto_frame_button.setEnabled(True)
            self.fit_video_button.setEnabled(True)
        elif self.auto_framing:
            self.auto_frame_button.setEnabled(False)
            self.fit_video_button.setEnabled(True)
        else:
            self.auto_frame_button.setEnabled(True)
            self.fit_video_button.setEnabled(False)

    def reset_manual_mode(self):
        self.manual_mode = False
        self.manual_zoom = 1.0
        self.manual_center = None
        self.manual_base_crop_size = None
        self.manual_anchor_kind = None
        self.manual_anchor_face = None
        self.manual_zoom_factor = 1.0
        self.manual_pan_x = 0.0
        self.manual_pan_y = 0.0

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
    
    def largest_faces_for_current_frame(self, limit: int | None = 2) -> list[dict]:
        """
        Return the faces for the current frame sorted by descending face area.

        If `limit` is an integer, return at most that many faces.

        If `limit` is None, return all faces. 
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

        while self.drawer_layout.count():
            item = self.drawer_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.drawer_face_labels = []

        faces = self.largest_faces_for_current_frame(limit=None)

        for face in faces:
            label = QLabel(self.drawer_contents)
            label.setAlignment(Qt.AlignCenter)
            label.setFixedSize(self.drawer_thumb_w, self.drawer_thumb_h)
            label.setStyleSheet(
                "QLabel { color: white; background: rgba(255, 255, 255, 30); }"
            )
            label.mousePressEvent = (
                lambda event, selected_face=face: self.select_drawer_face(selected_face)
            )
            self.drawer_layout.addWidget(label)
            self.drawer_face_labels.append(label)

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
                    self.drawer_thumb_w,
                    self.drawer_thumb_h,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )

        self.drawer_layout.addStretch(1)

    def zoom_in(self):
        self.enter_manual_mode()
        self.manual_zoom_factor = min(6.0, self.manual_zoom_factor * 1.15)
        self.render_current_pixmap()

    def zoom_out(self):
        self.enter_manual_mode()
        self.manual_zoom_factor = max(1.0, self.manual_zoom_factor / 1.15)
        self.render_current_pixmap()

    def enter_manual_mode(self):
        """
        Enter manual framing while preserving the current visual framing.

        Manual zoom is measured relative to Fit Video:

            1.0  == Fit Video
            >1.0 == zoomed in from Fit Video

        When entering manual mode from Auto Frame, convert the current
        auto-face crop into a source-image center and a Fit-relative zoom.
        This allows later Zoom - operations to return all the way to Fit
        Video instead of stopping at the original Auto Frame crop.
        """
        if self.manual_mode:
            return

        if self.current_pixmap is None:
            return

        if self.auto_framing:
            face = self.auto_frame()
            if face is not None:
                self.begin_manual_from_face(face)
                return
            else:
                self.begin_manual_from_fit()
                return
        elif self.manual_center is None:
            self.begin_manual_from_fit()

        self.manual_mode = True
        self.update_framing_buttons()

    def minimum_manual_zoom(self) -> float:
        return 1.0

    def fit_source_size(self) -> tuple[float, float]:
        """
        Return the virtual source crop size corresponding to Fit Video.

        The crop may be larger than the actual source image in one dimension.
        That represents letterboxing/pillarboxing with black padding.
        """
        source_w = float(self.current_pixmap.width())
        source_h = float(self.current_pixmap.height())
        viewport_aspect = self.viewport_aspect()
        source_aspect = source_w / source_h

        if source_aspect > viewport_aspect:
            fit_w = source_w
            fit_h = source_w / viewport_aspect
        else:
            fit_h = source_h
            fit_w = source_h * viewport_aspect

        return fit_w, fit_h

    def fit_source_crop(self):
        """
        Return the virtual source crop corresponding to Fit Video.
        The crop may extend beyond the image, producing black bars.
        """
        fit_w, fit_h = self.fit_source_size()
        source_w = float(self.current_pixmap.width())
        source_h = float(self.current_pixmap.height())

        cx = source_w / 2.0
        cy = source_h / 2.0

        return (
            cx - fit_w / 2.0,
            cy - fit_h / 2.0,
            cx + fit_w / 2.0,
            cy + fit_h / 2.0,
        )
    
    def current_source_crop(self):
        if self.manual_mode:
            return self.manual_source_crop()

        if self.auto_framing:
            face = self.auto_frame()
            if face is not None:
                return self.normalized_source_crop(
                    self.crop_for_face(face, self.viewport_aspect())
                )

        return self.fit_source_crop()

    def current_source_center(self) -> tuple[float, float]:
        if self.last_visible_center is not None:
            return self.last_visible_center

        return (
            float(self.current_pixmap.width()) / 2.0,
            float(self.current_pixmap.height()) / 2.0,
        )

    def crop_center(self, crop) -> tuple[float, float]:
        x1, y1, x2, y2 = crop
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def zoom_for_source_crop(self, crop) -> float:
        fit_w, fit_h = self.fit_source_size()
        crop_w = max(1.0, crop[2] - crop[0])
        crop_h = max(1.0, crop[3] - crop[1])
        return max(1.0, min(fit_w / crop_w, fit_h / crop_h))
    
    def begin_manual_from_crop(self, crop):
        """
        Enter manual mode preserving the supplied source crop.
        """
        self.manual_center = self.crop_center(crop)
        self.manual_zoom = self.zoom_for_source_crop(crop)
        self.manual_base_crop_size = self.fit_source_size()
        self.manual_anchor_kind = "crop"
        self.manual_anchor_face = None
        self.manual_zoom_factor = self.zoom_for_source_crop(crop)
        self.manual_pan_x = 0.0
        self.manual_pan_y = 0.0
        self.manual_mode = True
        self.update_framing_buttons()

    def begin_manual_from_face(self, face):
        self.manual_anchor_kind = "face"
        self.manual_anchor_face = face
        crop = self.normalized_source_crop(
            self.crop_for_face(face, self.viewport_aspect())
        )
        self.manual_zoom_factor = self.zoom_for_source_crop(crop)
        self.manual_pan_x = 0.0
        self.manual_pan_y = 0.0
        self.manual_mode = True
        self.update_framing_buttons()

    def begin_manual_from_fit(self):
        self.manual_anchor_kind = "fit"
        self.manual_anchor_face = None
        self.manual_zoom_factor = 1.0
        self.manual_pan_x = 0.0
        self.manual_pan_y = 0.0
        self.manual_mode = True
        self.update_framing_buttons()

    def manual_anchor_crop(self):
        if self.manual_anchor_kind == "face" and self.manual_anchor_face is not None:
            return self.normalized_source_crop(
                self.crop_for_face(self.manual_anchor_face, self.viewport_aspect())
            )

        if self.manual_anchor_kind == "fit":
            return self.fit_source_crop()

        return self.current_source_crop()

    def manual_source_crop(self):
        anchor = self.manual_anchor_crop()
        ax, ay = self.crop_center(anchor)
        fit_w, fit_h = self.fit_source_size()
        visible_w = fit_w / max(1.0, self.manual_zoom_factor)
        visible_h = fit_h / max(1.0, self.manual_zoom_factor)

        cx = ax + self.manual_pan_x * visible_w
        cy = ay + self.manual_pan_y * visible_h
        self.manual_center = (cx, cy)
        return self.normalized_source_crop(
            (
                cx - visible_w / 2.0,
                cy - visible_h / 2.0,
                cx + visible_w / 2.0,
                cy + visible_h / 2.0,
            )
        )

    def normalized_source_crop(self, crop):
        """
        Clamp a virtual source crop center without forcing the crop itself
        inside the image.

        Crops larger than the image are valid: they render as black padding.
        """
        x1, y1, x2, y2 = crop
        source_w = float(self.current_pixmap.width())
        source_h = float(self.current_pixmap.height())

        crop_w = max(1.0, x2 - x1)
        crop_h = max(1.0, y2 - y1)

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        if crop_w >= source_w:
            cx = source_w / 2.0
        else:
            cx = max(crop_w / 2.0, min(source_w - crop_w / 2.0, cx))

        if crop_h >= source_h:
            cy = source_h / 2.0
        else:
            cy = max(crop_h / 2.0, min(source_h - crop_h / 2.0, cy))

        return (
            cx - crop_w / 2.0,
            cy - crop_h / 2.0,
            cx + crop_w / 2.0,
            cy + crop_h / 2.0,
        )
    
    def clamp_manual_center(self):
        if self.current_pixmap is None:
            return

        if self.manual_center is None:
            return

        fit_w, fit_h = self.fit_source_size()
        visible_w = fit_w / max(1.0, self.manual_zoom)
        visible_h = fit_h / max(1.0, self.manual_zoom)

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

        if self.manual_center is None:
            return

        self.manual_pan_x += dx
        self.manual_pan_y += dy
        self.render_current_pixmap()

    def select_drawer_face(self, face):
        self.auto_framing = False
        self.reset_manual_mode()
        self.begin_manual_from_face(face)
        self.render_current_pixmap()

    def paint_source_crop(self, src, canvas, crop):
        crop_x1, crop_y1, crop_x2, crop_y2 = crop
        crop_w = max(1.0, crop_x2 - crop_x1)
        crop_h = max(1.0, crop_y2 - crop_y1)

        ix1 = max(0.0, crop_x1)
        iy1 = max(0.0, crop_y1)
        ix2 = min(float(src.width()), crop_x2)
        iy2 = min(float(src.height()), crop_y2)

        if ix2 <= ix1 or iy2 <= iy1:
            return

        scale = min(canvas.width() / crop_w, canvas.height() / crop_h)
        draw_w = crop_w * scale
        draw_h = crop_h * scale
        ox = (canvas.width() - draw_w) / 2.0
        oy = (canvas.height() - draw_h) / 2.0

        piece = src.copy(int(ix1), int(iy1), int(ix2 - ix1), int(iy2 - iy1))

        painter = QPainter(canvas)
        painter.drawPixmap(
            int(ox + (ix1 - crop_x1) * scale),
            int(oy + (iy1 - crop_y1) * scale),
            int((ix2 - ix1) * scale),
            int((iy2 - iy1) * scale),
            piece,
        )
        painter.end()

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