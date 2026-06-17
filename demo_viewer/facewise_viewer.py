from __future__ import annotations

import argparse
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

        self.video_label = QLabel("Open a video to begin")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 450)

        self.open_button = QPushButton("Open…")
        self.open_button.clicked.connect(self.open_video)

        self.open_json_button = QPushButton("Open JSON…")
        self.open_json_button.clicked.connect(self.open_json)
        self.json_path: Path | None = None
        self.facewise_json: dict | None = None

        self.start_button = QPushButton("|<<")
        self.start_button.clicked.connect(self.go_to_start)

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

        self.face_index: FacewiseJsonIndex | None = None
        self.displayed_frame_idx = 0

        controls = QHBoxLayout()
        controls.addWidget(self.open_button)
        controls.addWidget(self.open_json_button)
        controls.addWidget(self.start_button)
        # controls.addWidget(self.prev_button)
        controls.addWidget(self.play_button)
        controls.addWidget(self.next_button)
        controls.addWidget(self.end_button)
        controls.addWidget(self.slider)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addLayout(controls)

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

        if not self._slider_dragging:
            self.slider.blockSignals(True)
            self.slider.setValue(min(self.reader.current_frame_idx - 1, self.slider.maximum()))
            self.slider.blockSignals(False)

    def render_current_pixmap(self):
        if self.current_pixmap is None:
            return

        pix = self.current_pixmap.copy()

        if self.face_index is not None:
            faces = self.face_index.faces_at(self.displayed_frame_idx)

            painter = QPainter(pix)
            pen = QPen(Qt.red)
            pen.setWidth(3)
            painter.setPen(pen)

            for face in faces:
                x1, y1, x2, y2 = face["bbox"]
                painter.drawRect(
                    int(x1),
                    int(y1),
                    int(x2 - x1),
                    int(y2 - y1),
                )

                label = face.get("face_label") or f"g={face.get('global_id')} s={face.get('segment_id')}"
                painter.drawText(int(x1), max(12, int(y1) - 6), label)

            painter.end()

        self.video_label.setPixmap(
            pix.scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

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