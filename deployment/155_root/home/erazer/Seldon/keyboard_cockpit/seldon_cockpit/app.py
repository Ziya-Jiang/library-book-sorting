import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import time

from PyQt5 import QtCore, QtGui, QtWidgets
from .backend import latest, offer, run_control
from .camera import run_camera
from .control import validate_config

ROOT = Path(__file__).resolve().parents[1]
KEYS = {QtCore.Qt.Key_Up: "up", QtCore.Qt.Key_Down: "down",
        QtCore.Qt.Key_Left: "left", QtCore.Qt.Key_Right: "right",
        QtCore.Qt.Key_W: "w", QtCore.Qt.Key_S: "s", QtCore.Qt.Key_A: "a",
        QtCore.Qt.Key_D: "d", QtCore.Qt.Key_Shift: "shift", QtCore.Qt.Key_Space: "space"}
MOVEMENT = {"up", "down", "left", "right", "w", "s", "a", "d"}


class Viewfinder(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 270)
        self.picture = QtGui.QPixmap()
        self.fresh = False
        self.note = "等待头部 ORBBEC 彩色画面"
        self.keys = set()

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor("#0c1119"))
        if not self.picture.isNull() and self.fresh:
            size = self.picture.size().scaled(self.size(), QtCore.Qt.KeepAspectRatio)
            dst = QtCore.QRect((self.width()-size.width())//2, (self.height()-size.height())//2,
                               size.width(), size.height())
            p.drawPixmap(dst, self.picture)
        else:
            p.setPen(QtGui.QColor("#9aaec4"))
            p.setFont(QtGui.QFont("Sans", 16))
            p.drawText(self.rect().adjusted(40, 40, -40, -40), QtCore.Qt.AlignCenter | QtCore.Qt.TextWordWrap, self.note)
        cx, cy = self.width()//2, self.height()//2
        p.setPen(QtGui.QPen(QtGui.QColor("#68e4db"), 2))
        for a, b, c, d in ((cx-25, cy, cx-10, cy), (cx+10, cy, cx+25, cy),
                           (cx, cy-25, cx, cy-10), (cx, cy+10, cx, cy+25)):
            p.drawLine(a, b, c, d)
        p.fillRect(16, 16, 230, 34, QtGui.QColor(9, 15, 23, 210))
        p.setFont(QtGui.QFont("Sans", 11))
        p.drawText(28, 39, "HEAD CAM  /  ORBBEC")
        p.end()


class Cockpit(QtWidgets.QMainWindow):
    def __init__(self, config, real, camera, intents, statuses, frames, worker, screenshot=None):
        super().__init__()
        self.config, self.real, self.camera = config, real, camera
        self.intents, self.statuses, self.frames, self.worker = intents, statuses, frames, worker
        self.screenshot = screenshot
        self.pressed, self.first_press = {}, {}
        self.arm_id, self.stopped, self.frame_time = 0, True, 0.0
        self.status = {"state": "LOCKED", "message": "按 E 接管；J 停车", "velocity": [0., 0., 0.], "gear": "normal"}
        self.setWindowTitle("Seldon · 机器人第一人称驾驶舱")
        self.resize(1280, 820)
        self.setMinimumSize(720, 550)
        host = QtWidgets.QWidget()
        self.setCentralWidget(host)
        layout = QtWidgets.QVBoxLayout(host)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)
        bar = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("SELDON  /  驾驶舱")
        title.setObjectName("title")
        bar.addWidget(title)
        bar.addStretch()
        self.mode = QtWidgets.QLabel("实车接口" if real else "离线演示 · 不发送 ROS 指令")
        bar.addWidget(self.mode)
        full = QtWidgets.QPushButton("全屏  F11")
        full.clicked.connect(self.toggle_fullscreen)
        full.setFocusPolicy(QtCore.Qt.NoFocus)
        bar.addWidget(full)
        layout.addLayout(bar)
        self.view = Viewfinder()
        if camera == "none":
            self.view.note = "离线操控演示\n尚未接入相机 · 此处显示机器人真实视角"
        layout.addWidget(self.view, 1)
        self.state_label = QtWidgets.QLabel("已锁定")
        self.state_label.setObjectName("state")
        layout.addWidget(self.state_label)
        self.message = QtWidgets.QLabel("按 E 接管；J 停车")
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        telemetry = QtWidgets.QHBoxLayout()
        self.speed = QtWidgets.QLabel("指令  0.00 m/s   ·   0.00 rad/s")
        self.measured = QtWidgets.QLabel("实测速度  —")
        self.video = QtWidgets.QLabel("画面未接入")
        telemetry.addWidget(self.speed)
        telemetry.addStretch()
        telemetry.addWidget(self.measured)
        telemetry.addStretch()
        telemetry.addWidget(self.video)
        layout.addLayout(telemetry)
        controls = QtWidgets.QHBoxLayout()
        help_text = QtWidgets.QLabel("↑↓ / W S  前后   ·   ←→  转向   ·   A D  横移\nShift  高速   ·   空格  低速   ·   松键减速")
        controls.addWidget(help_text, 1)
        self.arm = QtWidgets.QPushButton("接管  E")
        self.arm.clicked.connect(self.request_arm)
        self.arm.setFocusPolicy(QtCore.Qt.NoFocus)
        controls.addWidget(self.arm)
        stop = QtWidgets.QPushButton("停车锁定  J")
        stop.setObjectName("stop")
        stop.setFocusPolicy(QtCore.Qt.NoFocus)
        stop.clicked.connect(self.stop)
        controls.addWidget(stop)
        layout.addLayout(controls)
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #131c28; color: #e8f0f8; font: 14px 'Sans'; }
            QLabel#title { font-size: 22px; font-weight: bold; letter-spacing: 2px; }
            QLabel#state { color: #68e4db; font-size: 21px; font-weight: bold; }
            QPushButton { background: #233447; border: 1px solid #476078; padding: 13px 22px; border-radius: 6px; }
            QPushButton:hover { background: #31495f; }
            QPushButton#stop { background: #9c3040; border-color: #ff7b85; font-weight: bold; }
        """)
        QtWidgets.QApplication.instance().installEventFilter(self)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(20)

    def request_arm(self):
        if self.pressed or self.status["state"] != "LOCKED":
            return
        self.stopped = False
        self.arm_id += 1
        self.send_intent()

    def stop(self):
        self.stopped = True
        self.pressed.clear()
        self.first_press.clear()
        self.send_intent()

    def toggle_fullscreen(self):
        self.stop()
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    def send_intent(self):
        offer(self.intents, {"time": time.monotonic(), "keys": list(self.pressed),
                            "focused": self.isActiveWindow() and not self.isMinimized(),
                            "stop": self.stopped, "arm_id": self.arm_id, "frame_time": self.frame_time})

    def eventFilter(self, obj, event):
        kind = event.type()
        if kind == QtCore.QEvent.ApplicationDeactivate or (obj is self and kind == QtCore.QEvent.WindowDeactivate):
            self.stop()
        if kind not in (QtCore.QEvent.KeyPress, QtCore.QEvent.KeyRelease) or not self.isActiveWindow():
            return super().eventFilter(obj, event)
        key, down = event.key(), kind == QtCore.QEvent.KeyPress
        if key in (QtCore.Qt.Key_J, QtCore.Qt.Key_Escape):
            if down:
                self.stop()
                if key == QtCore.Qt.Key_Escape and self.isFullScreen():
                    self.showNormal()
            return True
        if key in (QtCore.Qt.Key_E, QtCore.Qt.Key_F11):
            if down and not event.isAutoRepeat():
                self.request_arm() if key == QtCore.Qt.Key_E else self.toggle_fullscreen()
            return True
        if key in KEYS:
            name, now = KEYS[key], time.monotonic()
            if down and self.status["state"] == "ACTIVE" and not self.stopped:
                # Autorepeat may refresh only a physically started key; stale
                # repeats after J/focus loss cannot resurrect a held key.
                if not event.isAutoRepeat() or name in self.pressed:
                    self.pressed[name] = now
                    self.first_press.setdefault(name, now)
            elif not down and not event.isAutoRepeat():
                self.pressed.pop(name, None)
                self.first_press.pop(name, None)
            self.send_intent()
            return True
        return super().eventFilter(obj, event)

    def tick(self):
        now = time.monotonic()
        frame = latest(self.frames)
        if frame:
            if "jpeg" in frame:
                picture = QtGui.QPixmap()
                if picture.loadFromData(frame["jpeg"], "JPG"):
                    self.view.picture, self.frame_time = picture, frame["time"]
            elif "error" in frame:
                self.view.note = frame["error"]
                self.frame_time = 0.0
        age = now-self.frame_time
        self.view.fresh = 0 <= age <= self.config["camera_timeout"]
        if self.frame_time and not self.view.fresh:
            self.view.note = "画面已中断 · 控制已锁定"
        self.video.setText(f"画面年龄 {age*1000:.0f} ms" if self.view.fresh else "画面不可用")
        key_fault = ""
        movement = self.pressed.keys() & MOVEMENT
        # X11 repeats only the most recently pressed repeatable key. A chord
        # such as Up+A+Space must not expire Up while Space is repeating.
        if movement and now-max(self.pressed.values()) > self.config["key_repeat_timeout"]:
            key_fault = "按键续传超时；松键后重新接管"
        for key in movement:
            if now-self.first_press[key] > self.config["max_hold_seconds"]:
                key_fault = f"连续按键达到 {self.config['max_hold_seconds']:g} 秒；松键后重新接管"
        if key_fault:
            self.stop()
        self.send_intent()
        self.status = latest(self.statuses, self.status)
        stale = now-self.status.get("time", now) > 0.5 or not self.worker.is_alive()
        if stale:
            self.stop()
        state = "ERROR" if stale else self.status["state"]
        names = {"LOCKED": "已锁定", "ZEROING": "准备接管", "MODE": "切换模式", "ENABLE": "使能中", "ACTIVE": "手动驾驶", "ERROR": "控制进程不可用"}
        gear = {"slow": "低速", "normal": "常速", "fast": "高速"}[self.status["gear"]]
        self.state_label.setText(f"{names[state]}  /  {gear}")
        error = self.status["message"] if self.status["state"] == "ERROR" else "控制进程中断，等待底盘看门狗停车"
        self.message.setText(key_fault or (error if stale else self.status["message"]))
        x, y, w = self.status["velocity"] if not stale else (0., 0., 0.)
        self.speed.setText(f"指令  前后 {x:+.2f} / 横移 {y:+.2f} m/s  ·  转向 {w:+.2f} rad/s")
        measured = self.status.get("measured")
        self.measured.setText(f"实测 {measured[0]:+.2f}, {measured[1]:+.2f} m/s" if measured else "实测速度  —")
        self.view.update()

    def closeEvent(self, event):
        self.stop()
        self.timer.stop()
        QtWidgets.QApplication.instance().removeEventFilter(self)
        event.accept()


def main():
    parser = argparse.ArgumentParser(description="Seldon ORBBEC keyboard cockpit (default: offline)")
    parser.add_argument("--ros", action="store_true", help="Enable real ROS chassis interface; E still required")
    parser.add_argument("--camera", choices=("none", "orbbec", "ros"), default=None)
    parser.add_argument("--config", type=Path, default=ROOT/"config.json")
    parser.add_argument("--camera-topic")
    parser.add_argument("--compressed", action="store_true")
    parser.add_argument("--orbbec-serial")
    parser.add_argument("--smoke-test", action="store_true", help="Offline only, exit after 1 s and save screenshot")
    args = parser.parse_args()
    if args.smoke_test and args.ros:
        parser.error("--smoke-test cannot control real hardware")
    config = validate_config(json.loads(args.config.read_text()))
    if args.camera_topic:
        config["camera_topic"] = args.camera_topic
    if args.compressed:
        config["camera_compressed"] = True
    if args.orbbec_serial:
        config["orbbec_serial"] = args.orbbec_serial
    camera = args.camera or ("orbbec" if args.ros else "none")
    if args.ros and camera == "none":
        parser.error("实车第一人称操控需要 orbbec 或 ros 相机画面")
    if args.smoke_test:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    ctx = mp.get_context("spawn")
    intents, statuses, frames = ctx.Queue(2), ctx.Queue(2), ctx.Queue(1)
    shutdown = ctx.Event()
    worker = ctx.Process(target=run_control, args=(config, args.ros, intents, statuses, shutdown), name="seldon-control")
    camera_worker = ctx.Process(target=run_camera, args=(camera, config, frames, shutdown), name="seldon-camera")
    worker.start()
    if camera != "none":
        camera_worker.start()
    application = QtWidgets.QApplication(sys.argv[:1])
    cockpit = Cockpit(config, args.ros, camera, intents, statuses, frames, worker)
    cockpit.show()
    signal.signal(signal.SIGINT, lambda *_: cockpit.close())
    signal.signal(signal.SIGTERM, lambda *_: cockpit.close())
    if args.smoke_test:
        def capture():
            cockpit.grab().save(str(ROOT/"cockpit-preview.png"))
            cockpit.close()
        QtCore.QTimer.singleShot(1000, capture)
    try:
        result = application.exec_()
    finally:
        shutdown.set()
        worker.join(config["service_timeout"]+2)
        if worker.is_alive():
            worker.terminate()
            worker.join(1)
        if camera_worker.pid:
            camera_worker.join(2)
            if camera_worker.is_alive():
                camera_worker.terminate()
                camera_worker.join(1)
    return result


if __name__ == "__main__":
    sys.exit(main())
