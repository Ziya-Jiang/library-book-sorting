"""Camera acquisition isolated from movement watchdog. Never start an arm driver."""
import fcntl
import os
import struct
import time
from pathlib import Path
from .backend import offer


def _video_serial(device):
    sys_device = (Path('/sys/class/video4linux')/Path(device).name/'device').resolve()
    for parent in (sys_device, *sys_device.parents):
        serial_file = parent/'serial'
        try:
            return serial_file.read_text().strip()
        except OSError:
            continue
    return ''


def _v4l2_formats(device):
    formats = set()
    try:
        fd = os.open(device, os.O_RDWR | os.O_NONBLOCK)
    except OSError:
        return formats
    try:
        for index in range(32):
            descriptor = bytearray(64)
            struct.pack_into('II', descriptor, 0, index, 1)  # VIDEO_CAPTURE
            try:
                fcntl.ioctl(fd, 0xC0405602, descriptor, True)  # VIDIOC_ENUM_FMT
            except OSError:
                break
            formats.add(bytes(descriptor[44:48]).decode('ascii', errors='ignore'))
    finally:
        os.close(fd)
    return formats


def _find_v4l2_color(serial):
    for sys_video in sorted(Path('/sys/class/video4linux').glob('video*')):
        device = '/dev/'+sys_video.name
        if _video_serial(device) == serial and 'MJPG' in _v4l2_formats(device):
            return device
    return ''


def _run_v4l2(cv2, device, config, send, shutdown):
    capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.get('camera_width', 640))
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.get('camera_height', 480))
    capture.set(cv2.CAP_PROP_FPS, config.get('camera_fps', 15))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f'无法打开 UVC 彩色节点 {device}')
    try:
        last_received = time.monotonic()
        while not shutdown.is_set():
            ok, image = capture.read()
            now = time.monotonic()
            if ok and image is not None and image.size:
                last_received = now
                send(image, now)
            elif now-last_received > 5.0:
                raise RuntimeError(f'{device} 连续 5 秒未返回图像帧')
    finally:
        capture.release()


def _run_orbbec_once(cv2, np, config, send, shutdown):
    """Run one Orbbec connection until shutdown or a device/stream error.

    A USB disconnect invalidates the SDK pipeline object. Let the caller
    dispose of it and create a new Context/Pipeline after re-enumeration.
    """
    if config.get('shared_rgbd_enabled', False):
        from .shared_rgbd_v34 import run_shared_head
        return run_shared_head(config, send, shutdown)
    pipeline = None
    sdk_error = None
    try:
        from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, Context
        context = Context()
        devices = context.query_devices()
        if devices.get_count() == 0:
            raise RuntimeError("未检测到 ORBBEC 相机")
        serial = config.get("orbbec_serial", "")
        if serial:
            device = devices.get_device_by_serial_number(serial)
        elif devices.get_count() == 1:
            device = devices.get_device_by_index(0)
        else:
            raise RuntimeError("有多台 ORBBEC，请在 config.json 指定头部相机序列号")
        pipeline = Pipeline(device)
        stream_config = Config()
        profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        stream_config.enable_stream(profiles.get_video_stream_profile(
            config.get("camera_width", 640), config.get("camera_height", 480),
            OBFormat.MJPG, config.get("camera_fps", 15)))
        pipeline.start(stream_config)
    except Exception as exc:
        sdk_error = exc
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass
            pipeline = None

    if pipeline is None:
        serial = config.get("orbbec_serial", "")
        uvc_device = _find_v4l2_color(serial) if serial else ''
        if not uvc_device:
            raise RuntimeError(f"ORBBEC SDK 打开失败: {sdk_error}; 未找到同序列号的 UVC MJPEG 彩色节点")
        _run_v4l2(cv2, uvc_device, config, send, shutdown)
        return

    try:
        last_stamp = None
        last_received = time.monotonic()
        while not shutdown.is_set():
            packet = pipeline.wait_for_frames(100)
            frame = packet.get_color_frame() if packet else None
            if frame is None:
                if time.monotonic() - last_received > 5.0:
                    raise RuntimeError("Gemini 305 未返回图像帧；请检查 USB 数据连接")
                continue
            stamp = frame.get_timestamp()
            if stamp == last_stamp:
                continue
            last_stamp = stamp
            received = time.monotonic()
            last_received = received
            data = np.frombuffer(frame.get_data(), dtype=np.uint8)
            h, w, fmt = frame.get_height(), frame.get_width(), frame.get_format()
            if fmt == OBFormat.MJPG:
                image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            elif fmt == OBFormat.RGB:
                image = cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
            elif fmt == OBFormat.BGR:
                image = data.reshape(h, w, 3)
            elif fmt in (OBFormat.YUYV, OBFormat.UYVY):
                conversion = cv2.COLOR_YUV2BGR_YUYV if fmt == OBFormat.YUYV else cv2.COLOR_YUV2BGR_UYVY
                image = cv2.cvtColor(data.reshape(h, w, 2), conversion)
            elif fmt in (OBFormat.NV12, OBFormat.NV21, OBFormat.I420):
                conversion = {OBFormat.NV12: cv2.COLOR_YUV2BGR_NV12,
                              OBFormat.NV21: cv2.COLOR_YUV2BGR_NV21,
                              OBFormat.I420: cv2.COLOR_YUV2BGR_I420}[fmt]
                image = cv2.cvtColor(data.reshape(h*3//2, w), conversion)
            else:
                raise RuntimeError(f"不支持的 ORBBEC 彩色格式: {fmt}")
            send(image, received)
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass


def run_camera(source, config, frames, shutdown):
    node = None
    try:
        import cv2
        import numpy as np

        def send(image, stamp):
            if image is None or not image.size:
                return
            h, w = image.shape[:2]
            if w > 1280:
                image = cv2.resize(image, (1280, max(1, round(h*1280/w))))
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                offer(frames, {"time": stamp, "jpeg": encoded.tobytes()})

        if source == "orbbec":
            # Keep the worker alive across USB disconnect/re-enumeration. The
            # SDK objects cannot be reused after a disconnect, so every retry
            # creates a fresh Context and Pipeline and reselects by serial.
            retry_delay = 1.0
            while not shutdown.is_set():
                try:
                    _run_orbbec_once(cv2, np, config, send, shutdown)
                    if shutdown.is_set():
                        break
                    raise RuntimeError("Orbbec stream ended")
                except Exception as exc:
                    if shutdown.is_set():
                        break
                    offer(frames, {"error": f"相机断开，{retry_delay:.1f}s 后重连: {exc}"})
                    shutdown.wait(retry_delay)
                    retry_delay = min(5.0, retry_delay * 1.5)
        elif source == "uvc":
            serial = config.get("chassis_camera_serial", "")
            if not serial:
                raise RuntimeError("未配置底盘相机序列号")
            device = _find_v4l2_color(serial)
            if not device:
                raise RuntimeError(f"未找到序列号 {serial} 的 UVC MJPEG 彩色节点")
            chassis_config = dict(config)
            chassis_config.update(
                camera_width=config.get("chassis_camera_width", 640),
                camera_height=config.get("chassis_camera_height", 480),
                camera_fps=config.get("chassis_camera_fps", 15),
            )
            _run_v4l2(cv2, device, chassis_config, send, shutdown)
        elif source == "ros":
            import rclpy
            from cv_bridge import CvBridge
            from sensor_msgs.msg import Image, CompressedImage
            from rclpy.qos import qos_profile_sensor_data
            rclpy.init(args=[])
            node = rclpy.create_node("seldon_head_camera")
            bridge = CvBridge()
            last_stamp = [None]

            def callback(msg):
                try:
                    stamp = msg.header.stamp.sec*1_000_000_000+msg.header.stamp.nanosec
                    age = (node.get_clock().now().nanoseconds-stamp)/1e9
                    if stamp == last_stamp[0] or not 0 <= age <= config["camera_timeout"]:
                        return
                    last_stamp[0] = stamp
                    image = (bridge.compressed_imgmsg_to_cv2(msg, "bgr8") if config["camera_compressed"]
                             else bridge.imgmsg_to_cv2(msg, "bgr8"))
                    send(image, time.monotonic()-age)
                except Exception as exc:
                    offer(frames, {"error": str(exc)})

            node.create_subscription(CompressedImage if config["camera_compressed"] else Image,
                                     config["camera_topic"], callback, qos_profile_sensor_data)
            while not shutdown.is_set():
                rclpy.spin_once(node, timeout_sec=0.1)
        elif source != "none":
            raise ValueError("Unknown camera source")
    except Exception as exc:
        offer(frames, {"error": f"相机不可用: {exc}"})
    finally:
        if node:
            node.destroy_node()
            rclpy.shutdown()
