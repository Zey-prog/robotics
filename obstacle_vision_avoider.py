#!/usr/bin/env python3
# File: obstacle_vision_avoider.py
# Run on laptop. Requires ROS2 (rclpy), ultralytics YOLO, OpenCV installed.

import socket
import time
import math
import threading
import queue

import cv2
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan

# -------------------------
# CONFIG (edit these)
# -------------------------
ESP32_MOTOR_IP = "192.168.8.100"   # ESP32-S3 motor + PCA9685 server IP (change)
ESP32_MOTOR_PORT = 5050

ESP32_CAM_STREAM = "http://192.168.8.101:81/stream"  # ESP32-CAM stream URL (change)
YOLO_MODEL_PATH = "yolov5s.pt"  # path to your pretrained model

# Pick / drop distances and thresholds
ARM_TRIGGER_CM = 20.0       # when ultrasonic <= this, trigger pick
MIN_CLEAR_M = 0.45          # lidar front threshold (m)
FRAME_CENTER_DEADZONE = 60  # px tolerance to consider centered
CAM_PROC_FPS = 5            # approx frames per second to process

# servo IDs on PCA9685 (configure order)
SERVO_BOTTOM = 0   # servo id for bottom pitch
SERVO_MIDDLE = 1   # servo id for middle pitch
SERVO_GRIP = 2     # gripper open/close
SERVO_GRIP_ROLL = 3 # gripper roll
# adjust angles per your calibration:
GRIP_OPEN_ANGLE = 70
GRIP_CLOSED_ANGLE = 25

# speeds for servo movement (deg step per interpolation step)
DEFAULT_SERVO_SPEED = 40  # smaller => slower motion

# -------------------------
# Helper utilities
# -------------------------
def clamp(v, lo, hi): return max(lo, min(hi, v))

# Thread-safe state container
class SharedVisionState:
    def __init__(self):
        self.lock = threading.Lock()
        self.detected = False
        self.box = None    # (x, y, w, h) in full frame coords
        self.conf = 0.0
        self.timestamp = 0.0

# -------------------------
# ESP32 TCP client class
# -------------------------
class ESP32Client:
    def __init__(self, host, port=5050, timeout=2.0):
        self.host = host
        self.port = port
        self.sock = None
        self.lock = threading.Lock()
        self.timeout = timeout

    def connect(self):
        with self.lock:
            if self.sock:
                try:
                    self.sock.close()
                finally:
                    self.sock = None
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(self.timeout)
            s.connect((self.host, self.port))
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.sock = s
            return True

    def send(self, payload: bytes, expect_reply=False, reply_timeout=3.0):
        with self.lock:
            if not self.sock:
                try:
                    self.connect()
                except Exception as e:
                    return False, None
            try:
                self.sock.sendall(payload)
                if expect_reply:
                    self.sock.settimeout(reply_timeout)
                    data = b""
                    # read until newline
                    while True:
                        ch = self.sock.recv(1)
                        if not ch:
                            break
                        data += ch
                        if ch == b"\n":
                            break
                    return True, data.decode(errors="ignore").strip()
                return True, None
            except Exception as e:
                try:
                    self.sock.close()
                except:
                    pass
                self.sock = None
                return False, None

    def close(self):
        with self.lock:
            if self.sock:
                try:
                    self.sock.close()
                except:
                    pass
                self.sock = None

# -------------------------
# Vision worker thread: reads ESP32-CAM stream and runs YOLO
# -------------------------
def vision_worker(shared: SharedVisionState, stop_event: threading.Event):
    model = None
    try:
        print("Loading YOLO model...")
        model = YOLO(YOLO_MODEL_PATH)
        print("YOLO loaded.")
    except Exception as e:
        print("Failed to load YOLO model:", e)
        return

    cap = cv2.VideoCapture(ESP32_CAM_STREAM)
    if not cap.isOpened():
        print("Failed to open camera stream:", ESP32_CAM_STREAM)
        return

    last_proc = 0.0
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.5)
            cap.release()
            cap = cv2.VideoCapture(ESP32_CAM_STREAM)
            continue

        now = time.time()
        if now - last_proc < 1.0 / CAM_PROC_FPS:
            time.sleep(0.01)
            continue
        last_proc = now

        # small resize to speed up detection
        small = cv2.resize(frame, (320, 320))
        try:
            results = model.predict(source=small, imgsz=320, conf=0.4, verbose=False)
        except Exception as e:
            # detection failed this frame
            continue

        found = False
        best_box = None
        best_conf = 0.0
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                # label name
                label = model.names.get(cls_id, str(cls_id))
                # Only consider typical trash labels — adjust as needed
                if label in ("bottle", "cup", "can", "plastic", "trash", "bottle,plastic"):  # flexible
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    w = x2 - x1; h = y2 - y1
                    if conf > best_conf and w > 8 and h > 8:
                        best_conf = conf
                        best_box = (int(x1), int(y1), int(w), int(h))
                        found = True

        if found and best_box:
            # scale box back to original frame size
            fx = frame.shape[1] / 320.0
            fy = frame.shape[0] / 320.0
            bx, by, bw, bh = best_box
            full_box = (int(bx * fx), int(by * fy), int(bw * fx), int(bh * fy))
            with shared.lock:
                shared.detected = True
                shared.box = full_box
                shared.conf = best_conf
                shared.timestamp = now
        else:
            # no detection
            with shared.lock:
                shared.detected = False
                shared.box = None
                shared.conf = 0.0
                shared.timestamp = now
    cap.release()
    print("Vision worker stopped.")

# -------------------------
# Main ROS2 node (integrates lidar + vision + ESP32 commands)
# -------------------------
class IntegratedNode(Node):
    def __init__(self):
        super().__init__("integrated_vision_lidar_node")

        # params
        self.declare_parameter("esp32_ip", ESP32_MOTOR_IP)
        self.declare_parameter("esp32_port", ESP32_MOTOR_PORT)
        self.declare_parameter("min_clear_m", MIN_CLEAR_M)
        self.declare_parameter("arm_trigger_cm", ARM_TRIGGER_CM)

        self.esp32_ip = self.get_parameter("esp32_ip").get_parameter_value().string_value
        self.esp32_port = int(self.get_parameter("esp32_port").get_parameter_value().integer_value)
        self.min_clear_m = float(self.get_parameter("min_clear_m").get_parameter_value().double_value)
        self.arm_trigger_cm = float(self.get_parameter("arm_trigger_cm").get_parameter_value().double_value)

        # ESP32 client
        self.cli = ESP32Client(self.esp32_ip, self.esp32_port)
        try:
            ok = self.cli.connect()
            self.get_logger().info(f"Connected to ESP32 at {self.esp32_ip}:{self.esp32_port}")
        except Exception as e:
            self.get_logger().warn(f"ESP32 connect failed: {e}")

        # LIDAR QoS best-effort
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=10)
        self.sub = self.create_subscription(LaserScan, "/scan", self.on_scan, qos)

        # shared vision state
        self.vision = SharedVisionState()
        self.vstop = threading.Event()
        self.vthread = threading.Thread(target=vision_worker, args=(self.vision, self.vstop), daemon=True)
        self.vthread.start()

        # internal state
        self.last_send_ms = 0
        self.cooldown_ms = 120   # throttle tcp motor commands
        self.mode = "SEARCH"     # SEARCH, APPROACH, PICKING, DROPPING, AVOID
        self.last_cmd = None

        self.get_logger().info("Node ready. Mode=SEARCH")

    def send_cmd_throttled(self, cmd: str, spd: int = None, expect_reply=False, reply_timeout=2.0):
        now_ms = int(time.time() * 1000)
        if now_ms - self.last_send_ms < self.cooldown_ms:
            return True, None
        self.last_send_ms = now_ms
        data = cmd if spd is None else f"{cmd} {spd}"
        try:
            ok, reply = self.cli.send((data + "\n").encode("ascii"), expect_reply=expect_reply, reply_timeout=reply_timeout)
            return ok, reply
        except Exception as e:
            return False, None

    # ROS scan callback
    def on_scan(self, msg: LaserScan):
        # compute frontal min and side mins as before
        ranges = list(msg.ranges)
        if not ranges or msg.angle_increment == 0.0:
            return
        angle_min = msg.angle_min
        angle_inc = msg.angle_increment

        # build pairs
        pairs = []
        for i, r in enumerate(ranges):
            ang = angle_min + i * angle_inc
            ang_deg = math.degrees(ang)
            if ang_deg > 180.0: ang_deg -= 360.0
            if r <= 0.0 or math.isinf(r) or math.isnan(r): continue
            pairs.append((ang_deg, r))

        if not pairs:
            return

        def sector_min(center_deg, half_deg):
            lo = center_deg - half_deg
            hi = center_deg + half_deg
            vals = [d for (a, d) in pairs if lo <= a <= hi]
            return min(vals) if vals else float("inf")

        front_min = sector_min(0.0, 60.0)
        left_min  = sector_min(90.0, 60.0)
        right_min = sector_min(-90.0, 60.0)

        # Now decide based on mode
        with self.vision.lock:
            vis_detected = self.vision.detected
            vis_box = self.vision.box
            vis_conf = self.vision.conf
            vis_time = self.vision.timestamp

        # If we are currently picking/dropping, do not interrupt
        if self.mode in ("PICKING", "DROPPING"):
            return

        # If vision sees a trash recently, try to approach it
        if vis_detected and (time.time() - vis_time) < 2.0:
            # Go to APPROACH mode
            self.mode = "APPROACH"
            self.get_logger().info(f"Vision sees trash (conf {vis_conf:.2f}). Mode=APPROACH")
        else:
            # If not seeing trash, continue SEARCH/AVOID
            if self.mode != "SEARCH":
                self.mode = "SEARCH"

        # Mode handling:
        if self.mode == "APPROACH":
            # If obstacle too near in front by lidar, attempt to avoid first
            if front_min <= self.min_clear_m:
                # obstacle detected — perform a short pivot (use side with more space)
                if left_min > right_min:
                    self.send_cmd_throttled("L", 140)
                else:
                    self.send_cmd_throttled("R", 140)
                return

            # Use vision box to align
            if vis_box:
                x, y, w, h = vis_box
                frame_w = 640  # assume stream common width; or compute from cap if necessary
                center_x = x + w/2
                # compute left/right error
                # For safety, fetch the frame width from a detection frame if you prefer.
                # We'll assume 640 px — if you use a different cam, edit FRAME_CENTER_DEADZONE accordingly.
                frame_center = frame_w / 2
                dx = center_x - frame_center
                if abs(dx) < FRAME_CENTER_DEADZONE:
                    # centered => move forward
                    ok, _ = self.send_cmd_throttled("F", 150)
                elif dx < 0:
                    ok, _ = self.send_cmd_throttled("L", 110)
                else:
                    ok, _ = self.send_cmd_throttled("R", 110)

                # Poll ultrasonic distance from ESP32 to check if we are close to object
                ok, reply = self.send_cmd_throttled("D", None, expect_reply=True, reply_timeout=0.6)
                if ok and reply and reply.startswith("D:"):
                    try:
                        cm = float(reply.split(":")[1])
                    except:
                        cm = 999.0
                else:
                    cm = 999.0

                # If close enough, stop and run picking sequence
                if cm <= self.arm_trigger_cm:
                    self.send_cmd_throttled("S", None)
                    self.get_logger().info(f"Within {cm:.1f} cm — triggering PICK sequence")
                    # Run pick sequence synchronous: send 'ARM PICK' and wait for OK
                    ok, resp = self.send_cmd_throttled("ARM PICK", None, expect_reply=True, reply_timeout=10.0)
                    if ok and resp and resp.strip() == "OK":
                        self.get_logger().info("Pick sequence OK. Now dropping to bin.")
                        # Run drop sequence
                        ok2, resp2 = self.send_cmd_throttled("ARM DROP", None, expect_reply=True, reply_timeout=10.0)
                        if ok2 and resp2 and resp2.strip() == "OK":
                            self.get_logger().info("Drop sequence OK. Returning to SEARCH")
                        else:
                            self.get_logger().warn("Drop sequence failed or timed out")
                    else:
                        self.get_logger().warn("Pick sequence failed or timed out")
                    # After attempting pickup & drop, go back to SEARCH
                    self.mode = "SEARCH"
                return

        # Default SEARCH: lidar-based obstacle avoidance
        if front_min > self.min_clear_m:
            self.send_cmd_throttled("F", 160)
        else:
            if left_min > right_min:
                self.send_cmd_throttled("L", 150)
            else:
                self.send_cmd_throttled("R", 150)

    def destroy_node(self):
        self.vstop.set()
        self.vthread.join(timeout=2.0)
        self.cli.close()
        super().destroy_node()

# -------------------------
# Entrypoint
# -------------------------
def main():
    rclpy.init()
    node = IntegratedNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down node...")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
