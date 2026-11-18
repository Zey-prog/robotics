#!/usr/bin/env python3
# integrated_vision_lidar_arm.py
#
# Laptop brain:
# - Subscribes to /scan (LiDAR)
# - Uses ESP32-CAM + YOLOv8 to detect trash
# - Controls ESP32-S3 (motors + PCA9685 servos + ultrasonic) over TCP
# - Receives ultrasonic data from ESP32 (gripper and bin level)
# - Implements SEARCH → APPROACH → COLLECT → DROP
# - Robot turns left/right using L298N to center trash
# - Arm starts collecting when trash is big enough in camera
# - Arm maintains position to trash during reaching
# - Monitors trash bin level and triggers buzzer when full

import socket
import time
import math
import threading
from collections import deque
from dataclasses import dataclass

import cv2
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan

# ---------------- CONFIG (EDIT THESE) ----------------

ESP32_MOTOR_IP = "192.168.8.100"    # ESP32-S3 IP
ESP32_MOTOR_PORT = 5050

ESP32_CAM_STREAM = "http://192.168.8.101:81/stream"  # ESP32-CAM stream URL
YOLO_MODEL_PATH = "yolov8s.pt"     # Use yolov8s; put absolute path if needed

# Navigation parameters
MIN_CLEAR_M = 0.45        # lidar clearance for forward movement
HARD_STOP_M = 0.25        # emergency stop threshold
OBSTACLE_TIMEOUT = 3.0    # seconds to try avoiding obstacle before giving up

# Collection parameters - NOW BASED ON CAMERA SIZE
MIN_TRASH_WIDTH = 100     # pixels - trash must be at least this wide to collect
MIN_TRASH_HEIGHT = 100    # pixels - trash must be at least this tall to collect
ULTRASONIC_SAFETY_CM = 5.0  # minimum distance for safety

# Ultrasonic parameters
GRIPPER_DISTANCE_CM = 10.0  # distance for gripper to consider object reached
BIN_FULL_DISTANCE_CM = 6.0  # distance indicating trash bin is full
BIN_CHECK_INTERVAL = 2.0    # seconds between bin level checks

# Vision parameters
FRAME_CENTER_DEADZONE = 40     # px, center alignment tolerance
STABLE_FRAMES_REQUIRED = 3     # number of consecutive frames trash must be centered

CAM_PROC_FPS = 30          # detection rate (YOLO inferences per second)

# Motor speeds (0-255)
SEARCH_FORWARD_SPEED = 150
SEARCH_TURN_SPEED = 190
APPROACH_FORWARD_SPEED = 100
CENTERING_TURN_SPEED = 140
OBSTACLE_AVOID_SPEED = 150
EMERGENCY_TURN_SPEED = 150

# Servo IDs on PCA9685
SERVO_BASE = 0        # base pitch (up/down at base)
SERVO_SHOULDER = 1    # shoulder pitch (up/down at middle)  
SERVO_GRIP = 2        # gripper open/close
SERVO_WRIST = 3       # wrist rotation (if needed)

# Servo angles (tune these!)
# Search/neutral pose - arm raised up
SEARCH_BASE = 90
SEARCH_SHOULDER = 90
SEARCH_WRIST = 10
GRIP_OPEN = 20
GRIP_CLOSED = 40

# Collection sequence poses - arm reaching down
PREPARE_BASE = 100
PREPARE_SHOULDER = 40
REACH_BASE = 80
REACH_SHOULDER = 100
LIFT_BASE = 40
LIFT_SHOULDER = 60

# Drop poses (adjust for your trash bin location)
DROP_BASE = 20       # Arm raised high for dropping
DROP_SHOULDER = 30   # Arm retracted for dropping

# Servo speeds (delay per degree step on ESP32 side; bigger = slower)
SERVO_SLOW = 30
SERVO_MED = 20
SERVO_FAST = 10

# YOLO label filter (class names must match model)
TRASH_LABELS = {"bottle", "cup", "can", "plastic", "trash"}

# ----------------------------------------------------

@dataclass
class UltrasonicData:
    """Container for ultrasonic sensor data received from ESP32"""
    gripper_distance: float = 999.0  # distance from gripper ultrasonic (cm)
    bin_distance: float = 999.0      # distance from bin level ultrasonic (cm)
    timestamp: float = 0.0           # when data was received
    bin_full: bool = False           # whether bin is full (based on distance)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class ESP32Client:
    def __init__(self, host, port=5050, timeout=5.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock = None
        self.lock = threading.Lock()
        self.last_connect_attempt = 0
        self.connect_cooldown = 3.0
        self.ultrasonic_data = UltrasonicData()
        self.bin_check_timer = 0.0

    def connect(self):
        with self.lock:
            if self.sock:
                try:
                    self.sock.close()
                except:
                    pass
                self.sock = None
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(self.timeout)
            s.connect((self.host, self.port))
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.sock = s
            print(f"[ESP32Client] Connected to {self.host}:{self.port}")
            return True

    def send(self, payload: bytes, expect_reply=False, reply_timeout=3.0):
        with self.lock:
            if not self.sock:
                now = time.time()
                if now - self.last_connect_attempt < self.connect_cooldown:
                    return False, None
                try:
                    if self.connect():
                        self.last_connect_attempt = now
                    else:
                        return False, None
                except Exception as e:
                    print(f"[ESP32Client] connect failed: {e}")
                    self.last_connect_attempt = now
                    return False, None
            
            try:
                self.sock.sendall(payload)
                if expect_reply:
                    self.sock.settimeout(reply_timeout)
                    data = b""
                    while True:
                        try:
                            ch = self.sock.recv(1)
                            if not ch:
                                break
                            data += ch
                            if ch == b"\n":
                                break
                        except socket.timeout:
                            break
                    return True, data.decode(errors="ignore").strip()
                return True, None
            except Exception as e:
                print(f"[ESP32Client] send failed: {e}")
                try:
                    self.sock.close()
                except:
                    pass
                self.sock = None
                return False, None

    def read_ultrasonic_data(self):
        """Read both ultrasonic sensors from ESP32"""
        now = time.time()
        
        # Read gripper distance
        ok1, reply1 = self.send("D", expect_reply=True, timeout=1.0)
        gripper_dist = 999.0
        if ok1 and reply1 and reply1.startswith("D:"):
            try:
                gripper_dist = float(reply1.split(":")[1])
            except:
                gripper_dist = 999.0
        
        # Read bin level periodically
        bin_dist = 999.0
        if now - self.bin_check_timer >= BIN_CHECK_INTERVAL:
            ok2, reply2 = self.send("BIN", expect_reply=True, timeout=1.0)
            if ok2 and reply2 and reply2.startswith("BIN:"):
                try:
                    bin_dist = float(reply2.split(":")[1])
                except:
                    bin_dist = 999.0
            self.bin_check_timer = now
        
        # Update ultrasonic data
        self.ultrasonic_data.gripper_distance = gripper_dist
        self.ultrasonic_data.bin_distance = bin_dist
        self.ultrasonic_data.timestamp = now
        self.ultrasonic_data.bin_full = (bin_dist <= BIN_FULL_DISTANCE_CM)
        
        return self.ultrasonic_data

    def get_ultrasonic_data(self):
        """Get the latest ultrasonic data without reading from ESP32"""
        return self.ultrasonic_data

    def control_buzzer(self, state: bool):
        """Control the buzzer on ESP32"""
        cmd = "BUZZER ON" if state else "BUZZER OFF"
        success, _ = self.send((cmd + "\n").encode("ascii"), False)
        return success

    def close(self):
        with self.lock:
            if self.sock:
                try:
                    self.sock.close()
                except:
                    pass
                self.sock = None


class SharedVisionState:
    """
    Shared state between vision thread and ROS node.
    Stores latest bounding box and a small history for stability checks.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.detected = False
        self.box = None  # (x, y, w, h)
        self.conf = 0.0
        self.timestamp = 0.0
        self.frame_width = 640
        self.frame_height = 480
        # for stability: last N centers
        self.history = deque(maxlen=10)

    def update(self, box, conf, fw, fh):
        now = time.time()
        with self.lock:
            self.detected = box is not None
            self.box = box
            self.conf = conf if box is not None else 0.0
            self.timestamp = now
            self.frame_width = fw
            self.frame_height = fh
            if box:
                x, y, w, h = box
                cx = x + w / 2
                cy = y + h / 2
                self.history.append((now, cx, cy))


def vision_worker(shared: SharedVisionState, stop_event: threading.Event):
    # Load YOLOv8
    try:
        print("[Vision] Loading YOLOv8 model...")
        model = YOLO(YOLO_MODEL_PATH)
        print("[Vision] YOLOv8 loaded")
    except Exception as e:
        print("[Vision] Failed to load YOLOv8:", e)
        return

    cap = cv2.VideoCapture(ESP32_CAM_STREAM)
    if not cap.isOpened():
        print("[Vision] Failed to open camera stream")
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
        # Show live feed even when skipping YOLO to keep it responsive
        if now - last_proc < 1.0 / CAM_PROC_FPS:
            cv2.imshow("ESP32-CAM YOLOv8", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
            time.sleep(0.01)
            continue

        last_proc = now

        fh, fw = frame.shape[:2]
        small = cv2.resize(frame, (320, 320))

        # YOLOv8 inference
        try:
            results = model(small, imgsz=320, conf=0.4, verbose=False)
        except Exception as e:
            print("[Vision] YOLO inference failed:", e)
            cv2.imshow("ESP32-CAM YOLOv8", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
            continue

        best_box = None
        best_conf = 0.0
        r = results[0]

        if r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id]

                if label not in TRASH_LABELS:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w = x2 - x1
                h = y2 - y1
                if w < 8 or h < 8:
                    continue

                if conf > best_conf:
                    best_conf = conf
                    best_box = (int(x1), int(y1), int(w), int(h))

        # Scale to full frame & update shared state
        full_box = None
        if best_box:
            sx = fw / 320.0
            sy = fh / 320.0
            x, y, w, h = best_box
            full_box = (int(x * sx), int(y * sy), int(w * sx), int(h * sy))
            shared.update(full_box, best_conf, fw, fh)
        else:
            shared.update(None, 0.0, fw, fh)

        # --- Live debug window: draw boxes / confidence ---
        debug_frame = frame.copy()
        if full_box:
            bx, by, bw, bh = full_box
            cv2.rectangle(debug_frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            
            # Show size information
            size_text = f"{bw}x{bh}"
            cv2.putText(
                debug_frame,
                f"{best_conf:.2f} {size_text}",
                (bx, by - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Draw center line and deadzone
            frame_center = fw // 2
            cv2.line(debug_frame, (frame_center, 0), (frame_center, fh), (255, 0, 0), 2)
            cv2.line(debug_frame, (frame_center - FRAME_CENTER_DEADZONE, 0), 
                    (frame_center - FRAME_CENTER_DEADZONE, fh), (0, 255, 255), 1)
            cv2.line(debug_frame, (frame_center + FRAME_CENTER_DEADZONE, 0), 
                    (frame_center + FRAME_CENTER_DEADZONE, fh), (0, 255, 255), 1)
            
            # Draw size threshold indicator
            if bw >= MIN_TRASH_WIDTH and bh >= MIN_TRASH_HEIGHT:
                cv2.putText(
                    debug_frame,
                    "SIZE OK - READY TO COLLECT",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        cv2.imshow("ESP32-CAM YOLOv8", debug_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[Vision] Stopped")


class IntegratedNode(Node):
    def __init__(self):
        super().__init__("integrated_vision_lidar_arm")

        # parameters
        self.declare_parameter("esp32_ip", ESP32_MOTOR_IP)
        self.declare_parameter("esp32_port", ESP32_MOTOR_PORT)
        self.declare_parameter("bin_full_threshold", BIN_FULL_DISTANCE_CM)

        self.esp32_ip = self.get_parameter("esp32_ip").get_parameter_value().string_value
        self.esp32_port = int(self.get_parameter("esp32_port").get_parameter_value().integer_value)
        self.bin_full_threshold = float(self.get_parameter("bin_full_threshold").get_parameter_value().double_value)

        # TCP client to ESP32-S3
        self.cli = ESP32Client(self.esp32_ip, self.esp32_port)
        try:
            self.cli.connect()
        except Exception as e:
            self.get_logger().warn(f"ESP32 connect failed: {e}")

        # QoS for LiDAR
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.sub = self.create_subscription(LaserScan, "/scan", self.on_scan, qos)

        # last scan storage
        self.last_pairs = None  # list of (angle_deg, dist)

        # vision thread
        self.vision = SharedVisionState()
        self.stop_vision = threading.Event()
        self.vision_thread = threading.Thread(
            target=vision_worker,
            args=(self.vision, self.stop_vision),
            daemon=True
        )
        self.vision_thread.start()

        # state machine
        self.mode = "SEARCH"  # SEARCH, APPROACH, COLLECT, DROP
        self.last_cmd_time = 0.0
        self.cmd_cooldown_ms = 120

        # track current servo angles
        self.base_angle = SEARCH_BASE
        self.shoulder_angle = SEARCH_SHOULDER
        self.wrist_angle = SEARCH_WRIST

        # collection sequence state
        self.collection_start_time = 0.0
        self.collection_phase = 0

        # stability tracking
        self.centered_frames = 0
        
        # obstacle avoidance tracking
        self.obstacle_start_time = 0.0
        self.obstacle_detected = False

        # ultrasonic data
        self.last_ultrasonic_check = 0.0
        self.us_check_interval = 0.5  # seconds
        self.gripper_reached = False
        self.bin_full = False
        self.buzzer_state = False
        self.last_buzzer_toggle = 0.0
        self.buzzer_interval = 0.4  # seconds for blinking when bin full

        # timer for decisions
        self.timer = self.create_timer(0.1, self.control_loop)  # 10 Hz

        # initialize arm to search pose
        self.arm_to_search_pose()

        self.get_logger().info("Integrated node started. Mode=SEARCH")

    # ---- low-level helpers ----
    def send_cmd(self, cmd: str, speed: int = None, expect_reply=False, timeout=3.0):
        now = time.time() * 1000
        if now - self.last_cmd_time < self.cmd_cooldown_ms and not expect_reply:
            return False, None
        self.last_cmd_time = now
        line = cmd if speed is None else f"{cmd} {int(speed)}"
        return self.cli.send((line + "\n").encode("ascii"), expect_reply, timeout)

    def servo_cmd(self, servo_id: int, angle: int, speed: int):
        angle = clamp(angle, 0, 180)
        speed = max(5, speed)
        cmd = f"SV {servo_id} {angle} {speed}"
        success, _ = self.cli.send((cmd + "\n").encode("ascii"), False)
        if success:
            # track servo angles only if command was successful
            if servo_id == SERVO_BASE:
                self.base_angle = angle
            elif servo_id == SERVO_SHOULDER:
                self.shoulder_angle = angle
            elif servo_id == SERVO_WRIST:
                self.wrist_angle = angle
        return success

    def update_ultrasonic_data(self):
        """Read ultrasonic data from ESP32"""
        now = time.time()
        if now - self.last_ultrasonic_check >= self.us_check_interval:
            us_data = self.cli.read_ultrasonic_data()
            self.last_ultrasonic_check = now
            
            # Check if gripper has reached object
            self.gripper_reached = (us_data.gripper_distance <= GRIPPER_DISTANCE_CM)
            
            # Check if bin is full
            self.bin_full = us_data.bin_full
            
            # Control buzzer based on bin level
            self.update_buzzer()
            
            # Log if bin is full
            if self.bin_full and not self.buzzer_state:
                self.get_logger().warn(f"TRASH BIN FULL! Distance: {us_data.bin_distance:.1f}cm")
    
    def start_collection_sequence(self):
        """Start final collection sequence"""
        self.collection_start_time = time.time()
        self.collection_phase = 1
        self.gripper_closed = False
        self.object_lifted = False
        self.object_dropped = False
        self.get_logger().info("Starting final collection sequence")
        
    def update_buzzer(self):
        """Control buzzer based on bin level"""
        now = time.time()
        
        if self.bin_full:
            # Blink buzzer when bin is full
            if now - self.last_buzzer_toggle >= self.buzzer_interval:
                self.buzzer_state = not self.buzzer_state
                self.cli.control_buzzer(self.buzzer_state)
                self.last_buzzer_toggle = now
        else:
            # Turn off buzzer when bin is not full
            if self.buzzer_state:
                self.buzzer_state = False
                self.cli.control_buzzer(False)

    # ---- arm poses ----
    def arm_to_search_pose(self):
        """Move arm to neutral search position (raised up)"""
        self.servo_cmd(SERVO_BASE, SEARCH_BASE, SERVO_SLOW)
        self.servo_cmd(SERVO_SHOULDER, SEARCH_SHOULDER, SERVO_SLOW)
        self.servo_cmd(SERVO_WRIST, SEARCH_WRIST, SERVO_SLOW)
        self.servo_cmd(SERVO_GRIP, GRIP_OPEN, SERVO_SLOW)

    def arm_to_collection_pose(self):
        """Prepare arm for collection (lower to prepare for reach)"""
        self.servo_cmd(SERVO_BASE, PREPARE_BASE, SERVO_MED)
        self.servo_cmd(SERVO_SHOULDER, PREPARE_SHOULDER, SERVO_MED)
        self.servo_cmd(SERVO_GRIP, GRIP_OPEN, SERVO_MED)

    def start_collection_sequence(self):
        """Start the timed collection sequence"""
        self.collection_start_time = time.time()
        self.collection_phase = 1
        self.get_logger().info("Starting collection sequence")

    def update_collection_sequence(self):
        """Execute collection sequence phases with timing"""
        if self.collection_phase == 0:
            return False

        elapsed = time.time() - self.collection_start_time

        if self.collection_phase == 1:  # Reach forward/down to trash
            if elapsed < 2.0:
                # Extend both base and shoulder gradually to reach down/forward
                progress = elapsed / 2.0
                base_angle = PREPARE_BASE + int((REACH_BASE - PREPARE_BASE) * progress)
                shoulder_angle = PREPARE_SHOULDER + int((REACH_SHOULDER - PREPARE_SHOULDER) * progress)
                self.servo_cmd(SERVO_BASE, base_angle, SERVO_FAST)
                self.servo_cmd(SERVO_SHOULDER, shoulder_angle, SERVO_FAST)
                
                # MAINTAIN ARM POSITION TO TRASH DURING REACHING
                self.maintain_arm_to_trash_during_reach()
                
                # Check if gripper ultrasonic detects object
                if self.gripper_reached:
                    self.get_logger().info("Gripper reached object, moving to grab phase")
                    self.collection_phase = 2
            else:
                self.collection_phase = 2

        elif self.collection_phase == 2:  # Close gripper to grab trash
            self.servo_cmd(SERVO_GRIP, GRIP_CLOSED, SERVO_MED)
            if elapsed > 3.0:
                self.collection_phase = 3

        elif self.collection_phase == 3:  # Lift arm with trash
            if elapsed < 4.5:
                progress = (elapsed - 3.0) / 1.5
                base_angle = REACH_BASE + int((LIFT_BASE - REACH_BASE) * progress)
                shoulder_angle = REACH_SHOULDER + int((LIFT_SHOULDER - REACH_SHOULDER) * progress)
                self.servo_cmd(SERVO_BASE, base_angle, SERVO_MED)
                self.servo_cmd(SERVO_SHOULDER, shoulder_angle, SERVO_MED)
            else:
                self.collection_phase = 4

        elif self.collection_phase == 4:  # Move to drop position (back bin)
            self.servo_cmd(SERVO_BASE, DROP_BASE, SERVO_SLOW)
            self.servo_cmd(SERVO_SHOULDER, DROP_SHOULDER, SERVO_SLOW)
            if elapsed > 6.0:
                self.collection_phase = 5

        elif self.collection_phase == 5:  # Release gripper and return
            self.servo_cmd(SERVO_GRIP, GRIP_OPEN, SERVO_MED)
            if elapsed > 7.0:
                self.arm_to_search_pose()
                self.collection_phase = 0
                self.mode = "SEARCH"
                self.get_logger().info("Collection sequence completed - trash dropped in bin")
                return True

        return False

    def maintain_arm_to_trash_during_reach(self):
        """Adjust arm position during reaching to maintain alignment with trash"""
        detected, box, conf, ts, fw, fh, hist = self.vision_status()
        
        if not detected or box is None:
            return
            
        x, y, w, h = box
        cx = x + w / 2
        frame_center = fw / 2
        dx = cx - frame_center
        
        # Adjust arm base slightly based on trash position
        # This helps maintain alignment during the reach
        if abs(dx) > FRAME_CENTER_DEADZONE:
            base_adjust = int(dx * 0.05)  # Very small gain for fine adjustment
            new_base_angle = clamp(self.base_angle + base_adjust, 50, 130)
            self.servo_cmd(SERVO_BASE, new_base_angle, SERVO_FAST)

    # ---- LiDAR callback ----
    def on_scan(self, msg: LaserScan):
        ranges = list(msg.ranges)
        if not ranges or msg.angle_increment == 0.0:
            return
        angle_min = msg.angle_min
        inc = msg.angle_increment
        pairs = []
        for i, r in enumerate(ranges):
            ang = angle_min + i * inc
            deg = math.degrees(ang)
            if deg > 180.0:
                deg -= 360.0
            if r <= 0.0 or math.isinf(r) or math.isnan(r):
                continue
            pairs.append((deg, r))
        self.last_pairs = pairs

    # ---- Sector query ----
    def sector_min(self, center_deg, half_width):
        if not self.last_pairs:
            return float("inf")
        lo = center_deg - half_width
        hi = center_deg + half_width
        vals = [d for (a, d) in self.last_pairs if lo <= a <= hi]
        return min(vals) if vals else float("inf")

    # ---- stability check for vision ----
    def vision_status(self):
        with self.vision.lock:
            detected = self.vision.detected
            box = self.vision.box
            conf = self.vision.conf
            ts = self.vision.timestamp
            fw = self.vision.frame_width
            fh = self.vision.frame_height
            hist = list(self.vision.history)
        return detected, box, conf, ts, fw, fh, hist

    def is_trash_centered(self, box, fw):
        if not box:
            self.centered_frames = 0
            return False
        
        x, y, w, h = box
        cx = x + w / 2
        frame_center = fw / 2
        dx = cx - frame_center
        
        centered = abs(dx) <= FRAME_CENTER_DEADZONE
        
        if centered:
            self.centered_frames += 1
        else:
            self.centered_frames = 0
            
        return centered

    def is_trash_big_enough(self, box):
        """Check if trash is big enough in camera view to start collection"""
        if not box:
            return False
            
        x, y, w, h = box
        return w >= MIN_TRASH_WIDTH and h >= MIN_TRASH_HEIGHT

    def center_trash_with_robot(self, box, fw):
        """Center trash by rotating robot"""
        if not box:
            return "centered"  # Changed from None to "centered"
        
        x, y, w, h = box
        cx = x + w / 2
        frame_center = fw / 2
        dx = cx - frame_center
        
        if abs(dx) > FRAME_CENTER_DEADZONE:
            if dx < 0:
                self.send_cmd("L", CENTERING_TURN_SPEED)
                self.last_centering_action = "turning_left"
                return "turning_left"
            else:
                self.send_cmd("R", CENTERING_TURN_SPEED)
                self.last_centering_action = "turning_right"
                return "turning_right"
        else:
            # Only send stop if we were previously turning
            if self.last_centering_action in ["turning_left", "turning_right"]:
                self.send_cmd("S")
                self.last_centering_action = "centered"
            return "centered"

    def check_obstacle_timeout(self):
        """Check if we've been stuck on an obstacle for too long"""
        if self.obstacle_detected:
            if time.time() - self.obstacle_start_time > OBSTACLE_TIMEOUT:
                self.get_logger().info("Obstacle timeout - abandoning current trash and resuming search")
                self.obstacle_detected = False
                self.mode = "SEARCH"
                self.arm_to_search_pose()
                return True
        return False

    # ---- main control loop (10 Hz) ----
    def control_loop(self):
        # Update ultrasonic data
        self.update_ultrasonic_data()
        
        # Check connection status
        if not self.cli.sock:
            self.get_logger().warn("ESP32 disconnected, attempting reconnect...")
            try:
                self.cli.connect()
            except Exception as e:
                self.get_logger().warn(f"Reconnect failed: {e}")
            return  # Skip this control cycle if no connection

        # If we're in collection sequence, let it run
        if self.collection_phase > 0:
            self.send_cmd("S")  # Stop base movement during collection
            self.update_collection_sequence()
            return

        # LiDAR sectors for navigation
        front_min = self.sector_min(0.0, 45.0)
        left_min = self.sector_min(90.0, 45.0)
        right_min = self.sector_min(-90.0, 45.0)

        detected, box, conf, ts, fw, fh, hist = self.vision_status()
        now = time.time()

        # emergency: very close obstacle
        if front_min < HARD_STOP_M:
            self.send_cmd("S")
            if left_min > right_min:
                self.send_cmd("L", EMERGENCY_TURN_SPEED)
            else:
                self.send_cmd("R", EMERGENCY_TURN_SPEED)
            return

        # Check if we should abandon due to obstacle timeout
        if self.check_obstacle_timeout():
            return

        # vision stale?
        if not detected or (now - ts) > 2.0:
            if self.mode != "SEARCH":
                self.mode = "SEARCH"
                self.arm_to_search_pose()
                self.centered_frames = 0
                self.obstacle_detected = False

        # -------- STATE MACHINE --------
        if self.mode == "SEARCH":
            self.arm_to_search_pose()
            self.obstacle_detected = False

            # Wander using LiDAR
            if front_min > MIN_CLEAR_M:
                self.send_cmd("F", SEARCH_FORWARD_SPEED)
            else:
                if left_min > right_min:
                    self.send_cmd("L", SEARCH_TURN_SPEED)
                else:
                    self.send_cmd("R", SEARCH_TURN_SPEED)

            # Transition to APPROACH when trash detected
            if detected and conf > 0.5:
                self.mode = "APPROACH"
                self.centered_frames = 0
                self.obstacle_detected = False
                self.get_logger().info("Trash detected → APPROACH")

        elif self.mode == "APPROACH":
            if not detected or box is None:
                self.mode = "SEARCH"
                self.obstacle_detected = False
                return

            # Continuously center trash using robot rotation
            self.center_trash_with_robot(box, fw)

            # Check safety distance using ultrasonic
            us_data = self.cli.get_ultrasonic_data()
            gripper_distance = us_data.gripper_distance

            # Obstacle detection and avoidance
            if front_min <= MIN_CLEAR_M:
                if not self.obstacle_detected:
                    self.obstacle_detected = True
                    self.obstacle_start_time = time.time()
                    self.get_logger().info("Obstacle detected - attempting to avoid")
                
                self.send_cmd("S")
                if left_min > right_min:
                    self.send_cmd("L", OBSTACLE_AVOID_SPEED)
                else:
                    self.send_cmd("R", OBSTACLE_AVOID_SPEED)
                return
            else:
                # Clear path - reset obstacle timer
                self.obstacle_detected = False

            # Move forward only when trash is centered and path is clear
            if self.is_trash_centered(box, fw) and front_min > MIN_CLEAR_M:
                self.send_cmd("F", APPROACH_FORWARD_SPEED)

                # CHECK IF TRASH IS BIG ENOUGH TO START COLLECTION
                if (self.is_trash_big_enough(box) and 
                    self.centered_frames >= STABLE_FRAMES_REQUIRED and
                    gripper_distance > ULTRASONIC_SAFETY_CM):
                    
                    self.send_cmd("S")  # Stop robot
                    self.mode = "COLLECT"
                    self.arm_to_collection_pose()
                    self.obstacle_detected = False
                    self.get_logger().info(
                        f"Trash size {box[2]}x{box[3]} - Starting collection"
                    )

        elif self.mode == "COLLECT":
            # Robot remains stopped, arm takes over
            self.send_cmd("S")

            # Start collection immediately when in COLLECT mode
            # (we already checked size and centering in APPROACH mode)
            self.start_collection_sequence()

    def destroy_node(self):
        self.stop_vision.set()
        self.vision_thread.join(timeout=2.0)
        self.cli.control_buzzer(False)  # Ensure buzzer is off
        self.cli.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = IntegratedNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down...")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
