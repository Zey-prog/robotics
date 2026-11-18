#!/usr/bin/env python3
# integrated_vision_lidar_arm_final.py
#
# FINAL VERSION WITH PROPER ULTRASONIC LOGIC AND VISUAL BOUNDING BOXES:
# 1. Camera detects trash FIRST → then use TRASH ultrasonic
# 2. BIN ultrasonic runs constantly → emergency stop if full
# 3. GRIPPER ultrasonic only during collection
# 4. Visual display with bounding boxes and labels

import socket
import time
import math
import threading
import sys
import select
import signal
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto

import cv2
import numpy as np
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan

# ---------------- CONFIG ----------------
ESP32_MOTOR_IP = "192.168.1.101"
ESP32_MOTOR_PORT = 5050
ESP32_CAM_STREAM = "http://192.168.1.100:81/stream"
YOLO_MODEL_PATH = "best.pt"

# ========== NAVIGATION PARAMETERS ==========
MIN_CLEAR_M = 0.55          # Minimum clearance to move forward
CLOSE_APPROACH_THRESHOLD = 0.45  # For APPROACH mode - allow closer
HARD_STOP_M = 0.35          # Emergency stop distance



FRONT_SECTOR_ANGLE = 45.0   # Degrees to check in front (45° each side)
SIDE_SECTOR_ANGLE = 40.0    # Degrees to check on sides
OBSTACLE_TIMEOUT = 3.0      # Seconds to try avoiding obstacle before timeout

# ========== ULTRASONIC PARAMETERS ==========
TRIGGER_COLLECTION_DISTANCE_CM = 37.0     # TRASH sensor - triggers collection AFTER camera detection
GRIPPER_GRAB_DISTANCE_CM = 7.0           # GRIPPER sensor - closes gripper during collection
GRIPPER_RELEASE_DISTANCE_CM = 4.0
ULTRASONIC_SAFETY_CM = 1.0
ULTRASONIC_CONTINUE_APPROACH_CM = 14.0
BIN_FULL_DISTANCE_CM = 5.0              # BIN sensor - CONSTANT m  onitoring, emergency stop if ≤ this
BIN_CHECK_INTERVAL = 2.5                 # Check bin every 500ms (fast for safety)

# ========== VISION PARAMETERS ==========
FRAME_CENTER_DEADZONE = 30
STABLE_FRAMES_REQUIRED = 3
CAM_PROC_FPS = 20

# ========== VISUALIZATION PARAMETERS ==========
BOX_COLOR = (0, 255, 0)           # Green for detected objects
BOX_THICKNESS = 2
TEXT_COLOR = (0, 255, 0)          # Green text
TEXT_BG_COLOR = (0, 0, 0)         # Black background for text
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.6
TEXT_THICKNESS = 2
CENTER_LINE_COLOR = (255, 0, 0)   # Blue center line
CENTER_LINE_THICKNESS = 1
DEADZONE_COLOR = (0, 0, 255)      # Red deadzone markers
DEADZONE_ALPHA = 0.2              # Transparency for deadzone overlay
STATUS_TEXT_COLOR = (255, 255, 0) # Yellow for status text
STATUS_TEXT_POS = (10, 30)        # Position for status display

# ========== MOTOR SPEEDS (0-255) ==========
SEARCH_FORWARD_SPEED = 130
SEARCH_TURN_SPEED = 190
APPROACH_FORWARD_SPEED = 100
SLOW_APPROACH_SPEED = 90
ULTRA_SLOW_APPROACH_SPEED = 60
CENTERING_TURN_SPEED = 1
OBSTACLE_AVOID_SPEED = 190
EMERGENCY_TURN_SPEED = 180

# ========== SERVO CONFIG ==========
SERVO_BASE = 0
SERVO_SHOULDER = 1
SERVO_GRIP = 2
SERVO_WRIST = 3

# Servo angles
SEARCH_BASE = 140
SEARCH_SHOULDER = 180
SEARCH_WRIST = 90
GRIP_OPEN = 40
GRIP_CLOSED = 110

# Collection poses
PREPARE_BASE = 110
PREPARE_SHOULDER = 120
REACH_BASE = 60
REACH_SHOULDER = 130
LIFT_BASE = 35
LIFT_SHOULDER = 10

TIMEOUT_BASE = 100      # Safer intermediate position for base
TIMEOUT_SHOULDER = 150  # Safer intermediate position for shoulder
TIMEOUT_WRIST = 90      # Neutral wrist position

# Drop poses
DROP_BASE = 80
DROP_SHOULDER = 1

# Wrist rotation angles
WRIST_START_ANGLE = 90
WRIST_LIFT_ANGLE = 90
WRIST_DROP_ANGLE = 90

# Servo speeds
SERVO_SLOW = 30
SERVO_MED = 30
SERVO_FAST = 30

# YOLO label mapping (update based on your model's class names)
TRASH_LABELS = {
    0: "Clear plastic bottle",
    1: "Other plastic bottle",
    2: "Drink can",
    3: "Food Can",
    4: "Plastic lid",
    5: "Plastic film",
    6: "Paper cup",
    7: "Disposable plastic cup",
    8: "Crisp packet",
    9: "Garbage bag",
    10: "Magazine paper",
    11: "Tissues",
    12: "Wrapping paper",
    13: "Normal paper",
    14: "Paper bag",
    15: "Plastified paper bag"
}


# ----------------------------------------------------

class RobotMode(Enum):
    """Robot operation modes"""
    SEARCH = auto()           # Searching for trash with camera
    CENTERING = auto()        # Centering trash in camera view
    APPROACH = auto()         # Approaching trash (using TRASH ultrasonic AFTER camera detection)
    COLLECT_APPROACH = auto() # Final approach (using GRIPPER ultrasonic)
    COLLECT = auto()          # Collection sequence
    EMERGENCY = auto()        # Bin full emergency stop
    MANUAL = auto()           # Manual control


@dataclass
class UltrasonicData:
    """Container for ultrasonic sensor data"""
    gripper_distance: float = 999.0      # Only used during collection
    trash_distance: float = 999.0        # Only used AFTER camera detection
    bin_distance: float = 999.0          # CONSTANTLY monitored
    timestamp: float = 0.0
    bin_full: bool = False


class ServoSequence:
    """Non-blocking servo sequence controller"""
    
    def __init__(self, client, node):
        self.client = client
        self.node = node
        self.sequence_queue = []
        self.current_sequence = None
        self.start_time = 0
        self.running = False
        
    def arm_to_search_pose(self):
        """Move arm to search pose"""
        self.client.servo_cmd(SERVO_SHOULDER, SEARCH_SHOULDER, SERVO_SLOW)
        self.client.servo_cmd(SERVO_BASE, SEARCH_BASE, SERVO_SLOW)
        time.sleep(0.5)
        self.client.servo_cmd(SERVO_WRIST, SEARCH_WRIST, SERVO_SLOW)
        self.client.servo_cmd(SERVO_GRIP, GRIP_OPEN, SERVO_SLOW)
    
    def arm_to_collection_pose(self):
        """Move arm to collection pose"""
        self.client.servo_cmd(SERVO_BASE, PREPARE_BASE, SERVO_MED)
        self.client.servo_cmd(SERVO_SHOULDER, PREPARE_SHOULDER, SERVO_MED)
        time.sleep(1.0)
        self.client.servo_cmd(SERVO_BASE, REACH_BASE, SERVO_MED)
        self.client.servo_cmd(SERVO_SHOULDER, REACH_SHOULDER, SERVO_MED)
        time.sleep(1.0)
        self.client.servo_cmd(SERVO_GRIP, GRIP_OPEN, SERVO_MED)


class Watchdog:
    """Safety watchdog timer"""
    
    def __init__(self, timeout, callback, node):
        self.timeout = timeout
        self.callback = callback
        self.node = node
        self.last_reset = time.time()
        self.active = True
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.thread.start()
    
    def _watch(self):
        while self.active:
            if time.time() - self.last_reset > self.timeout:
                self.node.get_logger().error(f"Watchdog timeout! No reset for {self.timeout}s")
                self.callback()
                break
            time.sleep(0.1)
    
    def reset(self):
        self.last_reset = time.time()
    
    def stop(self):
        self.active = False


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class ESP32Client:
    def __init__(self, host, port=5050, timeout=5.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock = None
        self.lock = threading.Lock()
        self.ultrasonic_data = UltrasonicData()
        
    def connect(self):
        """Connect to ESP32"""
        with self.lock:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(self.timeout)
                self.sock.connect((self.host, self.port))
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                print(f"[ESP32Client] Connected to {self.host}:{self.port}")
                return True
            except Exception as e:
                print(f"[ESP32Client] Connection failed: {e}")
                self.sock = None
                return False
    
    def send(self, payload: bytes, expect_reply=False, reply_timeout=2.0):
        with self.lock:
            if not self.sock:
                if not self.connect():
                    return False, None
            
            try:
                self.sock.sendall(payload)
                if expect_reply:
                    self.sock.settimeout(reply_timeout)
                    data = b""
                    start_time = time.time()
                    while time.time() - start_time < reply_timeout:
                        try:
                            chunk = self.sock.recv(1024)
                            if not chunk:
                                break
                            data += chunk
                            if b"\n" in chunk:
                                break
                        except socket.timeout:
                            break
                    
                    return True, data.decode(errors="ignore").strip()
                return True, None
            except Exception as e:
                print(f"[ESP32Client] Send failed: {e}")
                try:
                    self.sock.close()
                except:
                    pass
                self.sock = None
                return False, None
    
    def servo_cmd(self, servo_id: int, angle: int, speed: int):
        """Send servo command"""
        angle = clamp(angle, 0, 180)
        speed = max(5, speed)
        cmd = f"SV {servo_id} {angle} {speed}\n"
        success, _ = self.send(cmd.encode("ascii"), False)
        return success
    
    def read_gripper_distance(self):
        """Read gripper ultrasonic (ONLY during collection)"""
        now = time.time()
        ok, reply = self.send(b"GRIPPER\n", expect_reply=True, reply_timeout=1.0)
        gripper_dist = 999.0
        if ok and reply and reply.startswith("GRIPPER:"):
            try:
                gripper_dist = float(reply.split(":")[1])
            except:
                gripper_dist = 999.0
        
        self.ultrasonic_data.gripper_distance = gripper_dist
        self.ultrasonic_data.timestamp = now
        return gripper_dist
    
    def read_trash_distance(self):
        """Read trash ultrasonic (ONLY AFTER camera detection)"""
        now = time.time()
        ok, reply = self.send(b"TRASH\n", expect_reply=True, reply_timeout=1.0)
        trash_dist = 999.0
        if ok and reply and reply.startswith("TRASH:"):
            try:
                trash_dist = float(reply.split(":")[1])
            except:
                trash_dist = 999.0
        
        self.ultrasonic_data.trash_distance = trash_dist
        self.ultrasonic_data.timestamp = now
        return trash_dist
    
    def read_bin_distance(self):
        """Read bin ultrasonic (CONSTANT monitoring)"""
        now = time.time()
        ok, reply = self.send(b"BIN\n", expect_reply=True, reply_timeout=1.0)
        bin_dist = 999.0
        if ok and reply and reply.startswith("BIN:"):
            try:
                bin_dist = float(reply.split(":")[1])
            except:
                bin_dist = 999.0
        
        self.ultrasonic_data.bin_distance = bin_dist
        self.ultrasonic_data.bin_full = (bin_dist <= BIN_FULL_DISTANCE_CM)
        self.ultrasonic_data.timestamp = now
        return bin_dist
    
    def read_all_ultrasonic(self):
        """Read all sensors at once"""
        now = time.time()
        ok, reply = self.send(b"ALL\n", expect_reply=True, reply_timeout=1.5)
        
        if ok and reply:
            try:
                parts = reply.split()
                for part in parts:
                    if ":" in part:
                        name, value = part.split(":", 1)
                        try:
                            val = float(value)
                            if name == "GRIPPER":
                                self.ultrasonic_data.gripper_distance = val
                            elif name == "TRASH":
                                self.ultrasonic_data.trash_distance = val
                            elif name == "BIN":
                                self.ultrasonic_data.bin_distance = val
                                self.ultrasonic_data.bin_full = (val <= BIN_FULL_DISTANCE_CM)
                        except:
                            pass
                
                self.ultrasonic_data.timestamp = now
                return True
            except:
                pass
        
        return False
    
    def get_ultrasonic_data(self):
        return self.ultrasonic_data
    
    def control_buzzer(self, state: bool):
        cmd = "BUZZER ON\n" if state else "BUZZER OFF\n"
        success, _ = self.send(cmd.encode("ascii"), False)
        return success
    
    def close(self):
        with self.lock:
            if self.sock:
                try:
                    self.sock.close()
                except:
                    pass
                self.sock = None


class DetectionResult:
    """Container for detection results with visualization info"""
    def __init__(self):
        self.box = None           # (x, y, w, h)
        self.confidence = 0.0
        self.class_id = -1
        self.class_name = ""
        self.timestamp = 0.0
        self.centered = False
        self.distance_to_center = 0


class SharedVisionState:
    def __init__(self):
        self.lock = threading.Lock()
        self.detected = False
        self.detection_result = DetectionResult()
        self.frame_width = 480
        self.frame_height = 320
        self.display_frame = None
        self.history = deque(maxlen=10)
    
    def update(self, detection_result, fw, fh, display_frame=None):
        now = time.time()
        with self.lock:
            self.detected = detection_result.box is not None
            self.detection_result = detection_result
            self.detection_result.timestamp = now
            self.frame_width = fw
            self.frame_height = fh
            if display_frame is not None:
                self.display_frame = display_frame.copy()
            if detection_result.box:
                x, y, w, h = detection_result.box
                cx = x + w / 2
                cy = y + h / 2
                self.history.append((now, cx, cy))
    
    def get_display_frame(self):
        with self.lock:
            return self.display_frame.copy() if self.display_frame is not None else None


def draw_detections(frame, detection_result, mode_name, ultrasonic_data=None, centered=False):
    """
    Draw bounding boxes, labels, and additional information on frame
    
    Args:
        frame: Input frame
        detection_result: DetectionResult object
        mode_name: Current robot mode name
        ultrasonic_data: UltrasonicData object
        centered: Whether the object is centered
    """
    display_frame = frame.copy()
    h, w = display_frame.shape[:2]
    
    # Draw center line
    center_x = w // 2
    cv2.line(display_frame, (center_x, 0), (center_x, h), CENTER_LINE_COLOR, CENTER_LINE_THICKNESS)
    
    # Draw deadzone overlay (semi-transparent)
    deadzone_overlay = display_frame.copy()
    left_deadzone = center_x - FRAME_CENTER_DEADZONE
    right_deadzone = center_x + FRAME_CENTER_DEADZONE
    cv2.rectangle(deadzone_overlay, (left_deadzone, 0), (right_deadzone, h), DEADZONE_COLOR, -1)
    cv2.addWeighted(deadzone_overlay, DEADZONE_ALPHA, display_frame, 1 - DEADZONE_ALPHA, 0, display_frame)
    
    # Draw detection if present
    if detection_result.box is not None:
        x, y, box_w, box_h = detection_result.box
        
        color = BOX_COLOR
        
        # Draw bounding box
        cv2.rectangle(display_frame, (x, y), (x + box_w, y + box_h), color, BOX_THICKNESS)
        
        # Prepare label text
        label = f"{detection_result.class_name}: {detection_result.confidence:.2f}"
        
        # Calculate text size
        (text_width, text_height), baseline = cv2.getTextSize(label, TEXT_FONT, TEXT_SCALE, TEXT_THICKNESS)
        
        # Draw label background
        label_bg_top_left = (x, y - text_height - baseline - 5)
        label_bg_bottom_right = (x + text_width, y)
        cv2.rectangle(display_frame, label_bg_top_left, label_bg_bottom_right, TEXT_BG_COLOR, -1)
        
        # Draw label text
        cv2.putText(display_frame, label, (x, y - baseline - 5), 
                   TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)
        
        # Draw center dot of the bounding box
        center_x_box = x + box_w // 2
        center_y_box = y + box_h // 2
        cv2.circle(display_frame, (center_x_box, center_y_box), 4, (0, 0, 255), -1)
        
        # Draw line from frame center to object center
        cv2.line(display_frame, (center_x, h//2), (center_x_box, center_y_box), (255, 255, 0), 1)
        
        # Add centered status
        if centered:
            cv2.putText(display_frame, "CENTERED", (center_x_box + 10, center_y_box),
                       TEXT_FONT, 0.5, (0, 255, 0), 1)
    
    # Add status overlay at top
    status_bar = np.zeros((50, w, 3), dtype=np.uint8)
    
    # Mode information
    mode_text = f"Mode: {mode_name}"
    cv2.putText(status_bar, mode_text, (10, 30), TEXT_FONT, 0.5, STATUS_TEXT_COLOR, 1)
    
    # Ultrasonic information if available
    if ultrasonic_data is not None:
        us_text = f"Bin: {ultrasonic_data.bin_distance:.1f}cm | Trash: {ultrasonic_data.trash_distance:.1f}cm"
        cv2.putText(status_bar, us_text, (w - 400, 30), TEXT_FONT, 0.5, STATUS_TEXT_COLOR, 1)
        
        # Bin full warning
        if ultrasonic_data.bin_full:
            warning_text = "BIN FULL!"
            text_size = cv2.getTextSize(warning_text, TEXT_FONT, 0.7, 2)[0]
            warning_x = w // 2 - text_size[0] // 2
            cv2.putText(status_bar, warning_text, (warning_x, 30), TEXT_FONT, 0.7, (0, 0, 255), 2)
    
    # Add detection status
    if detection_result.box is not None:
        detect_text = f"Detected: {detection_result.class_name}"
        cv2.putText(status_bar, detect_text, (10, 45), TEXT_FONT, 0.5, (0, 255, 0), 1)
    else:
        cv2.putText(status_bar, "No detection", (10, 45), TEXT_FONT, 0.5, (0, 0, 255), 1)
    
    # Combine status bar with main frame
    display_frame = cv2.vconcat([status_bar, display_frame])
    
    return display_frame


def vision_worker(shared: SharedVisionState, stop_event: threading.Event, node):
    """Vision processing worker with visualizations"""
    try:
        model = YOLO(YOLO_MODEL_PATH)
        node.get_logger().info("YOLOv8 loaded")
    except Exception as e:
        node.get_logger().error(f"Failed to load YOLOv8: {e}")
        return
    
    reconnect_count = 0
    
    while not stop_event.is_set():
        cap = cv2.VideoCapture(ESP32_CAM_STREAM)
        if not cap.isOpened():
            reconnect_count += 1
            node.get_logger().warn(f"Camera reconnect attempt {reconnect_count}")
            time.sleep(1.0)
            continue
        
        reconnect_count = 0
        
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            
            fh, fw = frame.shape[:2]
            display_frame = frame.copy()
            
            # YOLOv8 inference
            try:
                small = cv2.resize(frame, (320, 320))
                results = model(small, imgsz=320, conf=0.4, verbose=False)

                best_box = None
                best_conf = 0.0
                best_class_id = -1
                best_class_name = ""
                r = results[0]
                
                if r.boxes is not None:
                    for box in r.boxes:
                        conf = float(box.conf[0])

                        # Confidence threshold
                        if conf < 0.55:
                            continue

                        # Get class information
                        cls_id = int(box.cls[0])
                        class_name = TRASH_LABELS.get(cls_id, f"Class {cls_id}")

                        # Get bounding box
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        w = x2 - x1
                        h = y2 - y1

                        # Ignore tiny noise detections
                        if w < 8 or h < 8:
                            continue

                        # Keep best detection
                        if conf > best_conf:
                            best_conf = conf
                            best_box = (int(x1), int(y1), int(w), int(h))
                            best_class_id = cls_id
                            best_class_name = class_name
                
                # Scale to full frame and prepare detection result
                detection_result = DetectionResult()
                if best_box:
                    sx = fw / 320.0
                    sy = fh / 320.0
                    x, y, w, h = best_box
                    full_box = (int(x * sx), int(y * sy), int(w * sx), int(h * sy))
                    
                    detection_result.box = full_box
                    detection_result.confidence = best_conf
                    detection_result.class_id = best_class_id
                    detection_result.class_name = best_class_name
                    
                    # Check if centered
                    cx_obj = full_box[0] + full_box[2] // 2
                    frame_center = fw // 2
                    detection_result.distance_to_center = abs(cx_obj - frame_center)
                    detection_result.centered = detection_result.distance_to_center <= FRAME_CENTER_DEADZONE
                
                # Update shared state with visualization frame
                shared.update(detection_result, fw, fh, display_frame)
                
            except Exception as e:
                node.get_logger().error(f"YOLO inference failed: {e}")
                # Still update with empty detection
                detection_result = DetectionResult()
                shared.update(detection_result, fw, fh, display_frame)
                continue
        
        cap.release()
        if not stop_event.is_set():
            time.sleep(1.0)
    
    cv2.destroyAllWindows()
    node.get_logger().info("Vision thread stopped")


class IntegratedNode(Node):
    def __init__(self):
        super().__init__("integrated_vision_lidar_arm")
        
        # Signal handler
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Parameters
        self.declare_parameter("esp32_ip", ESP32_MOTOR_IP)
        self.declare_parameter("esp32_port", ESP32_MOTOR_PORT)
        self.declare_parameter("bin_full_threshold", BIN_FULL_DISTANCE_CM)
        
        self.esp32_ip = self.get_parameter("esp32_ip").get_parameter_value().string_value
        self.esp32_port = int(self.get_parameter("esp32_port").get_parameter_value().integer_value)
        self.bin_full_threshold = float(self.get_parameter("bin_full_threshold").get_parameter_value().double_value)
        
        # TCP client to ESP32
        self.cli = ESP32Client(self.esp32_ip, self.esp32_port)
        
        # Servo sequence controller
        self.servo_controller = ServoSequence(self.cli, self)
        
        # Connect to ESP32
        try:
            if self.cli.connect():
                self.get_logger().info("ESP32 connected")
        except Exception as e:
            self.get_logger().warn(f"ESP32 connect failed: {e}")
        
        # LiDAR subscription
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.sub = self.create_subscription(LaserScan, "/scan", self.on_scan, qos)
        self.last_pairs = None
        
        # Vision thread
        self.vision = SharedVisionState()
        self.stop_vision = threading.Event()
        self.vision_thread = threading.Thread(
            target=vision_worker,
            args=(self.vision, self.stop_vision, self),
            daemon=True
        )
        self.vision_thread.start()
        
        # Display thread for showing annotated frames
        self.display_thread = threading.Thread(target=self.display_worker, daemon=True)
        self.display_thread.start()
        
        # State machine
        self.mode = RobotMode.SEARCH
        self.last_cmd_time = 0.0
        self.cmd_cooldown_ms = 80
        
        # Manual/Autonomous control
        self.autonomous_mode = True
        
        # Collection sequence state
        self.collection_start_time = 0.0
        self.collection_phase = 0
        self.gripper_closed = False
        self.object_lifted = False
        self.object_dropped = False
        self.arm_in_collection_pose = False
        
        # Vision tracking
        self.detection_lost_time = 0.0
        self.detection_timeout = 5.0
        
        # Stability tracking
        self.centered_frames = 0
        self.last_centered_time = 0.0
        
        # Obstacle avoidance
        self.obstacle_start_time = 0.0
        self.obstacle_detected = False
        
        # Ultrasonic data
        self.last_bin_check = 0.0
        self.last_trash_check = 0.0
        self.last_gripper_check = 0.0
        self.bin_check_interval = BIN_CHECK_INTERVAL  # 500ms for safety
        self.trash_check_interval = 0.3  # Only after camera detection
        self.gripper_check_interval = 0.3  # Only during collection
        
        # Bin full state
        self.bin_full = False
        self.buzzer_state = False
        self.last_buzzer_toggle = 0.0
        self.buzzer_interval = 0.2  # Rapid beeping when bin full
        
        # Motor control
        self.last_motor_speed = 0
        self.last_centering_action = "centered"
        
        # Movement control
        self.robot_movement_allowed = True
        
        # Watchdog
        self.watchdog = Watchdog(timeout=10.0, callback=self.watchdog_timeout, node=self)
        
        # Timer for decisions
        self.timer = self.create_timer(0.1, self.control_loop)
        
        # Keyboard input
        self.keyboard_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        self.keyboard_thread.start()
        
        # Display control
        self.display_enabled = True
        self.last_display_update = 0.0
        
        # Initialize arm to search pose
        self.servo_controller.arm_to_search_pose()
        
        self.get_logger().info("=== SYSTEM STARTED ===")
        self.get_logger().info(f"Bin monitoring: Every {BIN_CHECK_INTERVAL}s, emergency at {BIN_FULL_DISTANCE_CM}cm")
        self.get_logger().info("Logic: Camera FIRST → then TRASH ultrasonic → then GRIPPER ultrasonic")
    
    def signal_handler(self, signum, frame):
        """Handle termination signals"""
        self.get_logger().info(f"Signal {signum} received, shutting down...")
        self.destroy_node()
    
    def watchdog_timeout(self):
        """Watchdog timeout callback"""
        self.get_logger().error("WATCHDOG TIMEOUT - Emergency stop!")
        self.send_cmd("S")
        self.mode = RobotMode.EMERGENCY
    
    def display_worker(self):
        """Display worker thread for showing annotated frames"""
        while rclpy.ok():
            if not self.display_enabled:
                time.sleep(0.1)
                continue
            
            display_frame = self.vision.get_display_frame()
            if display_frame is not None:
                # Get current ultrasonic data
                us_data = self.cli.get_ultrasonic_data()
                
                # Get current detection result
                with self.vision.lock:
                    detection_result = self.vision.detection_result
                
                # Create annotated frame
                annotated_frame = draw_detections(
                    display_frame, 
                    detection_result,
                    self.mode.name,
                    us_data,
                    detection_result.centered
                )
                
                # Show the frame
                cv2.imshow("Trash Detection - ESP32-CAM YOLOv8", annotated_frame)
                cv2.waitKey(1) & 0xFF
                
                self.last_display_update = time.time()
            
            time.sleep(0.033)  # ~30 FPS for display
    
    def transition_to(self, new_mode, reason=""):
        """Transition between modes"""
        old_mode = self.mode
        self.mode = new_mode
        
        if new_mode == RobotMode.SEARCH:
            self.centered_frames = 0
            self.obstacle_detected = False
            self.detection_lost_time = 0.0
            self.robot_movement_allowed = True
            self.send_cmd("S")
            self.servo_controller.arm_to_search_pose()
                
        elif new_mode == RobotMode.CENTERING:
            self.centered_frames = 0
            self.obstacle_detected = False
            self.detection_lost_time = 0.0
            self.robot_movement_allowed = True
            
        elif new_mode == RobotMode.APPROACH:
            self.approach_start_time = time.time()
            self.robot_movement_allowed = True
            
        elif new_mode == RobotMode.COLLECT_APPROACH:
            self.collection_start_time = time.time()
            self.robot_movement_allowed = False
            self.send_cmd("S")
            self.servo_controller.arm_to_collection_pose()
                
        elif new_mode == RobotMode.COLLECT:
            self.start_collection_sequence()
            self.robot_movement_allowed = False
            
        elif new_mode == RobotMode.EMERGENCY:
            self.send_cmd("S")
            self.autonomous_mode = False
            self.robot_movement_allowed = False
            self.cli.control_buzzer(True)
            self.buzzer_state = True
            self.get_logger().error(f"EMERGENCY: {reason}")
            
        elif new_mode == RobotMode.MANUAL:
            self.send_cmd("S")
            self.robot_movement_allowed = False
        
        if old_mode != new_mode:
            self.get_logger().info(f"Mode: {old_mode.name} → {new_mode.name} ({reason})")
    
    def keyboard_listener(self):
        """Listen for keyboard input"""
        while rclpy.ok():
            if select.select([sys.stdin], [], [], 0.1)[0]:
                key = sys.stdin.read(1).lower()
                if key == 'q':
                    self.get_logger().info("Quitting...")
                    self.send_cmd("S")
                    self.destroy_node()
                elif key == 'e':
                    self.autonomous_mode = not self.autonomous_mode
                    if self.autonomous_mode:
                        self.transition_to(RobotMode.SEARCH, "Auto mode")
                    else:
                        self.transition_to(RobotMode.MANUAL, "Manual mode")
                    self.get_logger().info(f"Autonomous: {self.autonomous_mode}")
                elif key == 'v':
                    self.display_enabled = not self.display_enabled
                    self.get_logger().info(f"Display: {'ON' if self.display_enabled else 'OFF'}")
                elif key == 'd':
                    self.log_diagnostics()
                elif key == 'c' and self.mode == RobotMode.EMERGENCY and self.bin_full:
                    # Manual bin clearance
                    bin_dist = self.cli.read_bin_distance()
                    if bin_dist > BIN_FULL_DISTANCE_CM:
                        self.bin_full = False
                        self.cli.control_buzzer(False)
                        self.buzzer_state = False
                        self.autonomous_mode = True
                        self.transition_to(RobotMode.SEARCH, "Bin cleared")
                        self.get_logger().info(f"Bin cleared: {bin_dist:.1f}cm")
                elif not self.autonomous_mode:
                    if key == 'w':
                        self.send_cmd("F", APPROACH_FORWARD_SPEED)
                    elif key == 's':
                        self.send_cmd("B", APPROACH_FORWARD_SPEED)
                    elif key == 'a':
                        self.send_cmd("L", CENTERING_TURN_SPEED)
                    elif key == 'd':
                        self.send_cmd("R", CENTERING_TURN_SPEED)
                    elif key == 'x':
                        self.send_cmd("S")
    
    def send_cmd(self, cmd: str, speed: int = None, expect_reply=False, timeout=2.0):
        """Send command to ESP32"""
        if cmd in ["F", "B", "L", "R"] and not self.robot_movement_allowed:
            return False, None
            
        now = time.time() * 1000
        if now - self.last_cmd_time < self.cmd_cooldown_ms and not expect_reply:
            return False, None
        
        self.last_cmd_time = now
        line = cmd if speed is None else f"{cmd} {int(speed)}"
        success, reply = self.cli.send((line + "\n").encode("ascii"), expect_reply, timeout)
        
        if success:
            self.watchdog.reset()
            
        return success, reply
    
    def update_bin_distance(self):
        """Update bin distance - CONSTANT monitoring"""
        now = time.time()
        if now - self.last_bin_check >= self.bin_check_interval:
            distance = self.cli.read_bin_distance()
            self.last_bin_check = now
            self.bin_full = (distance <= BIN_FULL_DISTANCE_CM)
            
            # Update buzzer based on bin state
            if self.bin_full:
                # Rapid beeping when bin full
                if not self.buzzer_state or (now - self.last_buzzer_toggle >= self.buzzer_interval):
                    self.buzzer_state = not self.buzzer_state
                    self.cli.control_buzzer(self.buzzer_state)
                    self.last_buzzer_toggle = now
            else:
                # Turn off buzzer if not full
                if self.buzzer_state:
                    self.buzzer_state = False
                    self.cli.control_buzzer(False)
            
            return distance
        return self.cli.get_ultrasonic_data().bin_distance
    
    def start_collection_sequence(self):
        """Start final collection sequence"""
        self.collection_start_time = time.time()
        self.collection_phase = 1
        self.gripper_closed = False
        self.object_lifted = False
        self.object_dropped = False
        self.get_logger().info("Collection sequence started")
    
    def update_collection_sequence(self):
        """Execute collection sequence"""
        if self.collection_phase == 0:
            return False
        
        elapsed = time.time() - self.collection_start_time
        us_data = self.cli.get_ultrasonic_data()
        gripper_distance = us_data.gripper_distance
        
        if self.collection_phase == 1:  # Wait for gripper to be ready
            if not self.arm_in_collection_pose:
                self.cli.servo_cmd(SERVO_BASE, REACH_BASE, SERVO_FAST)
                self.cli.servo_cmd(SERVO_SHOULDER, REACH_SHOULDER, SERVO_FAST)
                self.arm_in_collection_pose = True
            
            # Read gripper ultrasonic for final approach
            if elapsed > 1.0:  # Wait for arm to be ready
                gripper_distance = self.cli.read_gripper_distance()
            
            if gripper_distance <= GRIPPER_GRAB_DISTANCE_CM and not self.gripper_closed:
                self.get_logger().info(f"Closing gripper at {gripper_distance:.1f}cm")
                self.cli.servo_cmd(SERVO_GRIP, GRIP_CLOSED, SERVO_MED)
                self.gripper_closed = True
                time.sleep(0.5)
                self.collection_phase = 2
                self.collection_start_time = time.time()
            elif elapsed > 8.0 and not self.gripper_closed:
                self.get_logger().warn("Gripper timeout - aborting")
                self.collection_phase = 8
        
        elif self.collection_phase == 2:  # Lift arm
            if elapsed < 2.0:
                progress = elapsed / 2.0
                self.cli.servo_cmd(SERVO_BASE, LIFT_BASE, SERVO_MED)
                shoulder_angle = REACH_SHOULDER + int((LIFT_SHOULDER - REACH_SHOULDER) * progress)
                self.cli.servo_cmd(SERVO_SHOULDER, shoulder_angle, SERVO_MED)
            else:
                self.object_lifted = True
                self.collection_phase = 3
        
        elif self.collection_phase == 3:  # Move to drop position
            self.cli.servo_cmd(SERVO_BASE, DROP_BASE, SERVO_SLOW)
            self.cli.servo_cmd(SERVO_SHOULDER, DROP_SHOULDER, SERVO_SLOW)
            if elapsed > 4.0:
                self.collection_phase = 4
        
        elif self.collection_phase == 4:  # Release gripper
            self.cli.servo_cmd(SERVO_GRIP, GRIP_OPEN, SERVO_MED)
            time.sleep(0.5)
            
            if gripper_distance > GRIPPER_RELEASE_DISTANCE_CM:
                self.object_dropped = True
                self.collection_phase = 5
            else:
                self.get_logger().info(f"Waiting for object to clear: {gripper_distance:.1f}cm")
        
        elif self.collection_phase == 5:  # Return to search
            self.cli.servo_cmd(SERVO_SHOULDER, SEARCH_SHOULDER, SERVO_SLOW)
            time.sleep(0.5)
            self.cli.servo_cmd(SERVO_BASE, SEARCH_BASE, SERVO_SLOW)
            self.cli.servo_cmd(SERVO_WRIST, SEARCH_WRIST, SERVO_SLOW)
            
            if elapsed > 6.0:
                self.collection_phase = 0
                self.gripper_closed = False
                self.object_lifted = False
                self.object_dropped = False
                self.arm_in_collection_pose = False
                self.robot_movement_allowed = True
                self.transition_to(RobotMode.SEARCH, "Collection complete")
                return True
        
        elif self.collection_phase == 8:  # Abort
            self.get_logger().info("Aborting collection")
            self.cli.servo_cmd(SERVO_SHOULDER, TIMEOUT_SHOULDER, SERVO_MED)
            self.cli.servo_cmd(SERVO_BASE, TIMEOUT_BASE, SERVO_MED)
            self.cli.servo_cmd(SERVO_WRIST, TIMEOUT_WRIST, SERVO_MED)
            time.sleep(1.5)
            
            if self.gripper_closed:
                self.cli.servo_cmd(SERVO_GRIP, GRIP_OPEN, SERVO_MED)
                time.sleep(0.5)
            
            self.cli.servo_cmd(SERVO_SHOULDER, SEARCH_SHOULDER, SERVO_SLOW)
            time.sleep(0.5)
            self.cli.servo_cmd(SERVO_BASE, SEARCH_BASE, SERVO_SLOW)
            
            self.collection_phase = 0
            self.gripper_closed = False
            self.object_lifted = False
            self.object_dropped = False
            self.arm_in_collection_pose = False
            self.robot_movement_allowed = True
            self.transition_to(RobotMode.SEARCH, "Collection aborted")
            return True
        
        return False
    
    def on_scan(self, msg: LaserScan):
        """LiDAR scan callback"""
        ranges = list(msg.ranges)
        if not ranges or msg.angle_increment == 0.0:
            return
        
        angle_min = msg.angle_min
        inc = msg.angle_increment
        pairs = []
        
        for i, r in enumerate(ranges):
            if r <= 0.0 or math.isinf(r) or math.isnan(r):
                continue
            
            ang = angle_min + i * inc
            deg = math.degrees(ang)
            if deg > 180.0:
                deg -= 360.0
            
            pairs.append((deg, r))
        
        self.last_pairs = pairs
    
    def sector_min(self, center_deg, half_width):
        """Get minimum distance in a sector"""
        if not self.last_pairs:
            return float("inf")
        
        lo = center_deg - half_width
        hi = center_deg + half_width
        vals = [d for (a, d) in self.last_pairs if lo <= a <= hi]
        return min(vals) if vals else float("inf")
    
    def vision_status(self):
        """Get current vision status"""
        with self.vision.lock:
            detected = self.vision.detected
            detection_result = self.vision.detection_result
            ts = detection_result.timestamp
            fw = self.vision.frame_width
            fh = self.vision.frame_height
        
        return detected, detection_result, ts, fw, fh
    
    def is_trash_centered(self, detection_result, fw):
        """Check if trash is centered in frame"""
        if detection_result.box is None:
            self.centered_frames = 0
            return False
        
        x, y, w, h = detection_result.box
        cx = x + w / 2
        frame_center = fw / 2
        dx = cx - frame_center
        
        centered = abs(dx) <= FRAME_CENTER_DEADZONE
        
        if centered:
            self.centered_frames += 1
            if self.centered_frames == 1:
                self.last_centered_time = time.time()
        else:
            self.centered_frames = 0
            
        return self.centered_frames >= STABLE_FRAMES_REQUIRED
    
    def center_trash_with_robot(self, detection_result, fw):
        """Center trash by rotating robot"""
        if detection_result.box is None:
            return None
        
        x, y, w, h = detection_result.box
        cx = x + w / 2
        frame_center = fw / 2
        dx = cx - frame_center
        
        if not self.robot_movement_allowed:
            return "centered"
            
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
            if self.last_centering_action in ["turning_left", "turning_right"]:
                self.send_cmd("S")
                self.last_centering_action = "centered"
            return "centered"
    
    def check_obstacle_timeout(self):
        """Check if obstacle avoidance is taking too long"""
        if self.obstacle_detected:
            if time.time() - self.obstacle_start_time > OBSTACLE_TIMEOUT:
                self.obstacle_detected = False
                self.transition_to(RobotMode.SEARCH, "Obstacle timeout")
                return True
        return False
    
    def log_diagnostics(self):
        """Log system diagnostics"""
        us_data = self.cli.get_ultrasonic_data()
        detected, detection_result, _, fw, _ = self.vision_status()
        
        front_min = self.sector_min(0.0, FRONT_SECTOR_ANGLE)
        
        self.get_logger().info("\n" + "="*60)
        self.get_logger().info("SYSTEM DIAGNOSTICS")
        self.get_logger().info("="*60)
        self.get_logger().info(f"Mode: {self.mode.name}")
        self.get_logger().info(f"Bin distance: {us_data.bin_distance:.1f}cm (Full: {self.bin_full})")
        self.get_logger().info(f"Trash distance: {us_data.trash_distance:.1f}cm")
        self.get_logger().info(f"Gripper distance: {us_data.gripper_distance:.1f}cm")
        self.get_logger().info(f"Vision detected: {detected}")
        if detected:
            self.get_logger().info(f"  Class: {detection_result.class_name}")
            self.get_logger().info(f"  Confidence: {detection_result.confidence:.2f}")
            self.get_logger().info(f"  Centered: {detection_result.centered}")
        self.get_logger().info(f"Front clearance: {front_min:.3f}m")
        self.get_logger().info(f"Collection phase: {self.collection_phase}")
        self.get_logger().info("="*60)
    
    def control_loop(self):
        """Main control loop"""
        now = time.time()
        
        # ========== HIGHEST PRIORITY: BIN FULL CHECK ==========
        # CONSTANT monitoring of bin ultrasonic
        bin_distance = self.update_bin_distance()
        
        # If bin is full, enter emergency mode immediately
        if self.bin_full:
            if self.mode != RobotMode.EMERGENCY:
                self.transition_to(RobotMode.EMERGENCY, f"Bin full! Distance: {bin_distance:.1f}cm")
                # Don't process any further commands
                return
        elif self.mode == RobotMode.EMERGENCY and not self.bin_full:
            # Bin was cleared, return to search
            self.cli.control_buzzer(False)
            self.buzzer_state = False
            self.autonomous_mode = True
            self.transition_to(RobotMode.SEARCH, "Bin cleared")
        
        # If in emergency mode (bin full), stop all processing
        if self.mode == RobotMode.EMERGENCY:
            # Continue rapid beeping
            if now - self.last_buzzer_toggle >= self.buzzer_interval:
                self.buzzer_state = not self.buzzer_state
                self.cli.control_buzzer(self.buzzer_state)
                self.last_buzzer_toggle = now
            return
        
        # ========== NORMAL OPERATION (Bin not full) ==========
        # Get vision data
        detected, detection_result, ts, fw, fh = self.vision_status()
        
        # Get LiDAR data for obstacle detection
        front_min = self.sector_min(0.0, FRONT_SECTOR_ANGLE)
        left_min = self.sector_min(90.0, SIDE_SECTOR_ANGLE)
        right_min = self.sector_min(-90.0, SIDE_SECTOR_ANGLE)
        
        # Emergency stop for close obstacles
        if front_min < HARD_STOP_M:
            self.send_cmd("S")
            self.get_logger().warn(f"Emergency stop! Obstacle at {front_min:.3f}m")
            if left_min > right_min:
                self.send_cmd("L", EMERGENCY_TURN_SPEED)
            else:
                self.send_cmd("R", EMERGENCY_TURN_SPEED)
            return
        
        # Check obstacle timeout
        if self.check_obstacle_timeout():
            return
        
        # Collection sequence has priority
        if self.collection_phase > 0:
            if self.collection_phase < 3:
                self.send_cmd("S")
            self.update_collection_sequence()
            return
        
        # Manual mode
        if not self.autonomous_mode:
            return
        
        # ========== STATE MACHINE ==========
        if self.mode == RobotMode.SEARCH:
            self.obstacle_detected = False
            self.centered_frames = 0
            
            # Obstacle check
            if front_min > MIN_CLEAR_M:
                if self.robot_movement_allowed:
                    self.send_cmd("F", SEARCH_FORWARD_SPEED)
            else:
                if not self.obstacle_detected:
                    self.obstacle_detected = True
                    self.obstacle_start_time = now
                    self.get_logger().info(f"Obstacle at {front_min:.3f}m")
                
                if left_min > right_min:
                    self.send_cmd("L", SEARCH_TURN_SPEED)
                else:
                    self.send_cmd("R", SEARCH_TURN_SPEED)
            
            # CAMERA DETECTION FIRST - only then use TRASH ultrasonic
            if detected and detection_result.confidence > 0.5:
                self.transition_to(RobotMode.CENTERING, f"Trash detected: {detection_result.class_name}")
        
        elif self.mode == RobotMode.CENTERING:
            if not detected or detection_result.box is None:
                if self.detection_lost_time == 0.0:
                    self.detection_lost_time = now
                
                if now - self.detection_lost_time > self.detection_timeout:
                    self.transition_to(RobotMode.SEARCH, "Detection lost")
                    return
                
                self.send_cmd("S")
                return
            else:
                self.detection_lost_time = 0.0
            
            # Obstacle check
            if front_min <= CLOSE_APPROACH_THRESHOLD:
                self.send_cmd("S")
                if left_min > right_min:
                    self.send_cmd("L", OBSTACLE_AVOID_SPEED)
                else:
                    self.send_cmd("R", OBSTACLE_AVOID_SPEED)
                return
            
            # Center the trash
            turn_status = self.center_trash_with_robot(detection_result, fw)
            
            # Move forward slowly while centering
            if self.robot_movement_allowed and front_min > MIN_CLEAR_M:
                self.send_cmd("F", 80)
            
            # When centered, start using TRASH ultrasonic
            if self.is_trash_centered(detection_result, fw):
                centered_duration = now - self.last_centered_time
                if centered_duration >= 1.0:
                    # NOW start using TRASH ultrasonic
                    self.transition_to(RobotMode.APPROACH, f"Trash centered: {detection_result.class_name}")
        
        elif self.mode == RobotMode.APPROACH:
            # In APPROACH mode, we use TRASH ultrasonic AFTER camera detection
            
            # Update TRASH ultrasonic reading
            if now - self.last_trash_check >= self.trash_check_interval:
                trash_distance = self.cli.read_trash_distance()
                self.last_trash_check = now
            else:
                trash_distance = self.cli.get_ultrasonic_data().trash_distance
            
            # Check if we still see the trash
            if not detected or detection_result.box is None:
                if self.detection_lost_time == 0.0:
                    self.detection_lost_time = now
                
                # If we lose sight but ultrasonic says we're close, continue
                if trash_distance < TRIGGER_COLLECTION_DISTANCE_CM:
                    self.get_logger().info(f"Lost sight but ultrasonic at {trash_distance:.1f}cm")
                    if self.robot_movement_allowed and front_min > MIN_CLEAR_M:
                        self.send_cmd("F", ULTRA_SLOW_APPROACH_SPEED)
                elif now - self.detection_lost_time > self.detection_timeout:
                    self.transition_to(RobotMode.SEARCH, "Approach timeout")
                    return
                else:
                    self.send_cmd("S")
                return
            else:
                self.detection_lost_time = 0.0
            
            # Obstacle check
            if front_min <= MIN_CLEAR_M:
                self.send_cmd("S")
                return
            
            # Center if needed
            self.center_trash_with_robot(detection_result, fw)
            
            # Use TRASH ultrasonic for approach speed
            if trash_distance > ULTRASONIC_SAFETY_CM and front_min > MIN_CLEAR_M and self.robot_movement_allowed:
                if trash_distance > 60.0:
                    speed = APPROACH_FORWARD_SPEED
                elif trash_distance > 40.0:
                    speed = APPROACH_FORWARD_SPEED - 20
                elif trash_distance > 25.0:
                    speed = SLOW_APPROACH_SPEED
                elif trash_distance > 15.0:
                    speed = SLOW_APPROACH_SPEED - 20
                elif trash_distance > 10.0:
                    speed = ULTRA_SLOW_APPROACH_SPEED
                else:
                    self.send_cmd("S")
                    return
                
                if abs(speed - self.last_motor_speed) > 10:
                    self.send_cmd("F", speed)
                    self.last_motor_speed = speed
                
                # TRASH ultrasonic triggers collection approach
                if trash_distance <= TRIGGER_COLLECTION_DISTANCE_CM:
                    self.get_logger().info(f"Trash at {trash_distance:.1f}cm - Starting collection")
                    self.send_cmd("S")
                    self.transition_to(RobotMode.COLLECT_APPROACH, f"Trash at {trash_distance:.1f}cm")
            else:
                self.send_cmd("S")
            
            # Approach timeout
            if now - self.approach_start_time > 15.0:
                self.transition_to(RobotMode.SEARCH, "Approach timeout")
        
        elif self.mode == RobotMode.COLLECT_APPROACH:
            # In COLLECT_APPROACH, use GRIPPER ultrasonic for final approach
            
            # Update GRIPPER ultrasonic
            if now - self.last_gripper_check >= self.gripper_check_interval:
                gripper_distance = self.cli.read_gripper_distance()
                self.last_gripper_check = now
            else:
                gripper_distance = self.cli.get_ultrasonic_data().gripper_distance
            
            # Final approach using GRIPPER ultrasonic
            if gripper_distance > ULTRASONIC_SAFETY_CM and self.robot_movement_allowed:
                if gripper_distance > 20.0:
                    speed = SLOW_APPROACH_SPEED
                else:
                    speed = ULTRA_SLOW_APPROACH_SPEED
                
                self.send_cmd("F", speed)
                
                # GRIPPER ultrasonic triggers final collection
                if gripper_distance <= GRIPPER_GRAB_DISTANCE_CM:
                    self.get_logger().info(f"Object at {gripper_distance:.1f}cm - Starting collection")
                    self.send_cmd("S")
                    self.transition_to(RobotMode.COLLECT, f"Object at {gripper_distance:.1f}cm")
            else:
                self.send_cmd("S")
                self.transition_to(RobotMode.COLLECT, f"Object very close at {gripper_distance:.1f}cm")
            
            # Timeout
            if now - self.collection_start_time > 10.0:
                self.transition_to(RobotMode.SEARCH, "Collection approach timeout")
    
    def destroy_node(self):
        """Graceful shutdown"""
        self.get_logger().info("Shutting down...")
        
        # Stop everything
        self.send_cmd("S")
        self.cli.control_buzzer(False)
        
        # Stop watchdog
        self.watchdog.stop()
        
        # Move arm to safe position
        self.cli.servo_cmd(SERVO_SHOULDER, SEARCH_SHOULDER, SERVO_FAST)
        self.cli.servo_cmd(SERVO_BASE, SEARCH_BASE, SERVO_FAST)
        
        # Stop vision thread
        self.stop_vision.set()
        if self.vision_thread.is_alive():
            self.vision_thread.join(timeout=2.0)
        
        # Close display window
        cv2.destroyAllWindows()
        
        # Close connection
        self.cli.close()
        
        # Destroy node
        super().destroy_node()
        self.get_logger().info("Shutdown complete")


def main():
    rclpy.init()
    node = IntegratedNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().error(f"Error: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()