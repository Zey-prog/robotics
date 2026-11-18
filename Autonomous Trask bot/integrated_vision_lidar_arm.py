#!/usr/bin/env python3
"""
Integrated Robot Controller with ESP32 Streaming Ultrasonic
----------------------------------------------------------
COMPLETE SINGLE FILE VERSION - Everything in one place!
"""

import socket
import time
import threading
import sys
import select
import signal
import math
from enum import Enum, auto
from typing import Optional, Tuple
from collections import deque
from dataclasses import dataclass, field
import queue
import cv2
import numpy as np
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan


# ================ CONFIGURATION ================

# Network
ESP32_MOTOR_IP = "192.168.1.101"
ESP32_MOTOR_PORT = 5050
ESP32_CAM_STREAM = "http://192.168.1.100:81/stream"

# YOLO
YOLO_MODEL_PATH = "best.pt"

# ================ NAVIGATION PARAMETERS ================
MIN_CLEAR_M = 0.55          # Minimum clearance to move forward
HARD_STOP_M = 0.45          # Emergency stop distance
FRONT_SECTOR_ANGLE = 45.0   # Degrees to check in front
SIDE_SECTOR_ANGLE = 40.0    # Degrees to check on sides
OBSTACLE_TIMEOUT = 3.0      # Seconds to try avoiding obstacle

# ================ ULTRASONIC PARAMETERS ================
BIN_FULL_DISTANCE_CM = 10.0               # Emergency stop if ≤ this
TRIGGER_COLLECTION_DISTANCE_CM = 37.0     # TRASH sensor triggers collection
GRIPPER_GRAB_DISTANCE_CM = 7.0           # GRIPPER sensor closes gripper
GRIPPER_RELEASE_DISTANCE_CM = 4.0
ULTRASONIC_SAFETY_CM = 1.0

# ================ VISION PARAMETERS ================
FRAME_CENTER_DEADZONE = 30
STABLE_FRAMES_REQUIRED = 3

# ================ MOTOR SPEEDS (0-255) ================
SEARCH_FORWARD_SPEED = 130
SEARCH_TURN_SPEED = 190
APPROACH_FORWARD_SPEED = 100
SLOW_APPROACH_SPEED = 90
ULTRA_SLOW_APPROACH_SPEED = 60
CENTERING_TURN_SPEED = 170
OBSTACLE_AVOID_SPEED = 190
EMERGENCY_TURN_SPEED = 180

# ================ SERVO CONFIGURATION ================
SERVO_BASE = 0
SERVO_SHOULDER = 1
SERVO_GRIP = 2
SERVO_WRIST = 3

# Servo angles
SEARCH_BASE = 140
SEARCH_SHOULDER = 180
SEARCH_WRIST = 0
GRIP_OPEN = 40
GRIP_CLOSED = 110

# Collection poses
PREPARE_BASE = 110
PREPARE_SHOULDER = 120
REACH_BASE = 60
REACH_SHOULDER = 130
LIFT_BASE = 35
LIFT_SHOULDER = 10
DROP_BASE = 80
DROP_SHOULDER = 1

# Servo speeds
SERVO_SLOW = 30
SERVO_MED = 30
SERVO_FAST = 30

# ================ YOLO CLASS LABELS ================
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
    9: "Garbage bag"
}


# ================ ULTRASONIC DATA CLASS ================

@dataclass
class UltrasonicData:
    """Container for ultrasonic sensor data with timestamp"""
    gripper_distance: float = 999.0
    trash_distance: float = 999.0
    bin_distance: float = 999.0
    bin_full_threshold: float = 10.0
    timestamp: float = 0.0
    
    @property
    def bin_full(self) -> bool:
        return self.bin_distance <= self.bin_full_threshold
    
    def __str__(self) -> str:
        return f"G:{self.gripper_distance:.1f}cm T:{self.trash_distance:.1f}cm B:{self.bin_distance:.1f}cm"


@dataclass
class SensorHistory:
    """Circular buffer for sensor data history"""
    gripper: deque = field(default_factory=lambda: deque(maxlen=100))
    trash: deque = field(default_factory=lambda: deque(maxlen=100))
    bin: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def add(self, timestamp: float, gripper: float, trash: float, bin_dist: float):
        self.gripper.append((timestamp, gripper))
        self.trash.append((timestamp, trash))
        self.bin.append((timestamp, bin_dist))


# ================ ESP32 CLIENT ================

class ESP32Client:
    """ESP32 TCP Client with continuous ultrasonic streaming"""
    
    def __init__(self, host: str, port: int = 5050, timeout: float = 5.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        
        self.sock: Optional[socket.socket] = None
        self.lock = threading.RLock()
        self.connected = False
        
        self.ultrasonic_data = UltrasonicData()
        self.sensor_history = SensorHistory()
        
        self.streaming_enabled = False
        self.stream_thread: Optional[threading.Thread] = None
        self.stop_streaming = threading.Event()
        
        self.response_queue = queue.Queue(maxsize=10)
        self.last_data_time = 0.0
        self.data_timeout = 1.0
        
    def connect(self) -> bool:
        """Establish connection to ESP32 and start streaming"""
        with self.lock:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(self.timeout)
                self.sock.connect((self.host, self.port))
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                
                self.connected = True
                print(f"[ESP32] Connected to {self.host}:{self.port}")
                
                try:
                    banner = self.sock.recv(1024).decode(errors="ignore")
                    print(f"[ESP32] {banner.splitlines()[0]}")
                except:
                    pass
                
                self.enable_streaming()
                return True
                
            except Exception as e:
                print(f"[ESP32] Connection failed: {e}")
                self.connected = False
                self.sock = None
                return False
    
    def enable_streaming(self) -> bool:
        """Enable continuous ultrasonic data streaming"""
        if self.streaming_enabled:
            return True
            
        success, _ = self._send_command("STREAM_ON", expect_reply=True)
        if success:
            self.streaming_enabled = True
            self.stop_streaming.clear()
            self.stream_thread = threading.Thread(
                target=self._stream_reader, 
                daemon=True,
                name="ESP32Stream"
            )
            self.stream_thread.start()
            print("[ESP32] Streaming enabled")
            return True
        return False
    
    def disable_streaming(self) -> bool:
        """Disable continuous ultrasonic data streaming"""
        self.streaming_enabled = False
        self.stop_streaming.set()
        success, _ = self._send_command("STREAM_OFF", expect_reply=True)
        print("[ESP32] Streaming disabled")
        return success
    
    def _stream_reader(self):
        """Background thread that continuously reads the ultrasonic stream"""
        buffer = ""
        
        while not self.stop_streaming.is_set() and self.connected:
            try:
                if not self.sock:
                    break
                    
                data = self.sock.recv(1024).decode(errors="ignore")
                if not data:
                    print("[ESP32] Connection lost")
                    self.connected = False
                    break
                
                buffer += data
                self.last_data_time = time.time()
                
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    
                    if not line:
                        continue
                    
                    if line.startswith("U:"):
                        self._parse_ultrasonic_stream(line)
                    
                    elif line.startswith(("GRIPPER:", "TRASH:", "BIN:", 
                                         "STREAM_ON:", "STREAM_OFF:", 
                                         "PONG", "STATUS:")):
                        try:
                            self.response_queue.put_nowait(line)
                        except queue.Full:
                            pass
                    
            except socket.timeout:
                continue
            except Exception as e:
                if not self.stop_streaming.is_set():
                    print(f"[ESP32] Stream error: {e}")
                break
        
        self.streaming_enabled = False
        print("[ESP32] Stream reader stopped")
    
    def _parse_ultrasonic_stream(self, line: str):
        """Parse 'U:gripper,trash,bin' format"""
        try:
            values = line[2:].split(",")
            if len(values) == 3:
                gripper = float(values[0])
                trash = float(values[1])
                bin_dist = float(values[2])
                
                now = time.time()
                self.ultrasonic_data.gripper_distance = gripper
                self.ultrasonic_data.trash_distance = trash
                self.ultrasonic_data.bin_distance = bin_dist
                self.ultrasonic_data.timestamp = now
                
                self.sensor_history.add(now, gripper, trash, bin_dist)
                
        except (ValueError, IndexError) as e:
            print(f"[ESP32] Parse error: {e} - Line: {line}")
    
    def _send_command(self, command: str, expect_reply: bool = False, 
                     timeout: float = 2.0) -> Tuple[bool, Optional[str]]:
        """Send raw command to ESP32"""
        with self.lock:
            if not self.connected or not self.sock:
                if not self.connect():
                    return False, None
            
            try:
                cmd_bytes = (command + "\n").encode("ascii")
                self.sock.sendall(cmd_bytes)
                
                if expect_reply:
                    try:
                        reply = self.response_queue.get(timeout=timeout)
                        return True, reply
                    except queue.Empty:
                        return False, None
                
                return True, None
                
            except Exception as e:
                print(f"[ESP32] Send failed: {e}")
                self.connected = False
                return False, None
    
    # ================ ULTRASONIC GETTERS (NON-BLOCKING) ================
    
    def get_ultrasonic_data(self) -> UltrasonicData:
        return self.ultrasonic_data
    
    def get_gripper_distance(self) -> float:
        return self.ultrasonic_data.gripper_distance
    
    def get_trash_distance(self) -> float:
        return self.ultrasonic_data.trash_distance
    
    def get_bin_distance(self) -> float:
        return self.ultrasonic_data.bin_distance
    
    def is_bin_full(self) -> bool:
        return self.ultrasonic_data.bin_full
    
    # ================ MOTOR COMMANDS ================
    
    def motor_cmd(self, cmd: str, speed: Optional[int] = None) -> bool:
        line = cmd if speed is None else f"{cmd} {int(speed)}"
        success, _ = self._send_command(line)
        return success
    
    # ================ SERVO COMMANDS ================
    
    def servo_cmd(self, servo_id: int, angle: int, speed: int = 30) -> bool:
        angle = max(0, min(180, angle))
        speed = max(5, min(100, speed))
        cmd = f"SV {servo_id} {angle} {speed}"
        success, _ = self._send_command(cmd)
        return success
    
    # ================ BUZZER COMMANDS ================
    
    def buzzer_on(self) -> bool:
        success, _ = self._send_command("BUZZER ON")
        return success
    
    def buzzer_off(self) -> bool:
        success, _ = self._send_command("BUZZER OFF")
        return success
    
    def buzzer_beep(self, duration_ms: int = 100):
        self.buzzer_on()
        time.sleep(duration_ms / 1000.0)
        self.buzzer_off()
    
    # ================ SYSTEM COMMANDS ================
    
    def ping(self) -> bool:
        success, reply = self._send_command("PING", expect_reply=True, timeout=1.0)
        return success and reply == "PONG"
    
    def close(self):
        self.disable_streaming()
        with self.lock:
            if self.sock:
                try:
                    self.sock.close()
                except:
                    pass
                self.sock = None
            self.connected = False
        print("[ESP32] Disconnected")


# ================ ROBOT STATE CLASSES ================

class RobotMode(Enum):
    """Robot operation modes"""
    SEARCH = auto()
    CENTERING = auto()
    APPROACH = auto()
    COLLECT_APPROACH = auto()
    COLLECT = auto()
    EMERGENCY = auto()
    MANUAL = auto()


class DetectionResult:
    """Container for detection results"""
    def __init__(self):
        self.box = None
        self.confidence = 0.0
        self.class_id = -1
        self.class_name = ""
        self.timestamp = 0.0
        self.centered = False
        self.distance_to_center = 0


class SharedVisionState:
    """Thread-safe container for vision results"""
    def __init__(self):
        self.lock = threading.Lock()
        self.detected = False
        self.detection_result = DetectionResult()
        self.frame_width = 480
        self.frame_height = 320
        self.display_frame = None
    
    def update(self, detection_result, fw, fh, display_frame=None):
        with self.lock:
            self.detected = detection_result.box is not None
            self.detection_result = detection_result
            self.detection_result.timestamp = time.time()
            self.frame_width = fw
            self.frame_height = fh
            if display_frame is not None:
                self.display_frame = display_frame.copy()
    
    def get_display_frame(self):
        with self.lock:
            return self.display_frame.copy() if self.display_frame is not None else None


# ================ MAIN ROBOT NODE ================

class IntegratedRobotNode(Node):
    """Main robot control node with continuous ultrasonic streaming"""
    
    def __init__(self):
        super().__init__("integrated_robot_controller")
        
        # ----- Signal Handling -----
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # ----- ESP32 Client with Continuous Ultrasonic Streaming -----
        self.esp32 = ESP32Client(ESP32_MOTOR_IP, ESP32_MOTOR_PORT)
        
        if not self.esp32.connect():
            self.get_logger().error("Failed to connect to ESP32")
        else:
            self.get_logger().info("ESP32 connected - Ultrasonic streaming active")
        
        # ----- LiDAR Subscription -----
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.lidar_sub = self.create_subscription(
            LaserScan, "/scan", self._lidar_callback, qos
        )
        self.lidar_data = None
        
        # ----- Vision System -----
        self.vision_state = SharedVisionState()
        self.stop_vision = threading.Event()
        self.vision_thread = threading.Thread(
            target=self._vision_worker,
            args=(self.vision_state, self.stop_vision),
            daemon=True
        )
        self.vision_thread.start()
        
        # ----- Display Thread -----
        self.display_thread = threading.Thread(
            target=self._display_worker, 
            daemon=True
        )
        self.display_thread.start()
        
        # ----- State Machine -----
        self.mode = RobotMode.SEARCH
        self.autonomous_mode = True
        
        # Collection sequence state
        self.collection_phase = 0
        self.collection_start_time = 0.0
        self.gripper_closed = False
        self.arm_in_collection_pose = False
        
        # Vision tracking
        self.detection_lost_time = 0.0
        self.detection_timeout = 5.0
        self.centered_frames = 0
        self.last_centered_time = 0.0
        
        # Obstacle avoidance
        self.obstacle_detected = False
        self.obstacle_start_time = 0.0
        
        # Movement control
        self.robot_movement_allowed = True
        self.last_cmd_time = 0.0
        self.cmd_cooldown_ms = 80
        
        # Bin full state
        self.bin_full = False
        self.buzzer_state = False
        self.last_buzzer_toggle = 0.0
        self.buzzer_interval = 0.2
        
        # ----- Main Control Timer (10Hz) -----
        self.timer = self.create_timer(0.1, self._control_loop)
        
        # ----- Keyboard Thread -----
        self.keyboard_thread = threading.Thread(
            target=self._keyboard_listener, 
            daemon=True
        )
        self.keyboard_thread.start()
        
        # Initialize arm
        self._arm_to_search_pose()
        
        self.get_logger().info("=" * 50)
        self.get_logger().info("ROBOT CONTROLLER INITIALIZED")
        self.get_logger().info(f"Mode: {self.mode.name}")
        self.get_logger().info("Ultrasonic: CONTINUOUS STREAMING - Zero latency")
        self.get_logger().info("=" * 50)
    
    # ================ ULTRASONIC ACCESS (INSTANT, NON-BLOCKING) ================
    
    def get_ultrasonic(self) -> UltrasonicData:
        return self.esp32.get_ultrasonic_data()
    
    def get_bin_distance(self) -> float:
        return self.esp32.get_bin_distance()
    
    def get_trash_distance(self) -> float:
        return self.esp32.get_trash_distance()
    
    def get_gripper_distance(self) -> float:
        return self.esp32.get_gripper_distance()
    
    def is_bin_full(self) -> bool:
        return self.esp32.is_bin_full()
    
    # ================ ARM CONTROL ================
    
    def _arm_to_search_pose(self):
        self.esp32.servo_cmd(SERVO_SHOULDER, SEARCH_SHOULDER, SERVO_SLOW)
        self.esp32.servo_cmd(SERVO_BASE, SEARCH_BASE, SERVO_SLOW)
        time.sleep(0.5)
        self.esp32.servo_cmd(SERVO_WRIST, SEARCH_WRIST, SERVO_SLOW)
        self.esp32.servo_cmd(SERVO_GRIP, GRIP_OPEN, SERVO_SLOW)
    
    def _arm_to_collection_pose(self):
        self.esp32.servo_cmd(SERVO_BASE, PREPARE_BASE, SERVO_MED)
        self.esp32.servo_cmd(SERVO_SHOULDER, PREPARE_SHOULDER, SERVO_MED)
        time.sleep(1.0)
        self.esp32.servo_cmd(SERVO_BASE, REACH_BASE, SERVO_MED)
        self.esp32.servo_cmd(SERVO_SHOULDER, REACH_SHOULDER, SERVO_MED)
        time.sleep(1.0)
        self.esp32.servo_cmd(SERVO_GRIP, GRIP_OPEN, SERVO_MED)
    
    # ================ MOTOR CONTROL ================
    
    def _send_motor_cmd(self, cmd: str, speed: int = None):
        now = time.time() * 1000
        if now - self.last_cmd_time < self.cmd_cooldown_ms:
            return False
        
        if not self.robot_movement_allowed and cmd in ["F", "B", "L", "R"]:
            return False
        
        success = self.esp32.motor_cmd(cmd, speed)
        if success:
            self.last_cmd_time = now
        return success
    
    # ================ LIDAR PROCESSING ================
    
    def _lidar_callback(self, msg: LaserScan):
        self.lidar_data = msg
    
    def _get_lidar_sector_min(self, center_deg: float, half_width: float) -> float:
        if not self.lidar_data:
            return float("inf")
        
        msg = self.lidar_data
        ranges = list(msg.ranges)
        angle_min = msg.angle_min
        inc = msg.angle_increment
        
        lo = center_deg - half_width
        hi = center_deg + half_width
        distances = []
        
        for i, r in enumerate(ranges):
            if r <= 0.0 or math.isinf(r) or math.isnan(r):
                continue
            
            ang = math.degrees(angle_min + i * inc)
            if ang > 180.0:
                ang -= 360.0
            
            if lo <= ang <= hi:
                distances.append(r)
        
        return min(distances) if distances else float("inf")
    
    # ================ VISION PROCESSING ================
    
    def _vision_worker(self, shared: SharedVisionState, stop_event: threading.Event):
        try:
            model = YOLO(YOLO_MODEL_PATH)
            self.get_logger().info("YOLOv8 model loaded")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO: {e}")
            return
        
        while not stop_event.is_set():
            cap = cv2.VideoCapture(ESP32_CAM_STREAM)
            if not cap.isOpened():
                time.sleep(1.0)
                continue
            
            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                
                fh, fw = frame.shape[:2]
                display_frame = frame.copy()
                
                try:
                    small = cv2.resize(frame, (320, 320))
                    results = model(small, imgsz=320, conf=0.4, verbose=False)
                    
                    best_box = None
                    best_conf = 0.0
                    best_class_id = -1
                    best_class_name = ""
                    
                    if results[0].boxes is not None:
                        for box in results[0].boxes:
                            conf = float(box.conf[0])
                            
                            if conf < 0.55:
                                continue
                            
                            cls_id = int(box.cls[0])
                            class_name = TRASH_LABELS.get(cls_id, f"Class {cls_id}")
                            
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            w = x2 - x1
                            h = y2 - y1
                            
                            if w < 8 or h < 8:
                                continue
                            
                            if conf > best_conf:
                                best_conf = conf
                                best_box = (int(x1), int(y1), int(w), int(h))
                                best_class_id = cls_id
                                best_class_name = class_name
                    
                    detection_result = DetectionResult()
                    if best_box:
                        sx = fw / 320.0
                        sy = fh / 320.0
                        x, y, w, h = best_box
                        full_box = (
                            int(x * sx), 
                            int(y * sy), 
                            int(w * sx), 
                            int(h * sy)
                        )
                        
                        detection_result.box = full_box
                        detection_result.confidence = best_conf
                        detection_result.class_id = best_class_id
                        detection_result.class_name = best_class_name
                        
                        cx_obj = full_box[0] + full_box[2] // 2
                        frame_center = fw // 2
                        detection_result.distance_to_center = abs(cx_obj - frame_center)
                        detection_result.centered = detection_result.distance_to_center <= FRAME_CENTER_DEADZONE
                    
                    shared.update(detection_result, fw, fh, display_frame)
                    
                except Exception as e:
                    self.get_logger().error(f"Vision error: {e}")
                    shared.update(DetectionResult(), fw, fh, display_frame)
            
            cap.release()
        
        cv2.destroyAllWindows()
    
    def _display_worker(self):
        while rclpy.ok():
            frame = self.vision_state.get_display_frame()
            if frame is not None:
                display = frame.copy()
                us_data = self.get_ultrasonic()
                
                cv2.putText(display, f"Mode: {self.mode.name}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display, f"Bin: {us_data.bin_distance:.1f}cm", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(display, f"Trash: {us_data.trash_distance:.1f}cm", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                if us_data.bin_full:
                    cv2.putText(display, "BIN FULL!", 
                               (frame.shape[1]//2 - 50, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                
                cv2.imshow("Robot Vision", display)
                cv2.waitKey(1)
            
            time.sleep(0.033)
    
    # ================ STATE MACHINE ================
    
    def _transition_to(self, new_mode: RobotMode, reason: str = ""):
        old_mode = self.mode
        self.mode = new_mode
        
        if new_mode == RobotMode.SEARCH:
            self.centered_frames = 0
            self.obstacle_detected = False
            self.detection_lost_time = 0.0
            self.robot_movement_allowed = True
            self._send_motor_cmd("S")
            self._arm_to_search_pose()
            
        elif new_mode == RobotMode.CENTERING:
            self.centered_frames = 0
            self.obstacle_detected = False
            self.detection_lost_time = 0.0
            self.robot_movement_allowed = True
            
        elif new_mode == RobotMode.APPROACH:
            self.robot_movement_allowed = True
            
        elif new_mode == RobotMode.COLLECT_APPROACH:
            self.collection_start_time = time.time()
            self.robot_movement_allowed = False
            self._send_motor_cmd("S")
            self._arm_to_collection_pose()
            
        elif new_mode == RobotMode.COLLECT:
            self._start_collection_sequence()
            self.robot_movement_allowed = False
            
        elif new_mode == RobotMode.EMERGENCY:
            self._send_motor_cmd("S")
            self.autonomous_mode = False
            self.robot_movement_allowed = False
            self.esp32.buzzer_on()
            self.buzzer_state = True
            self.get_logger().error(f"EMERGENCY: {reason}")
            
        elif new_mode == RobotMode.MANUAL:
            self._send_motor_cmd("S")
            self.robot_movement_allowed = False
        
        if old_mode != new_mode:
            self.get_logger().info(f"Mode: {old_mode.name} → {new_mode.name} ({reason})")
    
    def _start_collection_sequence(self):
        self.collection_start_time = time.time()
        self.collection_phase = 1
        self.gripper_closed = False
        self.arm_in_collection_pose = False
        self.get_logger().info("Collection sequence started")
    
    def _update_collection_sequence(self) -> bool:
        if self.collection_phase == 0:
            return False
        
        elapsed = time.time() - self.collection_start_time
        gripper_distance = self.get_gripper_distance()
        
        if self.collection_phase == 1:
            if not self.arm_in_collection_pose:
                self.esp32.servo_cmd(SERVO_BASE, REACH_BASE, SERVO_FAST)
                self.esp32.servo_cmd(SERVO_SHOULDER, REACH_SHOULDER, SERVO_FAST)
                self.arm_in_collection_pose = True
            
            if gripper_distance <= GRIPPER_GRAB_DISTANCE_CM and not self.gripper_closed:
                self.get_logger().info(f"Closing gripper at {gripper_distance:.1f}cm")
                self.esp32.servo_cmd(SERVO_GRIP, GRIP_CLOSED, SERVO_MED)
                self.gripper_closed = True
                time.sleep(0.5)
                self.collection_phase = 2
                self.collection_start_time = time.time()
            elif elapsed > 8.0 and not self.gripper_closed:
                self.get_logger().warn("Gripper timeout - aborting")
                self.collection_phase = 8
        
        elif self.collection_phase == 2:
            if elapsed < 2.0:
                progress = elapsed / 2.0
                self.esp32.servo_cmd(SERVO_BASE, LIFT_BASE, SERVO_MED)
                shoulder_angle = REACH_SHOULDER + int((LIFT_SHOULDER - REACH_SHOULDER) * progress)
                self.esp32.servo_cmd(SERVO_SHOULDER, shoulder_angle, SERVO_MED)
            else:
                self.collection_phase = 3
                self.collection_start_time = time.time()
        
        elif self.collection_phase == 3:
            self.esp32.servo_cmd(SERVO_BASE, DROP_BASE, SERVO_SLOW)
            self.esp32.servo_cmd(SERVO_SHOULDER, DROP_SHOULDER, SERVO_SLOW)
            if elapsed > 4.0:
                self.collection_phase = 4
                self.collection_start_time = time.time()
        
        elif self.collection_phase == 4:
            self.esp32.servo_cmd(SERVO_GRIP, GRIP_OPEN, SERVO_MED)
            time.sleep(0.5)
            
            if gripper_distance > GRIPPER_RELEASE_DISTANCE_CM:
                self.collection_phase = 5
                self.collection_start_time = time.time()
        
        elif self.collection_phase == 5:
            self.esp32.servo_cmd(SERVO_SHOULDER, SEARCH_SHOULDER, SERVO_SLOW)
            time.sleep(0.5)
            self.esp32.servo_cmd(SERVO_BASE, SEARCH_BASE, SERVO_SLOW)
            self.esp32.servo_cmd(SERVO_WRIST, SEARCH_WRIST, SERVO_SLOW)
            
            if elapsed > 6.0:
                self.collection_phase = 0
                self.gripper_closed = False
                self.arm_in_collection_pose = False
                self.robot_movement_allowed = True
                self._transition_to(RobotMode.SEARCH, "Collection complete")
                return True
        
        elif self.collection_phase == 8:
            self.get_logger().info("Aborting collection")
            self.esp32.servo_cmd(SERVO_SHOULDER, SEARCH_SHOULDER, SERVO_MED)
            self.esp32.servo_cmd(SERVO_BASE, SEARCH_BASE, SERVO_MED)
            self.esp32.servo_cmd(SERVO_WRIST, SEARCH_WRIST, SERVO_MED)
            time.sleep(1.5)
            
            if self.gripper_closed:
                self.esp32.servo_cmd(SERVO_GRIP, GRIP_OPEN, SERVO_MED)
                time.sleep(0.5)
            
            self.esp32.servo_cmd(SERVO_SHOULDER, SEARCH_SHOULDER, SERVO_SLOW)
            self.esp32.servo_cmd(SERVO_BASE, SEARCH_BASE, SERVO_SLOW)
            
            self.collection_phase = 0
            self.gripper_closed = False
            self.arm_in_collection_pose = False
            self.robot_movement_allowed = True
            self._transition_to(RobotMode.SEARCH, "Collection aborted")
            return True
        
        return False
    
    # ================ CONTROL LOOP ================
    
    def _control_loop(self):
        now = time.time()
        
        # ----- HIGHEST PRIORITY: Bin Full Check (INSTANT) -----
        us_data = self.get_ultrasonic()
        bin_full = us_data.bin_full
        
        if bin_full:
            if self.mode != RobotMode.EMERGENCY:
                self._transition_to(RobotMode.EMERGENCY, 
                                   f"Bin full! Distance: {us_data.bin_distance:.1f}cm")
            else:
                if now - self.last_buzzer_toggle >= self.buzzer_interval:
                    self.buzzer_state = not self.buzzer_state
                    if self.buzzer_state:
                        self.esp32.buzzer_on()
                    else:
                        self.esp32.buzzer_off()
                    self.last_buzzer_toggle = now
            return
        
        if self.mode == RobotMode.EMERGENCY and not bin_full:
            self.esp32.buzzer_off()
            self.buzzer_state = False
            self.autonomous_mode = True
            self._transition_to(RobotMode.SEARCH, "Bin cleared")
        
        # ----- LiDAR Obstacle Detection -----
        front_min = self._get_lidar_sector_min(0.0, FRONT_SECTOR_ANGLE)
        left_min = self._get_lidar_sector_min(90.0, SIDE_SECTOR_ANGLE)
        right_min = self._get_lidar_sector_min(-90.0, SIDE_SECTOR_ANGLE)
        
        if front_min < HARD_STOP_M:
            self._send_motor_cmd("S")
            self.get_logger().warn(f"Emergency stop! Obstacle at {front_min:.3f}m")
            if left_min > right_min:
                self._send_motor_cmd("L", EMERGENCY_TURN_SPEED)
            else:
                self._send_motor_cmd("R", EMERGENCY_TURN_SPEED)
            return
        
        # ----- Collection sequence has priority -----
        if self.collection_phase > 0:
            if self.collection_phase < 3:
                self._send_motor_cmd("S")
            self._update_collection_sequence()
            return
        
        # ----- Manual Mode -----
        if not self.autonomous_mode:
            return
        
        # ----- Get Vision Data -----
        with self.vision_state.lock:
            detected = self.vision_state.detected
            detection_result = self.vision_state.detection_result
            fw = self.vision_state.frame_width
        
        # ========== STATE MACHINE ==========
        
        # ----- SEARCH MODE -----
        if self.mode == RobotMode.SEARCH:
            if front_min > MIN_CLEAR_M:
                if self.robot_movement_allowed:
                    self._send_motor_cmd("F", SEARCH_FORWARD_SPEED)
            else:
                if not self.obstacle_detected:
                    self.obstacle_detected = True
                    self.obstacle_start_time = now
                    self.get_logger().info(f"Obstacle at {front_min:.3f}m")
                
                if left_min > right_min:
                    self._send_motor_cmd("L", SEARCH_TURN_SPEED)
                else:
                    self._send_motor_cmd("R", SEARCH_TURN_SPEED)
            
            if detected and detection_result.confidence > 0.5:
                self._transition_to(RobotMode.CENTERING, 
                                   f"Trash detected: {detection_result.class_name}")
        
        # ----- CENTERING MODE -----
        elif self.mode == RobotMode.CENTERING:
            if not detected or detection_result.box is None:
                if self.detection_lost_time == 0.0:
                    self.detection_lost_time = now
                
                if now - self.detection_lost_time > self.detection_timeout:
                    self._transition_to(RobotMode.SEARCH, "Detection lost")
                    return
                
                self._send_motor_cmd("S")
                return
            else:
                self.detection_lost_time = 0.0
            
            if front_min <= MIN_CLEAR_M:
                self._send_motor_cmd("S")
                if left_min > right_min:
                    self._send_motor_cmd("L", OBSTACLE_AVOID_SPEED)
                else:
                    self._send_motor_cmd("R", OBSTACLE_AVOID_SPEED)
                return
            
            x, y, w, h = detection_result.box
            cx = x + w / 2
            frame_center = fw / 2
            dx = cx - frame_center
            
            if abs(dx) > FRAME_CENTER_DEADZONE:
                if dx < 0:
                    self._send_motor_cmd("L", CENTERING_TURN_SPEED)
                else:
                    self._send_motor_cmd("R", CENTERING_TURN_SPEED)
                self.centered_frames = 0
            else:
                self._send_motor_cmd("S")
                self.centered_frames += 1
                if self.centered_frames == 1:
                    self.last_centered_time = now
                
                if self.centered_frames >= STABLE_FRAMES_REQUIRED:
                    if now - self.last_centered_time >= 1.0:
                        self._transition_to(RobotMode.APPROACH, 
                                           f"Trash centered: {detection_result.class_name}")
        
        # ----- APPROACH MODE (Using TRASH ultrasonic) -----
        elif self.mode == RobotMode.APPROACH:
            trash_distance = self.get_trash_distance()
            
            if not detected or detection_result.box is None:
                if self.detection_lost_time == 0.0:
                    self.detection_lost_time = now
                
                if trash_distance < TRIGGER_COLLECTION_DISTANCE_CM:
                    if self.robot_movement_allowed and front_min > MIN_CLEAR_M:
                        self._send_motor_cmd("F", ULTRA_SLOW_APPROACH_SPEED)
                elif now - self.detection_lost_time > self.detection_timeout:
                    self._transition_to(RobotMode.SEARCH, "Approach timeout")
                    return
                else:
                    self._send_motor_cmd("S")
                return
            else:
                self.detection_lost_time = 0.0
            
            if front_min <= MIN_CLEAR_M:
                self._send_motor_cmd("S")
                return
            
            x, y, w, h = detection_result.box
            cx = x + w / 2
            frame_center = fw / 2
            dx = cx - frame_center
            
            if abs(dx) > FRAME_CENTER_DEADZONE:
                if dx < 0:
                    self._send_motor_cmd("L", CENTERING_TURN_SPEED)
                else:
                    self._send_motor_cmd("R", CENTERING_TURN_SPEED)
            
            if trash_distance > ULTRASONIC_SAFETY_CM and front_min > MIN_CLEAR_M:
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
                    self._send_motor_cmd("S")
                    return
                
                self._send_motor_cmd("F", speed)
                
                if trash_distance <= TRIGGER_COLLECTION_DISTANCE_CM:
                    self.get_logger().info(f"Trash at {trash_distance:.1f}cm - Starting collection")
                    self._send_motor_cmd("S")
                    self._transition_to(RobotMode.COLLECT_APPROACH, 
                                       f"Trash at {trash_distance:.1f}cm")
            else:
                self._send_motor_cmd("S")
        
        # ----- COLLECT APPROACH MODE (Using GRIPPER ultrasonic) -----
        elif self.mode == RobotMode.COLLECT_APPROACH:
            gripper_distance = self.get_gripper_distance()
            
            if gripper_distance > ULTRASONIC_SAFETY_CM and self.robot_movement_allowed:
                if gripper_distance > 20.0:
                    speed = SLOW_APPROACH_SPEED
                else:
                    speed = ULTRA_SLOW_APPROACH_SPEED
                
                self._send_motor_cmd("F", speed)
                
                if gripper_distance <= GRIPPER_GRAB_DISTANCE_CM:
                    self.get_logger().info(f"Object at {gripper_distance:.1f}cm - Collecting")
                    self._send_motor_cmd("S")
                    self._transition_to(RobotMode.COLLECT, 
                                       f"Object at {gripper_distance:.1f}cm")
            else:
                self._send_motor_cmd("S")
                self._transition_to(RobotMode.COLLECT, 
                                   f"Object very close at {gripper_distance:.1f}cm")
    
    # ================ KEYBOARD CONTROL ================
    
    def _keyboard_listener(self):
        while rclpy.ok():
            if select.select([sys.stdin], [], [], 0.1)[0]:
                key = sys.stdin.read(1).lower()
                
                if key == 'q':
                    self.get_logger().info("Quitting...")
                    rclpy.shutdown()
                
                elif key == 'e':
                    self.autonomous_mode = not self.autonomous_mode
                    if self.autonomous_mode:
                        self._transition_to(RobotMode.SEARCH, "Auto mode")
                    else:
                        self._transition_to(RobotMode.MANUAL, "Manual mode")
                    self.get_logger().info(f"Autonomous: {self.autonomous_mode}")
                
                elif key == 'c' and self.mode == RobotMode.EMERGENCY:
                    if not self.is_bin_full():
                        self.esp32.buzzer_off()
                        self.buzzer_state = False
                        self.autonomous_mode = True
                        self._transition_to(RobotMode.SEARCH, "Bin cleared")
                
                elif not self.autonomous_mode:
                    if key == 'w':
                        self._send_motor_cmd("F", APPROACH_FORWARD_SPEED)
                    elif key == 's':
                        self._send_motor_cmd("B", APPROACH_FORWARD_SPEED)
                    elif key == 'a':
                        self._send_motor_cmd("L", CENTERING_TURN_SPEED)
                    elif key == 'd':
                        self._send_motor_cmd("R", CENTERING_TURN_SPEED)
                    elif key == 'x':
                        self._send_motor_cmd("S")
                    elif key == ' ':
                        self._send_motor_cmd("S")
    
    # ================ SIGNAL HANDLING ================
    
    def _signal_handler(self, signum, frame):
        self.get_logger().info(f"Signal {signum} received, shutting down...")
        self.destroy_node()
    
    def destroy_node(self):
        self.get_logger().info("Shutting down...")
        
        self._send_motor_cmd("S")
        self.esp32.buzzer_off()
        
        self.esp32.servo_cmd(SERVO_SHOULDER, SEARCH_SHOULDER, SERVO_FAST)
        self.esp32.servo_cmd(SERVO_BASE, SEARCH_BASE, SERVO_FAST)
        
        self.stop_vision.set()
        if self.vision_thread.is_alive():
            self.vision_thread.join(timeout=2.0)
        
        self.esp32.close()
        cv2.destroyAllWindows()
        
        super().destroy_node()
        self.get_logger().info("Shutdown complete")


def main():
    rclpy.init()
    node = IntegratedRobotNode()
    
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