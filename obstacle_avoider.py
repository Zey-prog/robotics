# File: obstacle_avoider.py
# ROS 2 + TCP client to ESP32-S3 L298N server

import socket
import time
import math
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy  # ✅ added for QoS fix


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class ESP32Client:
    def __init__(self, host: str, port: int = 5050, reconnect_sec: float = 2.0):
        self.host = host
        self.port = port
        self.reconnect_sec = reconnect_sec
        self.sock = None
        self.lock = threading.Lock()

    def connect(self):
        with self.lock:
            self._close_unlocked()
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2.0)
            s.connect((self.host, self.port))
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.sock = s

    def _close_unlocked(self):
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
        self.sock = None

    def send_cmd(self, cmd: str, speed: int = None):
        line = cmd if speed is None else f"{cmd} {int(speed)}"
        data = (line + "\n").encode("ascii", errors="ignore")
        with self.lock:
            if not self.sock:
                try:
                    self.connect()
                except:
                    return False
            try:
                self.sock.sendall(data)
                return True
            except:
                self._close_unlocked()
                return False

    def close(self):
        with self.lock:
            self._close_unlocked()


class AvoidNode(Node):
    def __init__(self):
        super().__init__("lidar_obstacle_avoider")

        # ====== Parameters you can tweak ======
        self.declare_parameter("esp32_ip", "192.168.8.103")  # ✅ Set your ESP32-S3 IP
        self.declare_parameter("esp32_port", 5050)
        self.declare_parameter("forward_speed", 255)         # 0..255
        self.declare_parameter("turn_speed", 200)            # 0..255
        self.declare_parameter("min_clear_m", 0.45)          # stop/turn if closer than this
        self.declare_parameter("front_deg", 60.0)            # half-angle; total front cone = 2*front_deg
        self.declare_parameter("side_deg", 60.0)             # sectors for left/right comparison
        self.declare_parameter("cooldown_ms", 150)           # min ms between TCP sends
        self.declare_parameter("turn_time_ms", 350)          # burst turn time when blocked
        self.declare_parameter("max_forward_ms", 120)        # burst forward time step

        # Load parameters
        self.esp32_ip = self.get_parameter("esp32_ip").get_parameter_value().string_value
        self.esp32_port = int(self.get_parameter("esp32_port").get_parameter_value().integer_value)
        self.forward_speed = int(self.get_parameter("forward_speed").get_parameter_value().integer_value)
        self.turn_speed = int(self.get_parameter("turn_speed").get_parameter_value().integer_value)
        self.min_clear_m = float(self.get_parameter("min_clear_m").get_parameter_value().double_value)
        self.front_deg = float(self.get_parameter("front_deg").get_parameter_value().double_value)
        self.side_deg = float(self.get_parameter("side_deg").get_parameter_value().double_value)
        self.cooldown_ms = int(self.get_parameter("cooldown_ms").get_parameter_value().integer_value)
        self.turn_time_ms = int(self.get_parameter("turn_time_ms").get_parameter_value().integer_value)
        self.max_forward_ms = int(self.get_parameter("max_forward_ms").get_parameter_value().integer_value)

        # Connect to ESP32
        self.cli = ESP32Client(self.esp32_ip, self.esp32_port)
        try:
            self.cli.connect()
            self.get_logger().info(f"Connected to ESP32 at {self.esp32_ip}:{self.esp32_port}")
        except Exception as e:
            self.get_logger().warn(f"ESP32 connect failed initially: {e}")

        # ✅ Use Best Effort QoS to match the LIDAR publisher
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribe to /scan topic using compatible QoS
        self.sub = self.create_subscription(LaserScan, "/scan", self.on_scan, qos_profile)
        self.get_logger().info("Avoider ready. Subscribed to /scan with Best Effort QoS")

        self.last_send_ms = 0

    def on_scan(self, msg: LaserScan):
        ranges = list(msg.ranges)
        angle_min = msg.angle_min
        angle_inc = msg.angle_increment
        n = len(ranges)
        if n == 0 or angle_inc == 0.0:
            return

        pairs = []
        for i, r in enumerate(ranges):
            ang = angle_min + i * angle_inc
            ang_deg = math.degrees(ang)
            if ang_deg > 180.0:
                ang_deg -= 360.0
            if r <= 0.0 or math.isinf(r) or math.isnan(r):
                continue
            pairs.append((ang_deg, r))

        if not pairs:
            return

        def sector_min(center_deg, half_width_deg):
            lo = center_deg - half_width_deg
            hi = center_deg + half_width_deg
            vals = [d for (a, d) in pairs if lo <= a <= hi]
            if not vals:
                return float("inf")
            return min(vals)

        front_min = sector_min(0.0, self.front_deg)
        left_min = sector_min(+90.0, self.side_deg)
        right_min = sector_min(-90.0, self.side_deg)

        now_ms = int(self.get_clock().now().nanoseconds / 1e6)

        def send_throttled(cmd, spd=None):
            nonlocal now_ms
            if now_ms - self.last_send_ms >= self.cooldown_ms:
                ok = self.cli.send_cmd(cmd, spd)
                self.last_send_ms = now_ms
                if not ok:
                    try:
                        self.cli.connect()
                        self.cli.send_cmd(cmd, spd)
                    except:
                        pass

        if front_min > self.min_clear_m:
            send_throttled("F", clamp(self.forward_speed, 0, 255))
            return

        if left_min > right_min:
            send_throttled("L", clamp(self.turn_speed, 0, 255))
        else:
            send_throttled("R", clamp(self.turn_speed, 0, 255))


def main():
    rclpy.init()
    node = AvoidNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cli.send_cmd("S")
        node.cli.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

