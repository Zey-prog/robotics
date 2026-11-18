#!/usr/bin/env python3
import serial
import time
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu


PORT = "/dev/ttyUSB0"
BAUD = 230400


class Gy91Publisher(Node):
    def __init__(self):
        super().__init__("gy91_publisher")

        self.pub = self.create_publisher(
            Imu,
            "/imu/data",
            qos_profile_sensor_data
        )

        self.ser = serial.Serial(PORT, BAUD, timeout=1)
        time.sleep(2)

        self.get_logger().info("GY-91 serial IMU publisher started")

        self.timer = self.create_timer(0.02, self.loop)

    def loop(self):
        line = self.ser.readline().decode(errors="ignore").strip()
        if not line or line.startswith("#"):
            return

        try:
            ax, ay, az, gx, gy, gz, temp = map(float, line.split(","))
        except:
            return

        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "imu_link"

        # Angular velocity (deg/s → rad/s)
        msg.angular_velocity.x = math.radians(gx)
        msg.angular_velocity.y = math.radians(gy)
        msg.angular_velocity.z = math.radians(gz)

        # Linear acceleration (g → m/s²)
        msg.linear_acceleration.x = ax * 9.81
        msg.linear_acceleration.y = ay * 9.81
        msg.linear_acceleration.z = az * 9.81

        # Orientation not provided yet (OK for yaw-rate EKF)

        self.pub.publish(msg)


def main():
    rclpy.init()
    node = Gy91Publisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

