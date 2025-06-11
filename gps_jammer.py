#!/usr/bin/env python3
import random
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, NavSatStatus


class GPSJammer(Node):
    def __init__(self) -> None:
        super().__init__('gps_jammer')

        self.pub = self.create_publisher(
            NavSatFix, '/mavros/global_position/raw/fix', 10
        )
        self.create_timer(0.1, self._jam)

    def _jam(self) -> None:
        sec = self.get_clock().now().nanoseconds // 1_000_000_000
        jam_now = (sec % 7) < 2

        if not jam_now:
            return

        msg = NavSatFix()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'jammed_gps'
        msg.status.status = NavSatStatus.STATUS_FIX
        msg.status.service = NavSatStatus.SERVICE_GPS

        msg.latitude = random.uniform(-90.0, 90.0)
        msg.longitude = random.uniform(-180.0, 180.0)
        msg.altitude = 0.0
        msg.position_covariance = [999.0] * 9
        msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_APPROXIMATED

        self.pub.publish(msg)


def main() -> None:
    rclpy.init()
    rclpy.spin(GPSJammer())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
