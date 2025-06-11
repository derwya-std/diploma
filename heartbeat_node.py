#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty


class Heartbeat(Node):

    def __init__(self) -> None:
        super().__init__('operator_heartbeat')

        self.declare_parameter('topic', '/operator/heartbeat')
        self.declare_parameter('rate_hz', 10.0)

        topic = self.get_parameter('topic').value
        rate_hz = self.get_parameter('rate_hz').value
        period = 1.0 / max(rate_hz, 0.1)

        self._pub = self.create_publisher(Empty, topic, 10)
        self.create_timer(period, lambda: self._pub.publish(Empty()))
        self.get_logger().info(f'Publishing heartbeat on {topic} @ {rate_hz:.1f} Hz')


def main() -> None:
    rclpy.init()
    rclpy.spin(Heartbeat())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
