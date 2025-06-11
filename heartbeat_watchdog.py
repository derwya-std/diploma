#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty
from mavros_msgs.msg import State
from mavros_msgs.srv import SetMode


class Watchdog(Node):

    def __init__(self) -> None:
        super().__init__('heartbeat_watchdog')

        self.declare_parameter('timeout', 1.0)
        self.declare_parameter('default_mode', 'AUTO.LOITER')

        self.timeout = float(self.get_parameter('timeout').value)
        self.default_mode = self.get_parameter('default_mode').value

        self.last_heartbeat = self.get_clock().now()
        self.current_mode = None
        self.in_rtl = False

        self.create_subscription(Empty, '/operator/heartbeat',
                                 self._heartbeat_cb, 10)
        self.create_subscription(State, '/mavros/state',
                                 self._state_cb, 10)

        self.cli = self.create_client(SetMode, '/mavros/set_mode')

        self.create_timer(0.2, self._check)
        self.get_logger().info(
            f'Watchdog started (timeout {self.timeout}s, default {self.default_mode})'
        )

    def _heartbeat_cb(self, _: Empty) -> None:
        self.last_heartbeat = self.get_clock().now()

        if self.in_rtl and self.current_mode != self.default_mode:
            self.get_logger().info(f'Heartbeat restored -> {self.default_mode}')
            self._set_mode(self.default_mode)
        self.in_rtl = False

    def _state_cb(self, msg: State) -> None:
        self.current_mode = msg.mode

    def _check(self) -> None:
        elapsed = (self.get_clock().now() - self.last_heartbeat).nanoseconds * 1e-9
        if elapsed > self.timeout and not self.in_rtl:
            self.get_logger().warn(f'No heartbeat for {elapsed:.2f}s -> AUTO.RTL')
            self._set_mode('AUTO.RTL')
            self.in_rtl = True

    def _set_mode(self, mode: str) -> None:
        if not self.cli.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn('set_mode service not ready yet')
            return

        req = SetMode.Request()
        req.base_mode = 0
        req.custom_mode = mode

        self.cli.call_async(req)


def main() -> None:
    rclpy.init()
    try:
        rclpy.spin(Watchdog())
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
