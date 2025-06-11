#!/usr/bin/env python3
import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix
from mavros_msgs.srv import CommandHome


class HomeRecorder(Node):

    def __init__(self) -> None:
        super().__init__('home_recorder')

        self.declare_parameter('odom_topic', '/mavros/local_position/odom')
        self.declare_parameter('gps_topic', '/mavros/global_position/raw/fix')
        self.declare_parameter('service_name', '/mavros/cmd/set_home')
        self.declare_parameter('output_file', 'home_pose.yaml')
        self.declare_parameter('use_gps', False)

        odom_topic = self.get_parameter('odom_topic').value
        gps_topic = self.get_parameter('gps_topic').value
        service_name = self.get_parameter('service_name').value
        self.output_file = self.get_parameter('output_file').value
        self.use_gps = self.get_parameter('use_gps').value

        self.first_msg = True
        self.gps_fix = None

        self.cli = self.create_client(CommandHome, service_name)
        if not self.cli.wait_for_service(timeout_sec=10.0):
            self.get_logger().error(f"Service '{service_name}' unavailable -> exit")
            rclpy.shutdown()
            return

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.create_subscription(Odometry, odom_topic, self._odom_cb, qos)
        if self.use_gps:
            self.create_subscription(NavSatFix, gps_topic, self._gps_cb, qos)

        self.get_logger().info(
            f"Listening on '{odom_topic}', will set home via '{service_name}' "
            f"and write '{self.output_file}'"
        )

    def _gps_cb(self, msg: NavSatFix) -> None:
        self.gps_fix = msg

    def _odom_cb(self, msg: Odometry) -> None:
        if not self.first_msg:
            return
        self.first_msg = False

        p = msg.pose.pose.position
        data = {'x': float(p.x), 'y': float(p.y), 'z': float(p.z)}

        if self.use_gps and self.gps_fix:
            data['latitude'] = self.gps_fix.latitude
            data['longitude'] = self.gps_fix.longitude
        else:
            data['latitude'] = data['longitude'] = 0.0

        try:
            with open(self.output_file, 'w') as f:
                yaml.dump(data, f)
            self.get_logger().info(f"Wrote home pose to '{self.output_file}'")
        except Exception as e:
            self.get_logger().error(f"Failed to write YAML: {e}")

        req = CommandHome.Request()
        req.current_gps = self.use_gps
        req.latitude = data['latitude']
        req.longitude = data['longitude']
        req.altitude = float(p.z)
        req.yaw = 0.0

        future = self.cli.call_async(req)
        future.add_done_callback(self._on_home_set)

    def _on_home_set(self, future) -> None:
        try:
            resp = future.result()
            if resp.success:
                self.get_logger().info('set_home succeeded')
            else:
                self.get_logger().error(f'set_home failed: {resp.result}')
        except Exception as e:
            self.get_logger().error(f'Exception during set_home: {e}')

        self.destroy_node()


def main() -> None:
    rclpy.init()
    node = HomeRecorder()
    if rclpy.ok():
        rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
