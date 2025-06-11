#!/usr/bin/env python3
from __future__ import annotations
from typing import Any, Dict, Tuple, Optional
import pathlib
import zlib
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.serialization import serialize_message

from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata
from std_srvs.srv import Trigger, Trigger_Request, Trigger_Response

from sensor_msgs.msg import Image, Imu, CompressedImage
from px4_msgs.msg import VehicleOdometry
from rosgraph_msgs.msg import Clock

# ───── тільки те, що треба моделі ─────
SUB_TOPICS: Dict[str, Tuple[Any, str]] = {
    "/camera/image":                (Image,             "sensor_msgs/msg/Image"),
    "/imu":                         (Imu,               "sensor_msgs/msg/Imu"),
    "/fmu/out/vehicle_odometry":    (VehicleOdometry,   "px4_msgs/msg/VehicleOdometry"),
    "/clock":                       (Clock,             "rosgraph_msgs/msg/Clock"),
}

COMPRESSED_OUT = {
    "/camera/image": ("/camera/image/cpsd", "sensor_msgs/msg/CompressedImage"),
}

DOWNSAMPLE = {
    "/imu": 2,                           # e.g. 250 Hz → ~125 Hz
    "/fmu/out/vehicle_odometry": 2,      # e.g. 20 Hz → ~10 Hz
    "/clock": 1,                         # write every /clock message
}

QOS_SUB = QoSProfile(
    depth=5,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
)


class FlightRecorder(Node):
    def __init__(self) -> None:
        super().__init__("flight_recorder")
        # Note: we do NOT call declare_parameter("use_sim_time") because it’s already declared by rclpy

        self.declare_parameter("bag_name", "flight")
        self.declare_parameter("folder", str(pathlib.Path.home() / "bags"))
        self.declare_parameter("storage_id", "mcap")
        self.declare_parameter("use_timestamp", True)

        self._writer: Optional[SequentialWriter] = None
        self._subs: Dict[str, Any] = {}
        self._cnt: Dict[str, int] = {t: 0 for t in SUB_TOPICS}

        self.create_service(Trigger, "/start_record", self._srv_start)
        self.create_service(Trigger, "/stop_record", self._srv_stop)
        self.get_logger().info("Ready – call /start_record")

    # ─────────── services ───────────
    def _srv_start(self, _req: Trigger_Request, res: Trigger_Response) -> Trigger_Response:
        if self._writer:
            res.success = False
            res.message = "already running"
            return res

        # Reset counters so each bag starts fresh
        self._cnt = {t: 0 for t in SUB_TOPICS}

        name = self.get_parameter("bag_name").value
        if self.get_parameter("use_timestamp").value:
            name += "_" + datetime.now().strftime("%Y%m%d_%H%M%S")

        bag_dir = pathlib.Path(self.get_parameter("folder").value) / name
        bag_dir.parent.mkdir(parents=True, exist_ok=True)

        # Initialize and open the writer
        self._writer = SequentialWriter()
        self._writer.open(
            StorageOptions(uri=str(bag_dir),
                           storage_id=self.get_parameter("storage_id").value),
            ConverterOptions("cdr", "cdr"),
        )

        # Create topics and subscriptions
        for topic, (msg_cls, ros_type) in SUB_TOPICS.items():
            if topic in COMPRESSED_OUT:
                out_topic, out_type = COMPRESSED_OUT[topic]
                self._writer.create_topic(
                    TopicMetadata(0, out_topic, out_type, "cdr", [], "")
                )
                self._subs[topic] = self.create_subscription(
                    msg_cls,
                    topic,
                    lambda m, t=topic: self._img_cb(m, t),
                    QOS_SUB,
                )
            else:
                self._writer.create_topic(
                    TopicMetadata(0, topic, ros_type, "cdr", [], "")
                )
                self._subs[topic] = self.create_subscription(
                    msg_cls,
                    topic,
                    lambda m, t=topic: self._gen_cb(m, t),
                    QOS_SUB,
                )

        res.success = True
        res.message = f"Recording to: {bag_dir}"
        self.get_logger().info(res.message)
        return res

    def _srv_stop(self, _req: Trigger_Request, res: Trigger_Response) -> Trigger_Response:
        if not self._writer:
            res.success = False
            res.message = "not running"
            return res

        self._writer.close()
        self._writer = None

        for sub in self._subs.values():
            self.destroy_subscription(sub)
        self._subs.clear()

        self.get_logger().info("Bag closed")
        res.success = True
        res.message = "stopped"
        return res

    # ─────────── callbacks ───────────
    def _gen_cb(self, msg: Any, topic: str) -> None:
        if not self._writer:
            return

        self._cnt[topic] += 1
        if self._cnt[topic] % DOWNSAMPLE.get(topic, 1) != 0:
            return

        # For VehicleOdometry, override timestamp with sim time (use_sim_time=True)
        if topic == "/fmu/out/vehicle_odometry":
            vo: VehicleOdometry = msg
            bag_ts = self.get_clock().now().nanoseconds
            self._writer.write(topic, serialize_message(vo), bag_ts)
            return

        # For all other topics, use get_clock().now() so they sync with /clock
        bag_ts = self.get_clock().now().nanoseconds
        self._writer.write(topic, serialize_message(msg), bag_ts)

    def _img_cb(self, msg: Image, topic: str) -> None:
        if not self._writer:
            return

        # Convert Image → CompressedImage
        ci = CompressedImage()
        ci.header = msg.header
        ci.format = "zlib"
        ci.data = zlib.compress(bytes(msg.data))

        out_topic, _ = COMPRESSED_OUT[topic]
        bag_ts = self.get_clock().now().nanoseconds
        self._writer.write(out_topic, serialize_message(ci), bag_ts)


# ────────────────────────────────────
def main() -> None:
    rclpy.init()
    rclpy.spin(FlightRecorder())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
