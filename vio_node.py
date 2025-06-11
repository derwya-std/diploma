#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import Image, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistWithCovariance
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
import message_filters
from vio_nn.model import VIOCNN  # Import your trained model


def quaternion_from_euler(roll, pitch, yaw):
    """
    Convert Euler angles to quaternion.
    Replace tf_transformations.quaternion_from_euler to avoid NumPy 2.0 compatibility issues.
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return [x, y, z, w]


class VIOLocalizer(Node):
    def __init__(self):
        super().__init__('vio_localizer')

        # Parameters (adjust based on your training config)
        self.declare_parameters(
            namespace='',
            parameters=[
                ('model_path', '/home/oleksii/uav_sim/colcon_ws/src/vio_nn/vio_nn/checkpoints/best_checkpoint.pth'),
                ('sequence_length', 6),
                ('imu_per_frame', 50),
                ('image_width', 640),
                ('image_height', 480),
                ('publish_rate', 20.0),
            ]
        )

        # Load parameters
        model_path = self.get_parameter('model_path').value
        self.sequence_length = self.get_parameter('sequence_length').value
        self.imu_per_frame = self.get_parameter('imu_per_frame').value
        self.image_size = (self.get_parameter('image_width').value,
                           self.get_parameter('image_height').value)

        # Load trained model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.model.eval()

        # Buffers
        self.image_buffer = deque(maxlen=self.sequence_length)
        self.imu_buffer = deque(maxlen=10000)  # Large buffer for IMU history

        # State variables
        self.current_pose = np.zeros(6)  # [x, y, z, roll, pitch, yaw]
        self.last_image_time = None

        # QoS profiles for sensor data
        image_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10
        )

        imu_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=100
        )

        # Subscribers
        self.image_sub = message_filters.Subscriber(self, Image, '/camera/image', qos_profile=image_qos)
        self.imu_sub = message_filters.Subscriber(self, Imu, '/imu', qos_profile=imu_qos)

        # Approximate time synchronizer
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.imu_sub],
            queue_size=10,
            slop=0.05  # 50ms tolerance
        )
        self.ts.registerCallback(self.sensor_callback)

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, '/vio/odometry', 10)
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/vio/pose', 10)

        # Inference timer
        publish_rate = self.get_parameter('publish_rate').value
        self.create_timer(1.0 / publish_rate, self.publish_pose)

        self.get_logger().info("VIO Localizer initialized")

    def load_model(self, model_path):
        """Load trained model from checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)
        model = VIOCNN(
            img_channels=3,
            imu_dim=6,
            emb_dim=192,
            hidden_size=256,
            gru_layers=3,
        ).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def sensor_callback(self, image_msg, imu_msg):
        """Process synchronized image and IMU data"""
        # Store IMU data with timestamp
        imu_time = self.get_timestamp(imu_msg)
        imu_data = [
            float(imu_msg.linear_acceleration.x),
            float(imu_msg.linear_acceleration.y),
            float(imu_msg.linear_acceleration.z),
            float(imu_msg.angular_velocity.x),
            float(imu_msg.angular_velocity.y),
            float(imu_msg.angular_velocity.z)
        ]
        self.imu_buffer.append((imu_time, imu_data))

        # Process image
        self.process_image(image_msg)

    def get_timestamp(self, msg):
        """Get timestamp in nanoseconds"""
        return msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec

    def process_image(self, image_msg):
        """Convert and store image with timestamp"""
        try:
            # Convert ROS Image to OpenCV
            if image_msg.encoding == 'bgr8':
                cv_image = np.frombuffer(image_msg.data, dtype=np.uint8).reshape(
                    image_msg.height, image_msg.width, 3)
            else:
                # Convert to BGR if needed
                cv_image = cv2.cvtColor(np.frombuffer(image_msg.data, dtype=np.uint8).reshape(
                    image_msg.height, image_msg.width, -1), cv2.COLOR_RGB2BGR)

            # Resize and normalize
            cv_image = cv2.resize(cv_image, self.image_size)
            tensor_image = torch.from_numpy(cv_image).permute(2, 0, 1).float() / 255.0
            tensor_image = (tensor_image - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / \
                           torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

            # Store with timestamp
            image_time = self.get_timestamp(image_msg)
            self.image_buffer.append((image_time, tensor_image))
            self.last_image_time = image_time

        except Exception as e:
            self.get_logger().error(f"Image processing failed: {str(e)}")

    def get_imu_window(self, image_time):
        """Get IMU samples preceding an image"""
        # Find closest IMU samples before image time
        samples = []
        for ts, imu in reversed(self.imu_buffer):
            if ts <= image_time:
                samples.append(imu)
                if len(samples) >= self.imu_per_frame:
                    break

        # Pad if necessary
        if len(samples) < self.imu_per_frame:
            samples = [np.zeros(6)] * (self.imu_per_frame - len(samples)) + samples

        return np.array(samples)

    def run_inference(self):
        """Run model inference on current buffers"""
        if len(self.image_buffer) < self.sequence_length:
            return None, None

        # Prepare image sequence
        image_sequence = []
        imu_windows = []
        timestamps = []

        for ts, img in self.image_buffer:
            image_sequence.append(img)
            imu_windows.append(self.get_imu_window(ts))
            timestamps.append(ts)

        # Convert to tensors
        images_tensor = torch.stack(image_sequence).unsqueeze(0).to(self.device)
        imu_tensor = torch.tensor(np.array(imu_windows), dtype=torch.float32).unsqueeze(0).to(self.device)
        seq_lengths = torch.tensor([self.sequence_length]).to(self.device)

        # Inference
        with torch.no_grad():
            deltas, confidences = self.model(images_tensor, imu_tensor, seq_lengths)

        # Get only the latest prediction
        return deltas[0, -1].cpu().numpy(), confidences[0, -1].item()

    def publish_pose(self):
        """Main publishing function"""
        if self.last_image_time is None:
            return

        # Run inference
        try:
            delta, confidence = self.run_inference()
            if delta is None:
                return

            # Ensure delta is numpy array and handle NaN/inf values
            delta = np.array(delta, dtype=np.float64)
            if np.any(np.isnan(delta)) or np.any(np.isinf(delta)):
                self.get_logger().warn("Invalid delta values detected, skipping update")
                return

            # Update pose (simple Euler integration)
            self.current_pose[:3] += delta[:3]  # Position update
            self.current_pose[3:] += delta[3:]  # Orientation update

        except Exception as e:
            self.get_logger().error(f"Inference failed: {str(e)}")
            return

        # Create odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'

        # Set position
        odom_msg.pose.pose.position.x = float(self.current_pose[0])
        odom_msg.pose.pose.position.y = float(self.current_pose[1])
        odom_msg.pose.pose.position.z = float(self.current_pose[2])

        # Convert Euler to quaternion
        q = quaternion_from_euler(
            float(self.current_pose[3]),  # roll
            float(self.current_pose[4]),  # pitch
            float(self.current_pose[5])  # yaw
        )
        odom_msg.pose.pose.orientation.x = float(q[0])
        odom_msg.pose.pose.orientation.y = float(q[1])
        odom_msg.pose.pose.orientation.z = float(q[2])
        odom_msg.pose.pose.orientation.w = float(q[3])

        # Create twist (using delta as velocity approximation)
        twist = TwistWithCovariance()
        twist.twist.linear.x = float(delta[0] * self.get_parameter('publish_rate').value)
        twist.twist.linear.y = float(delta[1] * self.get_parameter('publish_rate').value)
        twist.twist.linear.z = float(delta[2] * self.get_parameter('publish_rate').value)
        twist.twist.angular.x = float(delta[3] * self.get_parameter('publish_rate').value)
        twist.twist.angular.y = float(delta[4] * self.get_parameter('publish_rate').value)
        twist.twist.angular.z = float(delta[5] * self.get_parameter('publish_rate').value)
        odom_msg.twist = twist

        # Publish
        try:
            self.odom_pub.publish(odom_msg)

            # Publish pose separately if needed
            pose_msg = PoseWithCovarianceStamped()
            pose_msg.header = odom_msg.header
            pose_msg.pose.pose = odom_msg.pose.pose
            self.pose_pub.publish(pose_msg)

        except Exception as e:
            self.get_logger().error(f"Publishing failed: {str(e)}")


def main(args=None):
    rclpy.init(args=args)
    node = VIOLocalizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()