#!/usr/bin/env python3
import math, random, rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_srvs.srv import Trigger
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus

PX4_CUSTOM_MAIN_MODE_OFFBOARD = 6.0


class OffboardRandomMission(Node):
    takeoff_alt = -42.0
    map_min_x, map_max_x = -260.0, 70.0
    map_min_y, map_max_y = -100.0, 100.0
    min_alt, max_alt = -50.0, -40.0
    waypoint_tolerance = 1.0
    min_hop = 25.0
    leg_timeout_mul = 2.0
    mission_duration = 3600
    pause_sec = 3.0
    min_speed = 2.0
    max_speed = 10.0
    smoothing_alpha = 0.05

    def __init__(self):
        super().__init__("offboard_random_mission")
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._offb_mode_pub = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", qos
        )
        self._traj_pub = self.create_publisher(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", qos
        )
        self._cmd_pub = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", qos
        )
        self.create_subscription(
            VehicleLocalPosition, "/fmu/out/vehicle_local_position", self._on_pos, qos
        )
        self.create_subscription(
            VehicleStatus, "/fmu/out/vehicle_status_v1", self._on_status, qos
        )

        self._pos = VehicleLocalPosition()
        self._status = VehicleStatus()
        self._counter = 0
        self._stage = 0
        self._pause_t0 = None
        self._mission_t0 = None
        self._leg_t0 = None
        self._leg_timeout = None
        self._square_waypoints = [
            (self.map_min_x, self.map_min_y, self.takeoff_alt),
            (self.map_max_x, self.map_min_y, self.takeoff_alt),
            (self.map_max_x, self.map_max_y, self.takeoff_alt),
            (self.map_min_x, self.map_max_y, self.takeoff_alt),
        ]
        self._sq_idx = 0
        self._current_wp = None
        self._leg_speed = self.min_speed
        self._prev_dir = (0.0, 0.0, 0.0)
        self._prev_yaw = 0.0

        self._rec_start_cli = self.create_client(Trigger, "/start_record")
        self._rec_stop_cli = self.create_client(Trigger, "/stop_record")
        self._recording_started = False
        self._recording_stopped = False

        self._rec_start_cli.wait_for_service()
        self._rec_stop_cli.wait_for_service()

        self.create_timer(0.1, self._tick)
        self.get_logger().info("Node started, waiting to arm and switch to OFFBOARD")

    def _on_pos(self, m):
        self._pos = m

    def _on_status(self, m):
        self._status = m

    def _pub_offb_mode(self):
        m = OffboardControlMode()
        m.position = True
        m.timestamp = int(self.get_clock().now().nanoseconds / 1e3)
        self._offb_mode_pub.publish(m)

    def _interp(self, new, prev):
        return self.smoothing_alpha * new + (1 - self.smoothing_alpha) * prev

    def _yaw_to(self, dx, dy):
        return math.atan2(dy, dx)

    def _publish_hover(self, tgt):
        sp = TrajectorySetpoint()
        sp.position = [tgt[0], tgt[1], tgt[2]]
        sp.velocity = [0.0, 0.0, 0.0]
        sp.acceleration = [0.0, 0.0, 0.0]
        sp.yaw = float(self._prev_yaw)
        sp.timestamp = int(self.get_clock().now().nanoseconds / 1e3)
        self._traj_pub.publish(sp)
        self.get_logger().info(
            f"Hovering at → pos=({tgt[0]:.1f},{tgt[1]:.1f},{tgt[2]:.1f})"
        )

    def _select_random_wp(self):
        while True:
            x = random.uniform(self.map_min_x, self.map_max_x)
            y = random.uniform(self.map_min_y, self.map_max_y)
            z = random.uniform(self.min_alt, self.max_alt)
            if math.dist((x, y, z), (self._pos.x, self._pos.y, self._pos.z)) >= self.min_hop:
                break
        self._current_wp = (x, y, z)
        self._leg_speed = random.uniform(self.min_speed, self.max_speed)
        dist = math.dist((x, y, z), (self._pos.x, self._pos.y, self._pos.z))
        self._leg_t0 = self.get_clock().now()
        self._leg_timeout = max(dist / self._leg_speed * self.leg_timeout_mul, 8.0)
        self.get_logger().info(
            f"Stage 5: new random WP=({x:.1f},{y:.1f},{z:.1f}), "
            f"speed={self._leg_speed:.1f}, timeout={self._leg_timeout:.1f}s"
        )

    def _pub_setpoint(self, tgt):
        dx = tgt[0] - self._pos.x
        dy = tgt[1] - self._pos.y
        dz = tgt[2] - self._pos.z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist > 1e-6:
            ux, uy, uz = dx / dist, dy / dist, dz / dist
        else:
            ux, uy, uz = 0.0, 0.0, 0.0

        sx = self._interp(ux, self._prev_dir[0])
        sy = self._interp(uy, self._prev_dir[1])
        sz = self._interp(uz, self._prev_dir[2])
        mag = math.sqrt(sx * sx + sy * sy + sz * sz)
        if mag > 1e-6:
            sx, sy, sz = sx / mag, sy / mag, sz / mag
        else:
            sx, sy, sz = 0.0, 0.0, 0.0
        self._prev_dir = (sx, sy, sz)

        vx, vy, vz = sx * self._leg_speed, sy * self._leg_speed, sz * self._leg_speed
        raw_yaw = self._yaw_to(sx, sy)
        yy = self._interp(raw_yaw, self._prev_yaw)
        self._prev_yaw = yy

        sp = TrajectorySetpoint()
        sp.position = [tgt[0], tgt[1], tgt[2]]
        sp.velocity = [vx, vy, vz]
        sp.acceleration = [0.0, 0.0, 0.0]
        sp.yaw = float(yy)
        sp.timestamp = int(self.get_clock().now().nanoseconds / 1e3)
        self._traj_pub.publish(sp)
        self.get_logger().info(
            f"Publish setpoint → pos=({tgt[0]:.1f},{tgt[1]:.1f},{tgt[2]:.1f}) "
            f"vel=({vx:.1f},{vy:.1f},{vz:.1f}) yaw={yy:.2f}"
        )

    def _at(self, tgt):
        d = math.dist((self._pos.x, self._pos.y, self._pos.z), tgt)
        at = d < self.waypoint_tolerance
        self.get_logger().debug(f"_at check → dist={d:.2f}, result={at}")
        return at

    def _cmd(self, cmd, **p):
        m = VehicleCommand()
        m.command = cmd
        m.target_system = m.source_system = 1
        m.target_component = m.source_component = 1
        m.from_external = True
        m.param1 = p.get("p1", 0.0)
        m.param2 = p.get("p2", 0.0)
        m.timestamp = int(self.get_clock().now().nanoseconds / 1e3)
        self._cmd_pub.publish(m)
        self.get_logger().info(f"Sent command → cmd={cmd}, p1={m.param1}, p2={m.param2}")

    def _call_start_record(self):
        if not self._recording_started:
            req = Trigger.Request()
            self._rec_start_cli.call_async(req)
            self._recording_started = True
            self.get_logger().info("Called /start_record service")

    def _call_stop_record(self):
        if not self._recording_stopped:
            req = Trigger.Request()
            self._rec_stop_cli.call_async(req)
            self._recording_stopped = True
            self.get_logger().info("Called /stop_record service")

    def _tick(self):
        self._pub_offb_mode()

        if self._stage == 0:
            if self._counter < 10:
                self._counter += 1
                self.get_logger().debug(f"Stage 0: counter={self._counter}")
                return
            self._cmd(
                VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
                p1=1.0,
                p2=PX4_CUSTOM_MAIN_MODE_OFFBOARD,
            )
            self._cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, p1=1.0)
            self._stage = 1
            self.get_logger().info("Transition to Stage 1: arm+offboard command sent")
            return

        if self._stage == 1:
            armed = self._status.arming_state == VehicleStatus.ARMING_STATE_ARMED
            offb = self._status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
            self.get_logger().debug(f"Stage 1: armed={armed}, offb={offb}")
            if armed and offb:
                self._stage = 2
                self.get_logger().info("Transition to Stage 2: takeoff")
                self._call_start_record()
            return

        if self._stage == 2:
            current_alt = self._pos.z
            if abs(current_alt - self.takeoff_alt) > self.waypoint_tolerance:
                tgt = (0.0, 0.0, self.takeoff_alt)
                self._pub_setpoint(tgt)
                self.get_logger().debug(f"Stage 2: climbing, current z={current_alt:.2f}")
                return
            self._sq_idx = 0
            self._stage = 3
            self.get_logger().info("Transition to Stage 3: square mission start")
            return

        if self._stage == 3:
            tgt = self._square_waypoints[self._sq_idx]
            if not self._at(tgt):
                self._pub_setpoint(tgt)
                self.get_logger().debug(f"Stage 3: flying to square WP {self._sq_idx}")
                return
            self._pause_t0 = self.get_clock().now()
            self._stage = 4
            self.get_logger().info(f"Transition to Stage 4: hover at square WP {self._sq_idx}")
            return

        if self._stage == 4:
            elapsed = (self.get_clock().now() - self._pause_t0).nanoseconds / 1e9
            self.get_logger().debug(f"Stage 4: hovering {elapsed:.2f}s at square WP {self._sq_idx}")
            if elapsed < self.pause_sec:
                self._publish_hover(self._square_waypoints[self._sq_idx])
                return
            self._sq_idx += 1
            if self._sq_idx < len(self._square_waypoints):
                self._stage = 3
                self.get_logger().info(f"Transition back to Stage 3: next square WP {self._sq_idx}")
            else:
                self._mission_t0 = self.get_clock().now()
                self._select_random_wp()
                self._stage = 5
                self.get_logger().info("Transition to Stage 5: random mission")
            return

        if self._stage == 5:
            now = self.get_clock().now()
            elapsed_mission = (now - self._mission_t0).nanoseconds / 1e9
            self.get_logger().debug(f"Stage 5: mission elapsed {elapsed_mission:.2f}s")
            if elapsed_mission >= self.mission_duration:
                self._stage = 7
                self.get_logger().info("Transition to Stage 7: final hover")
                return

            elapsed_leg = (now - self._leg_t0).nanoseconds / 1e9
            self.get_logger().debug(f"Stage 5: leg elapsed {elapsed_leg:.2f}s (timeout {self._leg_timeout:.2f}s)")
            if elapsed_leg > self._leg_timeout:
                self.get_logger().warn("Stage 5: leg timeout, choosing new WP")
                self._select_random_wp()

            if not self._at(self._current_wp):
                self._pub_setpoint(self._current_wp)
                self.get_logger().debug("Stage 5: flying to random WP")
                return

            self._pause_t0 = now
            self._stage = 6
            self.get_logger().info("Transition to Stage 6: hover at random WP")
            return

        if self._stage == 6:
            elapsed_hover = (self.get_clock().now() - self._pause_t0).nanoseconds / 1e9
            self.get_logger().debug(f"Stage 6: hovering {elapsed_hover:.2f}s at random WP")
            if elapsed_hover < self.pause_sec:
                self._publish_hover(self._current_wp)
                return
            self._select_random_wp()
            self._stage = 5
            self.get_logger().info("Stage 6: hover complete, back to Stage 5")
            return

        if self._stage == 7:
            p = (self._pos.x, self._pos.y, self._pos.z)
            self._publish_hover((p[0] + 1e-3, p[1], p[2]))
            self.get_logger().debug("Stage 7: final hover")
            self._call_stop_record()
            return


def main(args=None):
    rclpy.init(args=args)
    node = OffboardRandomMission()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
