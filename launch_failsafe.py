#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    common_ns = 'failsafe'

    delay = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='failsafe',
                executable='heartbeat_node',
                namespace=common_ns,
                name='operator_heartbeat',
                output='screen',
                parameters=[{'use_sim_time': True,
                             'topic': '/operator/heartbeat',
                             'rate_hz': 10.0}],
            ),
            Node(
                package='failsafe',
                executable='heartbeat_watchdog',
                namespace=common_ns,
                name='heartbeat_watchdog',
                output='screen',
                parameters=[{'use_sim_time': True,
                             'timeout': 1.0,
                             'default_mode': 'AUTO.LOITER'}],
            ),
            Node(
                package='failsafe',
                executable='home_recorder',
                namespace=common_ns,
                name='home_recorder',
                output='screen',
                parameters=[{'use_sim_time': True,
                             'odom_topic': '/mavros/local_position/odom',
                             'gps_topic': '/mavros/global_position/raw/fix',
                             'service_name': '/mavros/cmd/set_home',
                             'output_file': 'home_pose.yaml',
                             'use_gps': False}],
            ),
            Node(
                package='failsafe',
                executable='gps_jammer',
                namespace=common_ns,
                name='gps_jammer',
                output='screen',
                parameters=[{'use_sim_time': True}],
            ),
        ]
    )

    return LaunchDescription([delay])
