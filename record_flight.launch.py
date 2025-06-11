from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    world_arg = DeclareLaunchArgument('world', default_value='baylands')
    flight_id_arg = DeclareLaunchArgument('flight_id', default_value='demo')

    output_bag = [LaunchConfiguration('world'), '_', LaunchConfiguration('flight_id')]

    recorder = Node(
        package='flight_recorder',
        executable='recorder_node',
        name='flight_recorder',
        parameters=[{
            'bag_name': output_bag,
            'topics': [
                '/camera/image',
                '/camera/depth/image_raw',
                '/imu',
                '/groundtruth/pose'
            ]
        }],
        output='screen'
    )

    return LaunchDescription([
        world_arg,
        flight_id_arg,
        TimerAction(period=2.0, actions=[recorder])
    ])
