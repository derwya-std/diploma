import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource


def generate_launch_description():
    world_arg = DeclareLaunchArgument(
        name='world',
        default_value='forest',
        description='Name of the world file (without .sdf) and SITL suffix'
    )
    headless_arg = DeclareLaunchArgument(
        name='headless',
        default_value='false',
        description='Run simulation in headless mode'
    )

    world = LaunchConfiguration('world')
    model = 'x500_mono_cam'

    pkg_share = get_package_share_directory('env_gen')
    env_gen_worlds = os.path.join(pkg_share, 'sim', 'worlds')
    existing = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    resource_path = ':'.join(filter(None, [env_gen_worlds, existing]))

    env = os.environ.copy()
    env['IGN_GAZEBO_RESOURCE_PATH'] = resource_path

    px4_env = env.copy()
    px4_env['WORLD'] = world

    px4_sitl = ExecuteProcess(
        cmd=[
            'make', 'px4_sitl', f'gz_{model}_$WORLD'
        ],
        cwd=os.path.expanduser('~/uav_sim/px4/PX4-Autopilot'),
        output='screen',
        additional_env=px4_env
    )

    def topic(path):
        return [
            TextSubstitution(text='/world/'),
            world,
            TextSubstitution(text=f'/{path}')
        ]

    def join(base_subst_list, suffix: str):
        return base_subst_list + [TextSubstitution(text=suffix)]

    def bridge_arg(topic_parts, ros_type, ign_type):
        return topic_parts + [TextSubstitution(text=f'@{ros_type}@{ign_type}')]

    # Create topic names
    imu_topic = topic(f'model/{model}_0/link/base_link/sensor/imu_sensor/imu')
    gps_topic = topic(f'model/{model}_0/link/base_link/sensor/navsat_sensor/navsat')
    mag_topic = topic(f'model/{model}_0/link/base_link/sensor/magnetometer_sensor/magnetometer')
    baro_topic = topic(f'model/{model}_0/link/base_link/sensor/air_pressure_sensor/air_pressure')
    img_topic = topic(f'model/{model}_0/link/camera_link/sensor/imager/image')
    depth_img_topic = topic(f'model/{model}_0/link/camera_link/sensor/depth_cam/depth_image')
    depth_pc_topic = topic(f'model/{model}_0/link/camera_link/sensor/depth_cam/depth_image/points')
    odom_topic = [TextSubstitution(text=f'model/{model}_0/odometry')]
    depth_ci_topic = topic(f'model/{model}_0/link/camera_link/sensor/depth_cam/camera_info')
    rgb_ci_topic = topic(f'model/{model}_0/link/camera_link/sensor/imager/camera_info')
    clock_topic = topic('clock')

    imu_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='imu_bridge',
        arguments=[bridge_arg(imu_topic, 'sensor_msgs/msg/Imu', 'gz.msgs.IMU')],
        remappings=[(imu_topic, '/imu')],
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    gps_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gps_bridge',
        arguments=[bridge_arg(gps_topic, 'sensor_msgs/msg/NavSatFix', 'gz.msgs.NavSat')],
        remappings=[(gps_topic, '/fix')],
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    mag_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='mag_bridge',
        arguments=[bridge_arg(mag_topic, 'sensor_msgs/msg/MagneticField', 'gz.msgs.Magnetometer')],
        remappings=[(mag_topic, '/mag')],
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    baro_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='baro_bridge',
        arguments=[bridge_arg(baro_topic, 'sensor_msgs/msg/FluidPressure', 'gz.msgs.FluidPressure')],
        remappings=[(baro_topic, '/pressure')],
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='clock_bridge',
        arguments=[bridge_arg(clock_topic, 'rosgraph_msgs/msg/Clock', 'gz.msgs.Clock')],
        remappings=[(clock_topic, '/clock')],
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    odom_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='clock_bridge',
        arguments=[bridge_arg(odom_topic, 'nav_msgs/msg/Odometry', 'gz.msgs.Odometry')],
        remappings=[(odom_topic, '/groundtruth/odom')],
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    image_bridge = Node(
        package='ros_gz_image',
        executable='image_bridge',
        name='image_bridge',
        arguments=[img_topic],
        remappings=[
            (join(img_topic, ''), '/camera/image'),
            (join(img_topic, '/compressed'), '/camera/image/compressed'),
            (join(img_topic, '/compressedDepth'), '/camera/image/compressedDepth'),
            (join(img_topic, '/zstd'), '/camera/image/zstd')
        ],
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    depth_image_bridge = Node(
        package='ros_gz_image',
        executable='image_bridge',
        name='depth_image_bridge',
        arguments=[depth_img_topic],
        remappings=[
            (join(depth_img_topic, ''), '/camera/depth/image_raw'),
            (join(depth_img_topic, '/compressed'), '/camera/depth/image_raw/compressed'),
            (join(depth_img_topic, '/compressedDepth'), '/camera/depth/image_raw/compressedDepth'),
            (join(depth_img_topic, '/zstd'), '/camera/depth/image_raw/zstd'),
        ],
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    depth_pc_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='depth_pc_bridge',
        arguments=[bridge_arg(depth_pc_topic, 'sensor_msgs/msg/PointCloud2', 'gz.msgs.PointCloudPacked')],
        remappings=[(depth_pc_topic, '/camera/depth/points')],
        parameters=[{'use_sim_time': True,
                     'qos_overridable': True,
                     'history': 'keep_last',
                     'depth': 5,
                     'reliability': 'reliable'
                     }],
        output='screen'
    )

    depth_ci_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='depth_camera_info_bridge',
        arguments=[bridge_arg(depth_ci_topic,
                              'sensor_msgs/msg/CameraInfo',
                              'gz.msgs.CameraInfo')],
        remappings=[(depth_ci_topic, '/camera/depth/camera_info')],
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    rgb_ci_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='rgb_camera_info_bridge',
        arguments=[bridge_arg(rgb_ci_topic,
                              'sensor_msgs/msg/CameraInfo',
                              'gz.msgs.CameraInfo')],
        remappings=[(rgb_ci_topic, '/camera/camera_info')],
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    mavros_launch = IncludeLaunchDescription(
        XMLLaunchDescriptionSource(os.path.join(
            get_package_share_directory('mavros'), 'launch', 'px4.launch'
        )),
        launch_arguments={
            'fcu_url': 'udp://:14540@127.0.0.1:14557',
            'namespace': 'mavros',
            'use_sim_time': 'true'
        }.items()
    )

    micro_xrce_agent = ExecuteProcess(
        cmd=[
            'MicroXRCEAgent', 'udp4', '--port', '8888'
        ],
        output='screen'
    )

    return LaunchDescription([
        world_arg,
        headless_arg,
        px4_sitl,
        imu_bridge,
        gps_bridge,
        mag_bridge,
        baro_bridge,
        image_bridge,
        depth_image_bridge,
        depth_pc_bridge,
        micro_xrce_agent,
        odom_bridge,
        depth_ci_bridge,
        rgb_ci_bridge,
        # mavros_launch,
        TimerAction(
            period=5.0,
            actions=[clock_bridge]
        )
    ])
