from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """Launch the full interceptor pipeline: WFB video → vision → MAVLink.

    Наземная архитектура (НСУ на Jetson Orin Nano):
      Борт (Orange Pi 3Z) → WFB radio → НСУ (Jetson) → MAVLink → Борт

    Узлы:
      1. wfb_video_receiver — приём EO+IR видео через WFB
      2. vision_node — YOLO + IR + OF + Fusion + PID + State Machine
      3. mavlink_bridge — MAVLink через WFB radio link
    """
    return LaunchDescription([
        # ─── Video source (WFB) ───
        DeclareLaunchArgument(
            'eo_udp_port', default_value='5600',
            description='UDP порт: EO-видео (WFB-ng gs_video0)'),
        DeclareLaunchArgument(
            'ir_udp_port', default_value='5601',
            description='UDP порт: IR-видео (WFB-ng gs_video1)'),
        DeclareLaunchArgument(
            'eo_format', default_value='h265',
            description='EO-видео кодек: h265 | h264'),
        DeclareLaunchArgument(
            'ir_format', default_value='h264',
            description='IR-видео кодек: h264 | h265'),
        DeclareLaunchArgument(
            'eo_width', default_value='1280'),
        DeclareLaunchArgument(
            'eo_height', default_value='720'),
        DeclareLaunchArgument(
            'ir_width', default_value='640'),
        DeclareLaunchArgument(
            'ir_height', default_value='480'),
        DeclareLaunchArgument(
            'enable_ir', default_value='true',
            description='Включить IR-поток (dual-band)'),

        # ─── MAVLink ───
        DeclareLaunchArgument(
            'link_mode', default_value='radio',
            description='MAVLink: radio (WFB-ng UDP) | direct (USB/UART)'),
        DeclareLaunchArgument(
            'mavlink_udp_port', default_value='14550',
            description='UDP порт для WFB-ng gs_mavlink'),
        DeclareLaunchArgument(
            'device', default_value='/dev/ttyACM0',
            description='MAVLink device (для link_mode=direct)'),
        DeclareLaunchArgument(
            'simulation', default_value='false',
            description='Тестовый режим (без железа)'),

        # ─── Vision ───
        DeclareLaunchArgument(
            'model_path',
            default_value='/home/alex/AntiUAV-Detector/runs/detect/train/runs/drone_v2-5/weights/best.engine',
            description='YOLO model (.pt, .onnx или .engine)'),
        DeclareLaunchArgument(
            'show_image', default_value='true'),
        DeclareLaunchArgument(
            'use_dual_band', default_value='true',
            description='Dual-band EO+IR fusion (требует IR-поток)'),
        DeclareLaunchArgument(
            'fusion_mode', default_value='eo_primary',
            description='eo_primary | ir_primary | fused'),
        DeclareLaunchArgument(
            'ir_threshold_mode', default_value='adaptive',
            description='adaptive | fixed | otsu | motion'),
        DeclareLaunchArgument(
            'link_latency', default_value='0.15',
            description='Round-trip задержка радиоканала (сек) для lead prediction'),

        # ─── Node 1: WFB Video Receiver ───
        Node(
            package='uav_interceptor',
            executable='wfb_video_receiver',
            name='wfb_video_receiver',
            output='screen',
            parameters=[{
                'eo_udp_port': LaunchConfiguration('eo_udp_port'),
                'ir_udp_port': LaunchConfiguration('ir_udp_port'),
                'eo_format': LaunchConfiguration('eo_format'),
                'ir_format': LaunchConfiguration('ir_format'),
                'eo_width': LaunchConfiguration('eo_width'),
                'eo_height': LaunchConfiguration('eo_height'),
                'ir_width': LaunchConfiguration('ir_width'),
                'ir_height': LaunchConfiguration('ir_height'),
                'enable_ir': LaunchConfiguration('enable_ir'),
            }],
        ),

        # ─── Node 2: Vision (YOLO + OF + IR + Dual-Band Fusion) ───
        Node(
            package='uav_interceptor',
            executable='vision_node',
            name='vision_node',
            output='screen',
            parameters=[
                {'model_path': LaunchConfiguration('model_path')},
                {'show_image': LaunchConfiguration('show_image')},
                {'use_dual_band': LaunchConfiguration('use_dual_band')},
                {'fusion_mode': LaunchConfiguration('fusion_mode')},
                {'ir_threshold_mode': LaunchConfiguration('ir_threshold_mode')},
                {'link_latency': LaunchConfiguration('link_latency')},
            ],
        ),

        # ─── Node 3: MAVLink bridge (radio или direct) ───
        Node(
            package='uav_interceptor',
            executable='mavlink_bridge',
            name='mavlink_bridge',
            output='screen',
            parameters=[
                {'link_mode': LaunchConfiguration('link_mode')},
                {'mavlink_udp_port': LaunchConfiguration('mavlink_udp_port')},
                {'device': LaunchConfiguration('device')},
                {'simulation': LaunchConfiguration('simulation')},
            ],
        ),
    ])
