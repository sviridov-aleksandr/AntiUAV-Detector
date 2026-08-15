from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """Launch the full interceptor pipeline: video -> vision -> mavlink."""
    return LaunchDescription([
        # ─── Video source ───
        DeclareLaunchArgument(
            'source_type',
            default_value='file',
            description='Video source: file | rtsp | usb'
        ),
        DeclareLaunchArgument(
            'video_path',
            default_value='/home/alex/AntiUAV-Detector/video-FPV/Video/v2.mp4',
            description='Path to test video file (source_type=file)'
        ),
        DeclareLaunchArgument(
            'rtsp_url',
            default_value='rtsp://192.168.1.10:554/live',
            description='RTSP stream URL (source_type=rtsp, OpenIPC MC800S-V3)'
        ),
        DeclareLaunchArgument(
            'usb_device',
            default_value='/dev/video0',
            description='USB camera device (source_type=usb)'
        ),
        DeclareLaunchArgument(
            'target_fps',
            default_value='30',
            description='Target publish FPS'
        ),

        # ─── MAVLink ───
        DeclareLaunchArgument(
            'device',
            default_value='/dev/ttyACM0',
            description='MAVLink serial device'
        ),
        DeclareLaunchArgument(
            'simulation',
            default_value='false',
            description='Run without real hardware (test mode)'
        ),

        # ─── Vision ───
        DeclareLaunchArgument(
            'model_path',
            default_value='/home/alex/AntiUAV-Detector/runs/detect/train/runs/drone_v2-5/weights/best.pt',
            description='YOLO model path (.pt, .onnx or .engine)'
        ),
        DeclareLaunchArgument(
            'show_image',
            default_value='true',
            description='Show detection window (needs display)'
        ),
        DeclareLaunchArgument(
            'use_dual_band',
            default_value='false',
            description='Enable dual-band EO+IR fusion (requires IR stream)'
        ),
        DeclareLaunchArgument(
            'fusion_mode',
            default_value='eo_primary',
            description='Fusion mode: eo_primary | ir_primary | fused'
        ),
        DeclareLaunchArgument(
            'ir_topic',
            default_value='/camera/ir_image_raw',
            description='ROS topic for IR camera stream'
        ),
        DeclareLaunchArgument(
            'ir_threshold_mode',
            default_value='adaptive',
            description='IR tracker mode: adaptive | fixed | otsu | motion'
        ),

        # ─── Node 1: Video publisher (camera or test video) ───
        Node(
            package='uav_interceptor',
            executable='video_publisher',
            name='video_publisher',
            output='screen',
            parameters=[{
                'source_type': LaunchConfiguration('source_type'),
                'video_path': LaunchConfiguration('video_path'),
                'rtsp_url': LaunchConfiguration('rtsp_url'),
                'usb_device': LaunchConfiguration('usb_device'),
                'target_fps': LaunchConfiguration('target_fps'),
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
                {'ir_topic': LaunchConfiguration('ir_topic')},
                {'ir_threshold_mode': LaunchConfiguration('ir_threshold_mode')},
            ],
        ),

        # ─── Node 3: MAVLink bridge (real autopilot or simulation) ───
        Node(
            package='uav_interceptor',
            executable='mavlink_bridge',
            name='mavlink_bridge',
            output='screen',
            parameters=[
                {'device': LaunchConfiguration('device')},
                {'simulation': LaunchConfiguration('simulation')},
            ],
        ),

        # ─── Node 4: RViz2 visualization (optional, needs display) ───
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', '/home/alex/aerial_nav_ws/src/uav_interceptor/config/interceptor.rviz'],
        ),
    ])