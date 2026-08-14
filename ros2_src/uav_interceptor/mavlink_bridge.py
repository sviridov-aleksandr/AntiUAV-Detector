#!/usr/bin/env python3
"""
ROS2 Node: MAVLink Bridge.
Subscribes to /cmd_vel (from vision_node) and sends MAVLink commands to the autopilot.
Publishes telemetry (altitude, velocity, mode, attitude) to ROS2 topics.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist, Vector3Stamped
from std_msgs.msg import Float64, String
from sensor_msgs.msg import NavSatFix, Imu
from pymavlink import mavutil
import time
import math


class MavlinkBridge(Node):
    def __init__(self):
        super().__init__('mavlink_bridge')

        # Parameters
        self.declare_parameter('device', '/dev/ttyUSB0')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('simulation', True)
        self.declare_parameter('strike_servo_channel', 6)   # AUX servo channel
        self.declare_parameter('strike_servo_pwm', 2000)    # PWM to trigger detonator

        self.device = self.get_parameter('device').value
        self.baudrate = self.get_parameter('baudrate').value
        self.simulation = self.get_parameter('simulation').value
        self.strike_servo_channel = self.get_parameter('strike_servo_channel').value
        self.strike_servo_pwm = self.get_parameter('strike_servo_pwm').value

        # QoS for telemetry (reliable)
        telemetry_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        # ROS Subscriber (cmd_vel)
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10)

        # ROS Subscriber (strike command)
        self.strike_sub = self.create_subscription(
            String, '/interceptor/strike', self.strike_callback, 10)

        # ROS Publishers (telemetry)
        self.altitude_pub = self.create_publisher(
            Float64, '/telemetry/altitude', telemetry_qos)
        self.ground_speed_pub = self.create_publisher(
            Float64, '/telemetry/ground_speed', telemetry_qos)
        self.heading_pub = self.create_publisher(
            Float64, '/telemetry/heading', telemetry_qos)
        self.mode_pub = self.create_publisher(
            String, '/telemetry/mode', telemetry_qos)
        self.gps_pub = self.create_publisher(
            NavSatFix, '/telemetry/gps', telemetry_qos)
        self.attitude_pub = self.create_publisher(
            Imu, '/telemetry/attitude', telemetry_qos)
        self.velocity_pub = self.create_publisher(
            Vector3Stamped, '/telemetry/velocity', telemetry_qos)

        # MAVLink connection
        self.mavlink_conn = None
        if not self.simulation:
            try:
                self.mavlink_conn = mavutil.mavlink_connection(
                    self.device, baud=self.baudrate)
                self.get_logger().info(f'MAVLink connected to {self.device}')
                self.wait_for_heartbeat()
                self.set_guided_mode()
                self.request_data_streams()
            except Exception as e:
                self.get_logger().error(f'MAVLink connection failed: {e}')
                self.simulation = True
                self.get_logger().info('Falling back to simulation mode')
        else:
            self.get_logger().info('SIMULATION MODE: commands logged, not sent to hardware')

        self.cmd_vel_received = False
        self.last_cmd_time = time.time()
        self.boot_time = time.time()

        # Timers
        self.status_timer = self.create_timer(1.0, self.status_callback)
        self.telemetry_timer = self.create_timer(0.1, self.telemetry_callback)

        self.get_logger().info('MAVLink Bridge Started')

    def wait_for_heartbeat(self):
        self.get_logger().info('Waiting for autopilot heartbeat...')
        try:
            self.mavlink_conn.wait_heartbeat(timeout=10)
            self.get_logger().info('Heartbeat received!')
        except Exception as e:
            self.get_logger().error(f'No heartbeat: {e}')

    def set_guided_mode(self):
        """Set autopilot to GUIDED mode for velocity control."""
        try:
            self.mavlink_conn.mav.command_long_send(
                1, 1,
                mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                0,
                mavutil.mavlink.MAV_MODE_GUIDED_ARMED,
                0, 0, 0, 0, 0, 0,
            )
            self.get_logger().info('GUIDED mode requested')
        except Exception as e:
            self.get_logger().error(f'Failed to set GUIDED mode: {e}')

    def request_data_streams(self):
        """Request telemetry data streams from autopilot."""
        if self.mavlink_conn is None:
            return
        streams = [
            mavutil.mavlink.MAV_DATA_STREAM_POSITION,
            mavutil.mavlink.MAV_DATA_STREAM_EXTRA1,  # attitude
            mavutil.mavlink.MAV_DATA_STREAM_EXTRA2,  # velocity
            mavutil.mavlink.MAV_DATA_STREAM_EXTENDED_STATUS,
        ]
        for stream in streams:
            self.mavlink_conn.mav.request_data_stream_send(
                1, 1, stream, 10, 1)  # 10 Hz
        self.get_logger().info('Telemetry streams requested (10 Hz)')

    def cmd_vel_callback(self, msg: Twist):
        self.cmd_vel_received = True
        self.last_cmd_time = time.time()

        vx = float(msg.linear.x)
        vy = float(msg.linear.y)
        vz = float(msg.linear.z)
        yaw_rate = float(msg.angular.z)

        if self.simulation:
            self.get_logger().debug(
                f'[SIM] vx={vx:.2f} vy={vy:.2f} vz={vz:.2f} yaw={yaw_rate:.2f}')
        else:
            self.send_velocity_command(vx, vy, vz, yaw_rate)

    def strike_callback(self, msg: String):
        """Подрыв БЧ: MAV_CMD_DO_SET_SERVO на AUX-канале (channel 6, PWM 2000).
        Срабатывает при получении /interceptor/strike от vision_node,
        когда дальность до цели < kill_radius."""
        self.get_logger().warn(f'*** STRIKE COMMAND RECEIVED: {msg.data} ***')

        if self.simulation:
            self.get_logger().warn('[SIM] BЧ detonation simulated '
                                   '(would send DO_SET_SERVO)')
            return

        if self.mavlink_conn is None:
            self.get_logger().error('Cannot strike: no MAVLink connection')
            return

        try:
            # Move servo to detonation position (e.g. firing pin / igniter)
            self.mavlink_conn.mav.command_long_send(
                1, 1,
                mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
                0,
                self.strike_servo_channel,
                self.strike_servo_pwm,
                0, 0, 0, 0, 0,
            )
            self.get_logger().warn(
                f'MAV_CMD_DO_SET_SERVO sent: channel={self.strike_servo_channel} '
                f'pwm={self.strike_servo_pwm}')
        except Exception as e:
            self.get_logger().error(f'Strike command failed: {e}')

    def send_velocity_command(self, vx, vy, vz, yaw_rate):
        """Отправка команды скорости в автопилот через MAVLink.
        Использует set_position_target_local_ned_send БЕЗ thrust (16 аргументов),
        frame=BODY_OFFSET_NED (относительно тела аппарата).
        Перед вызовом автопилот должен быть в режиме GUIDED."""
        if self.mavlink_conn is None:
            return

        type_mask = (
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
        )

        time_boot_ms = int((time.time() - self.boot_time) * 1e3) % (2**32)
        self.mavlink_conn.mav.set_position_target_local_ned_send(
            time_boot_ms,
            1, 1,
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
            type_mask,
            float(0), float(0), float(0),
            float(vx), float(vy), float(vz),
            float(0), float(0), float(0),
            float(0), float(yaw_rate),
        )

    def telemetry_callback(self):
        """Read MAVLink messages and publish telemetry."""
        if self.simulation or self.mavlink_conn is None:
            return

        try:
            # Non-blocking read
            msg = self.mavlink_conn.recv_msg()
            if msg is None:
                return

            mtype = msg.get_type()

            if mtype == 'GLOBAL_POSITION_INT':
                # Altitude
                alt_msg = Float64()
                alt_msg.data = msg.relative_alt / 1000.0  # mm -> m
                self.altitude_pub.publish(alt_msg)

                # GPS
                gps_msg = NavSatFix()
                gps_msg.latitude = msg.lat / 1e7
                gps_msg.longitude = msg.lon / 1e7
                gps_msg.altitude = msg.alt / 1000.0
                self.gps_pub.publish(gps_msg)

                # Heading
                heading_msg = Float64()
                heading_msg.data = msg.hdg / 100.0  # centideg -> deg
                self.heading_pub.publish(heading_msg)

            elif mtype == 'VFR_HUD':
                # Ground speed
                speed_msg = Float64()
                speed_msg.data = msg.groundspeed  # m/s
                self.ground_speed_pub.publish(speed_msg)

            elif mtype == 'ATTITUDE':
                # Attitude (IMU)
                imu_msg = Imu()
                imu_msg.orientation.x = 0.0
                imu_msg.orientation.y = 0.0
                imu_msg.orientation.z = math.sin(msg.yaw / 2)
                imu_msg.orientation.w = math.cos(msg.yaw / 2)
                imu_msg.angular_velocity.z = float(msg.yawspeed)
                self.attitude_pub.publish(imu_msg)

            elif mtype == 'LOCAL_POSITION_NED':
                # Velocity
                vel_msg = Vector3Stamped()
                vel_msg.header.stamp = self.get_clock().now().to_msg()
                vel_msg.vector.x = float(msg.vx)
                vel_msg.vector.y = float(msg.vy)
                vel_msg.vector.z = float(msg.vz)
                self.velocity_pub.publish(vel_msg)

            elif mtype == 'HEARTBEAT':
                # Flight mode
                mode = mavutil.mode_string_v2(msg)
                mode_msg = String()
                mode_msg.data = mode if mode else 'UNKNOWN'
                self.mode_pub.publish(mode_msg)

        except Exception as e:
            self.get_logger().debug(f'Telemetry read error: {e}')

    def status_callback(self):
        if self.cmd_vel_received:
            if time.time() - self.last_cmd_time > 1.0:
                self.get_logger().info('No commands for 1s - sending STOP (hover)')
                if not self.simulation:
                    self.send_velocity_command(0, 0, 0, 0)
                self.cmd_vel_received = False


def main(args=None):
    rclpy.init(args=args)
    node = MavlinkBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
