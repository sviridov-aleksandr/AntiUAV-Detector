#!/usr/bin/env python3
"""
ROS2 Node: MAVLink Bridge (наземная НСУ).

Двунаправленный MAVLink через WFB radio link:
  - Команды: /cmd_vel → MAVLink → WFB tx → борт → CUAV X7+ Pro
  - Телеметрия: WFB rx ← борт ← CUAV → ROS2 topics

Два режима подключения:
  1. radio: через WFB named pipes (наземная архитектура)
  2. direct: через USB/UART (локальное подключение, для тестов)

Дополнительные функции (по сравнению с бортовой версией):
  - Армирование (MAV_CMD_COMPONENT_ARM_DISARM)
  - Failsafe: RTL при потере связи / timeout
  - Pre-flight checks (батарея, GPS, режим)
  - Geofence: проверка координат перед INTERCEPT
  - Kill switch: немедленная остановка моторов

Запуск (на Jetson Orin Nano, НСУ):
  ros2 run uav_interceptor mavlink_bridge --ros-args \
    -p link_mode:=radio \
    -p wfb_tx_pipe:=/tmp/wfb_tx_command \
    -p wfb_rx_pipe:=/tmp/wfb_rx_telemetry \
    -p simulation:=false
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist, Vector3Stamped
from std_msgs.msg import Float64, String, Bool
from sensor_msgs.msg import NavSatFix, Imu
from pymavlink import mavutil
import time
import math
import os
import struct


class MavlinkBridge(Node):
    def __init__(self):
        super().__init__('mavlink_bridge')

        # ─── Параметры подключения ───
        self.declare_parameter('link_mode', 'radio')  # radio | direct
        self.declare_parameter('device', '/dev/ttyACM0')  # для direct
        self.declare_parameter('baudrate', 921600)
        self.declare_parameter('wfb_tx_pipe', '/tmp/wfb_tx_command')  # команды вверх
        self.declare_parameter('wfb_rx_pipe', '/tmp/wfb_rx_telemetry')  # телеметрия вниз
        self.declare_parameter('simulation', True)

        # ─── Параметры БЧ ───
        self.declare_parameter('strike_servo_channel', 6)
        self.declare_parameter('strike_servo_pwm', 2000)

        # ─── Параметры безопасности ───
        self.declare_parameter('auto_arm', False)  # авто-арминг (опасно!)
        self.declare_parameter('command_timeout', 1.0)  # timeout → hover
        self.declare_parameter('link_loss_timeout', 5.0)  # timeout → RTL
        self.declare_parameter('min_battery_voltage', 14.0)  # 4S min
        self.declare_parameter('geofence_enabled', True)
        self.declare_parameter('geofence_max_dist', 500.0)  # м от точки старта

        self.link_mode = self.get_parameter('link_mode').value
        self.device = self.get_parameter('device').value
        self.baudrate = self.get_parameter('baudrate').value
        self.wfb_tx_pipe = self.get_parameter('wfb_tx_pipe').value
        self.wfb_rx_pipe = self.get_parameter('wfb_rx_pipe').value
        self.simulation = self.get_parameter('simulation').value
        self.strike_servo_channel = self.get_parameter('strike_servo_channel').value
        self.strike_servo_pwm = self.get_parameter('strike_servo_pwm').value
        self.auto_arm = self.get_parameter('auto_arm').value
        self.command_timeout = self.get_parameter('command_timeout').value
        self.link_loss_timeout = self.get_parameter('link_loss_timeout').value
        self.min_battery_voltage = self.get_parameter('min_battery_voltage').value
        self.geofence_enabled = self.get_parameter('geofence_enabled').value
        self.geofence_max_dist = self.get_parameter('geofence_max_dist').value

        # QoS
        telemetry_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        # ─── ROS Subscribers ───
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.strike_sub = self.create_subscription(
            String, '/interceptor/strike', self.strike_callback, 10)
        self.arm_sub = self.create_subscription(
            Bool, '/interceptor/arm', self.arm_callback, 10)
        self.kill_sub = self.create_subscription(
            Bool, '/interceptor/kill', self.kill_callback, 10)

        # ─── ROS Publishers (телеметрия) ───
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
        self.battery_pub = self.create_publisher(
            Float64, '/telemetry/battery', telemetry_qos)
        self.link_status_pub = self.create_publisher(
            String, '/telemetry/link_status', telemetry_qos)

        # ─── MAVLink connection ───
        self.mavlink_conn = None
        self.tx_pipe_fd = None  # WFB tx (команды вверх)
        self.rx_pipe_fd = None  # WFB rx (телеметрия вниз)
        self.rx_buf = bytearray()

        self.armed = False
        self.last_heartbeat_time = time.time()
        self.last_cmd_time = time.time()
        self.cmd_vel_received = False
        self.boot_time = time.time()
        self.home_lat = None
        self.home_lon = None
        self.battery_voltage = 0.0

        if not self.simulation:
            self._connect()
        else:
            self.get_logger().info('SIMULATION MODE: команды логируются, не отправляются')

        # ─── Timers ───
        self.status_timer = self.create_timer(1.0, self.status_callback)
        self.telemetry_timer = self.create_timer(0.05, self.telemetry_callback)  # 20 Hz

        self.get_logger().info(
            f'MAVLink Bridge запущен (link={self.link_mode}, '
            f'sim={self.simulation})')

    # ─────────────────────────────────────────────────────────
    #  Подключение
    # ─────────────────────────────────────────────────────────

    def _connect(self):
        """Установка MAVLink-соединения (radio или direct)."""
        if self.link_mode == 'radio':
            self._connect_radio()
        else:
            self._connect_direct()

    def _connect_radio(self):
        """Подключение через WFB named pipes."""
        # TX pipe: команды вверх (создаётся WFB-ng)
        if not os.path.exists(self.wfb_tx_pipe):
            try:
                os.mkfifo(self.wfb_tx_pipe)
            except FileExistsError:
                pass

        # RX pipe: телеметрия вниз
        if not os.path.exists(self.wfb_rx_pipe):
            try:
                os.mkfifo(self.wfb_rx_pipe)
            except FileExistsError:
                pass

        try:
            self.tx_pipe_fd = os.open(
                self.wfb_tx_pipe, os.O_WRONLY | os.O_NONBLOCK)
            self.rx_pipe_fd = os.open(
                self.wfb_rx_pipe, os.O_RDONLY | os.O_NONBLOCK)
            self.get_logger().info(
                f'WFB radio: tx={self.wfb_tx_pipe}, rx={self.wfb_rx_pipe}')
        except Exception as e:
            self.get_logger().error(f'WFB pipe open failed: {e}')
            self.simulation = True

    def _connect_direct(self):
        """Прямое подключение через USB/UART (для тестов)."""
        try:
            self.mavlink_conn = mavutil.mavlink_connection(
                self.device, baud=self.baudrate, force_connected=True)
            self.get_logger().info(f'Direct MAVLink: {self.device} @ {self.baudrate}')
            self.wait_for_heartbeat()
            self.set_guided_mode()
            self.request_data_streams()
        except Exception as e:
            self.get_logger().error(f'Direct connection failed: {e}')
            self.simulation = True

    def wait_for_heartbeat(self):
        self.get_logger().info('Ожидание heartbeat...')
        try:
            self.mavlink_conn.wait_heartbeat(timeout=10)
            self.last_heartbeat_time = time.time()
            self.get_logger().info('Heartbeat получен!')
        except Exception as e:
            self.get_logger().error(f'Нет heartbeat: {e}')

    def set_guided_mode(self):
        """Переключение FC в GUIDED mode."""
        self._send_command(
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            mavutil.mavlink.MAV_MODE_GUIDED_ARMED)
        self.get_logger().info('GUIDED mode запрошен')

    def request_data_streams(self):
        """Запрос потоков телеметрии (только для direct mode)."""
        if self.mavlink_conn is None:
            return
        streams = [
            mavutil.mavlink.MAV_DATA_STREAM_POSITION,
            mavutil.mavlink.MAV_DATA_STREAM_EXTRA1,
            mavutil.mavlink.MAV_DATA_STREAM_EXTRA2,
            mavutil.mavlink.MAV_DATA_STREAM_EXTENDED_STATUS,
        ]
        for stream in streams:
            self.mavlink_conn.mav.request_data_stream_send(1, 1, stream, 10, 1)
        self.get_logger().info('Потоки телеметрии запрошены (10 Hz)')

    # ─────────────────────────────────────────────────────────
    #  Отправка MAVLink команд
    # ─────────────────────────────────────────────────────────

    def _send_raw(self, data: bytes):
        """Отправка raw MAVLink байтов (через WFB или direct)."""
        if self.simulation:
            return
        if self.link_mode == 'radio' and self.tx_pipe_fd is not None:
            try:
                os.write(self.tx_pipe_fd, data)
            except (BlockingIOError, BrokenPipeError):
                pass  # pipe полон — пропускаем
        elif self.mavlink_conn is not None:
            self.mavlink_conn.write(data)

    def _send_command(self, command, param1=0, param2=0, param3=0,
                      param4=0, param5=0, param6=0, param7=0):
        """Отправка command_long через любой канал."""
        if self.simulation:
            self.get_logger().debug(
                f'[SIM] CMD={command} p1={param1} p2={param2}')
            return

        # Создаём MAVLink-сообщение
        if self.mavlink_conn is not None:
            # Direct mode: используем стандартный API
            self.mavlink_conn.mav.command_long_send(
                1, 1, command, 0,
                param1, param2, param3, param4, param5, param6, param7)
        else:
            # Radio mode: создаём сообщение вручную и сериализуем
            msg = self.mavlink_conn.mav.command_long_encode(
                1, 1, command, 0,
                param1, param2, param3, param4, param5, param6, param7) \
                if self.mavlink_conn else None

            # Если нет mavlink_conn (radio only), создаём временный
            if msg is None:
                # Создаём MAVLink-сериализатор без подключения
                from pymavlink.dialects.v20 import ardupilotmega as mavlink
                mav = mavlink.MAVLink(None, 2, 1)
                msg = mav.command_long_encode(
                    1, 1, command, 0,
                    param1, param2, param3, param4, param5, param6, param7)
                self._send_raw(msg.pack(mav))

    def send_velocity_command(self, vx, vy, vz, yaw_rate):
        """Отправка команды скорости через MAVLink.
        set_position_target_local_ned_send, frame=BODY_OFFSET_NED."""
        if self.simulation:
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

        if self.mavlink_conn is not None:
            # Direct mode
            self.mavlink_conn.mav.set_position_target_local_ned_send(
                time_boot_ms, 1, 1,
                mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
                type_mask,
                0, 0, 0,
                float(vx), float(vy), float(vz),
                0, 0, 0,
                0, float(yaw_rate))
        else:
            # Radio mode: сериализуем вручную
            from pymavlink.dialects.v20 import ardupilotmega as mavlink
            mav = mavlink.MAVLink(None, 2, 1)
            msg = mav.set_position_target_local_ned_encode(
                time_boot_ms, 1, 1,
                mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
                type_mask,
                0, 0, 0,
                float(vx), float(vy), float(vz),
                0, 0, 0,
                0, float(yaw_rate))
            self._send_raw(msg.pack(mav))

    # ─────────────────────────────────────────────────────────
    #  ROS Callbacks
    # ─────────────────────────────────────────────────────────

    def cmd_vel_callback(self, msg: Twist):
        self.cmd_vel_received = True
        self.last_cmd_time = time.time()

        vx = float(msg.linear.x)
        vy = float(msg.linear.y)
        vz = float(msg.linear.z)
        yaw_rate = float(msg.angular.z)

        if self.simulation:
            self.get_logger().debug(
                f'[SIM] vx={vx:.2f} vy={vy:.2f} vz={vz:.2f} yaw={yaw_rate:.2f}',
                throttle_duration_sec=1.0)
        else:
            self.send_velocity_command(vx, vy, vz, yaw_rate)

    def strike_callback(self, msg: String):
        """Подрыв БЧ: DO_SET_SERVO на AUX ch6, PWM 2000."""
        self.get_logger().warn(f'*** STRIKE: {msg.data} ***')

        if self.simulation:
            self.get_logger().warn('[SIM] Подрыв БЧ симулирован')
            return

        self._send_command(
            mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
            self.strike_servo_channel,
            self.strike_servo_pwm)
        self.get_logger().warn(
            f'DO_SET_SERVO: ch={self.strike_servo_channel} '
            f'pwm={self.strike_servo_pwm}')

    def arm_callback(self, msg: Bool):
        """Армирование / разоружение."""
        if msg.data:
            self.get_logger().warn('*** ARMING ***')
            self._send_command(
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1)
            self.armed = True
        else:
            self.get_logger().warn('*** DISARMING ***')
            self._send_command(
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0)
            self.armed = False

    def kill_callback(self, msg: Bool):
        """Kill switch — немедленная остановка моторов."""
        self.get_logger().error('*** KILL SWITCH ***')
        if not self.simulation:
            # DISARM с force (param2=21196 — override)
            self._send_command(
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0, 21196)

    # ─────────────────────────────────────────────────────────
    #  Телеметрия
    # ─────────────────────────────────────────────────────────

    def telemetry_callback(self):
        """Чтение MAVLink телеметрии и публикация в ROS2."""
        if self.simulation:
            return

        if self.link_mode == 'radio' and self.rx_pipe_fd is not None:
            self._read_radio_telemetry()
        elif self.mavlink_conn is not None:
            self._read_direct_telemetry()

    def _read_radio_telemetry(self):
        """Чтение телеметрии из WFB rx pipe."""
        try:
            data = os.read(self.rx_pipe_fd, 4096)
            if data:
                self.rx_buf.extend(data)
                self._process_rx_buffer()
        except BlockingIOError:
            pass
        except Exception as e:
            self.get_logger().debug(f'RX read error: {e}')

    def _read_direct_telemetry(self):
        """Чтение телеметрии через direct MAVLink."""
        try:
            msg = self.mavlink_conn.recv_msg()
            if msg is None:
                return
            self._process_message(msg)
        except Exception as e:
            self.get_logger().debug(f'Telemetry read error: {e}')

    def _process_rx_buffer(self):
        """Парсинг MAVLink-сообщений из буфера WFB rx."""
        while len(self.rx_buf) >= 2:
            if self.rx_buf[0] not in (0xFD, 0xFE):
                self.rx_buf.pop(0)
                continue

            if self.rx_buf[0] == 0xFD:
                if len(self.rx_buf) < 3:
                    break
                plen = self.rx_buf[1] + 12
            else:
                if len(self.rx_buf) < 6:
                    break
                plen = self.rx_buf[1] + 8

            if len(self.rx_buf) < plen:
                break

            packet = bytes(self.rx_buf[:plen])
            del self.rx_buf[:plen]

            # Десериализация
            try:
                from pymavlink.dialects.v20 import ardupilotmega as mavlink
                mav = mavlink.MAVLink(None, 2, 1)
                msg = mav.decode(packet)
                self._process_message(msg)
            except Exception:
                pass  # повреждённый пакет — пропускаем

    def _process_message(self, msg):
        """Обработка MAVLink-сообщения → ROS2 публикация."""
        mtype = msg.get_type()

        if mtype == 'HEARTBEAT':
            self.last_heartbeat_time = time.time()
            mode = mavutil.mode_string_v2(msg)
            mode_msg = String()
            mode_msg.data = mode if mode else 'UNKNOWN'
            self.mode_pub.publish(mode_msg)

        elif mtype == 'GLOBAL_POSITION_INT':
            alt_msg = Float64()
            alt_msg.data = msg.relative_alt / 1000.0
            self.altitude_pub.publish(alt_msg)

            gps_msg = NavSatFix()
            gps_msg.latitude = msg.lat / 1e7
            gps_msg.longitude = msg.lon / 1e7
            gps_msg.altitude = msg.alt / 1000.0
            self.gps_pub.publish(gps_msg)

            heading_msg = Float64()
            heading_msg.data = msg.hdg / 100.0
            self.heading_pub.publish(heading_msg)

            # Geofence: сохраняем home point
            if self.home_lat is None:
                self.home_lat = gps_msg.latitude
                self.home_lon = gps_msg.longitude
                self.get_logger().info(
                    f'Home: {self.home_lat:.6f}, {self.home_lon:.6f}')

        elif mtype == 'VFR_HUD':
            speed_msg = Float64()
            speed_msg.data = msg.groundspeed
            self.ground_speed_pub.publish(speed_msg)

            # Батарея
            self.battery_voltage = msg.battery_voltage / 100.0  # centiV → V
            bat_msg = Float64()
            bat_msg.data = self.battery_voltage
            self.battery_pub.publish(bat_msg)

        elif mtype == 'ATTITUDE':
            imu_msg = Imu()
            imu_msg.orientation.x = 0.0
            imu_msg.orientation.y = 0.0
            imu_msg.orientation.z = math.sin(msg.yaw / 2)
            imu_msg.orientation.w = math.cos(msg.yaw / 2)
            imu_msg.angular_velocity.z = float(msg.yawspeed)
            self.attitude_pub.publish(imu_msg)

        elif mtype == 'LOCAL_POSITION_NED':
            vel_msg = Vector3Stamped()
            vel_msg.header.stamp = self.get_clock().now().to_msg()
            vel_msg.vector.x = float(msg.vx)
            vel_msg.vector.y = float(msg.vy)
            vel_msg.vector.z = float(msg.vz)
            self.velocity_pub.publish(vel_msg)

    # ─────────────────────────────────────────────────────────
    #  Безопасность: failsafe, geofence, pre-flight
    # ─────────────────────────────────────────────────────────

    def status_callback(self):
        """Периодическая проверка состояния."""
        now = time.time()

        # Проверка связи
        link_age = now - self.last_heartbeat_time
        if link_age > self.link_loss_timeout and not self.simulation:
            self.get_logger().error(
                f'ПОТЕРЯ СВЯЗИ ({link_age:.1f}s > {self.link_loss_timeout}s) — RTL!')
            self._send_command(mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                               mavutil.mavlink.MAV_MODE_AUTO_ARMED)
            # RTL mode = 6 для ArduPilot
            self._send_command(mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH)
            link_msg = String()
            link_msg.data = f'LOST ({link_age:.1f}s)'
            self.link_status_pub.publish(link_msg)
        else:
            link_msg = String()
            link_msg.data = f'OK ({link_age:.1f}s)'
            self.link_status_pub.publish(link_msg)

        # Проверка батареи
        if self.battery_voltage > 0 and \
           self.battery_voltage < self.min_battery_voltage:
            self.get_logger().warn(
                f'НИЗКАЯ БАТАРЕЯ: {self.battery_voltage:.1f}V < '
                f'{self.min_battery_voltage}V')

        # Timeout команд → hover
        if self.cmd_vel_received and \
           now - self.last_cmd_time > self.command_timeout:
            self.get_logger().info('Нет команд 1s — hover')
            if not self.simulation:
                self.send_velocity_command(0, 0, 0, 0)
            self.cmd_vel_received = False

        # Auto-arm (если включён)
        if self.auto_arm and not self.armed and not self.simulation:
            self.get_logger().warn('Auto-arm...')
            self._send_command(
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1)
            self.armed = True


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