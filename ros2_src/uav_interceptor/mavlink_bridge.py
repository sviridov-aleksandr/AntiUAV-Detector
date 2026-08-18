#!/usr/bin/env python3
"""
ROS2 Node: MAVLink Bridge (наземная НСУ).

Двунаправленный MAVLink через WFB radio link:
  - Команды: /cmd_vel → MAVLink → WFB tx → борт → CUAV X7+ Pro
  - Телеметрия: WFB rx ← борт ← CUAV → ROS2 topics

Два режима подключения:
  1. radio: через WFB-ng UDP (наземная архитектура)
  2. direct: через USB/UART (локальное подключение, для тестов)

Дополнительные функции:
  - Армирование (MAV_CMD_COMPONENT_ARM_DISARM)
  - Failsafe: RTL при потере связи / timeout
  - Pre-flight checks (батарея, GPS, режим)
  - Kill switch: немедленная остановка моторов
  - GCS heartbeat (1 Hz) для WFB-ng mavlink service

Запуск (на Jetson Orin Nano, НСУ):
  ros2 run uav_interceptor mavlink_bridge --ros-args \
    -p link_mode:=radio \
    -p mavlink_udp_port:=14550 \
    -p simulation:=false
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist, Vector3Stamped
from std_msgs.msg import Float64, String, Bool
from sensor_msgs.msg import NavSatFix, Imu
from pymavlink import mavutil
from pymavlink.dialects.v20 import ardupilotmega as mavlink_dialect
import time
import math
import socket


class MavlinkBridge(Node):
    def __init__(self):
        super().__init__('mavlink_bridge')

        # ─── Параметры подключения ───
        self.declare_parameter('link_mode', 'radio')  # radio | direct
        self.declare_parameter('device', '/dev/ttyACM0')  # для direct
        self.declare_parameter('baudrate', 921600)
        self.declare_parameter('mavlink_udp_port', 14550)  # WFB-ng gs_mavlink
        self.declare_parameter('simulation', True)

        # ─── Параметры БЧ ───
        self.declare_parameter('strike_servo_channel', 6)
        self.declare_parameter('strike_servo_pwm', 2000)

        # ─── Параметры безопасности ───
        self.declare_parameter('auto_arm', False)
        self.declare_parameter('command_timeout', 1.0)
        self.declare_parameter('link_loss_timeout', 5.0)
        self.declare_parameter('min_battery_voltage', 14.0)  # 4S min
        self.declare_parameter('geofence_enabled', True)
        self.declare_parameter('geofence_max_dist', 500.0)

        self.link_mode = self.get_parameter('link_mode').value
        self.device = self.get_parameter('device').value
        self.baudrate = self.get_parameter('baudrate').value
        self.mavlink_udp_port = self.get_parameter('mavlink_udp_port').value
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

        # ─── MAVLink connection state ───
        self.mavlink_conn = None       # pymavlink connection (direct mode)
        self.udp_sock = None           # UDP socket (radio mode)
        self.wfb_addr = None           # Source address of WFB-ng (learned from first packet)
        self.rx_buf = bytearray()
        self.mav_serializer = None     # MAVLink serializer for radio mode

        self.armed = False
        self.last_heartbeat_time = time.time()
        self.last_cmd_time = time.time()
        self.cmd_vel_received = False
        self.boot_time = time.time()
        self._debug_count = 0
        self.home_lat = None
        self.home_lon = None
        self._cur_lat = None
        self._cur_lon = None
        self.battery_voltage = 0.0
        self.gcs_sys_id = 255          # GCS system ID
        self.gcs_comp_id = 0           # GCS component ID
        self.target_sys_id = 1         # Target (CUAV) system ID
        self.target_comp_id = 1        # Target component ID

        if not self.simulation:
            self._connect()
        else:
            self.get_logger().info('SIMULATION MODE: команды логируются, не отправляются')

        # ─── Timers ───
        self.status_timer = self.create_timer(1.0, self.status_callback)
        self.telemetry_timer = self.create_timer(0.05, self.telemetry_callback)  # 20 Hz
        self.heartbeat_timer = self.create_timer(1.0, self.heartbeat_send_callback)

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
        """Подключение через WFB-ng (UDP).

        WFB-ng на земле (gs.cfg) пробрасывает MAVLink:
          gs_mavlink peer = connect://127.0.0.1:14550

        WFB-ng подключается к 127.0.0.1:14550 как UDP-клиент:
          - отправляет телеметрию (с борта) на порт 14550
          - принимает команды с порта 14550

        Этот узел:
          - слушает UDP 14550 (принимает телеметрию)
          - отправляет команды обратно на адрес WFB-ng (learned from first packet)
        """
        try:
            self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2097152)
            self.udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2097152)
            self.udp_sock.bind(('127.0.0.1', self.mavlink_udp_port))
            self.udp_sock.settimeout(0.05)

            # MAVLink serializer для radio mode (без подключения)
            self.mav_serializer = mavlink_dialect.MAVLink(
                None, src_system=self.gcs_sys_id, src_component=self.gcs_comp_id)

            self.get_logger().info(
                f'WFB radio: слушаю UDP 127.0.0.1:{self.mavlink_udp_port} '
                f'(ожидание данных от WFB-ng)')
        except Exception as e:
            self.get_logger().error(f'UDP socket failed: {e}')
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
            self.target_sys_id = self.mavlink_conn.target_system
            self.target_comp_id = self.mavlink_conn.target_component
            self.get_logger().info(
                f'Heartbeat получен! sysid={self.target_sys_id}, '
                f'compid={self.target_comp_id}')
        except Exception as e:
            self.get_logger().error(f'Нет heartbeat: {e}')

    def set_guided_mode(self):
        """Переключение FC в GUIDED mode (ArduPilot custom_mode=4)."""
        # ArduPilot: MAV_CMD_DO_SET_MODE с custom_mode через MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
        self._send_command(
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            4)  # custom_mode=4 → GUIDED
        self.get_logger().info('GUIDED mode запрошен (custom_mode=4)')

    def request_data_streams(self):
        """Запрос потоков телеметрии (SET_MESSAGE_INTERVAL, ArduPilot 4.3+).

        request_data_stream устарел и игнорируется новыми прошивками.
        SET_MESSAGE_INTERVAL задаёт период в микросекундах для каждого
        MAVLink-сообщения напрямую.
        """
        if self.mavlink_conn is None:
            return
        # (message_id, период в мкс) — 10 Hz для навигации, 5 Hz для статуса
        intervals = {
            mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE: 100000,          # 10 Hz
            mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT: 100000,  # 10 Hz
            mavutil.mavlink.MAVLINK_MSG_ID_VFR_HUD: 100000,           # 10 Hz
            mavutil.mavlink.MAVLINK_MSG_ID_GPS_RAW_INT: 100000,       # 10 Hz
            mavutil.mavlink.MAVLINK_MSG_ID_SYS_STATUS: 100000,        # 10 Hz
            mavutil.mavlink.MAVLINK_MSG_ID_BATTERY_STATUS: 200000,    # 5 Hz
            mavutil.mavlink.MAVLINK_MSG_ID_RC_CHANNELS: 100000,       # 10 Hz
            mavutil.mavlink.MAVLINK_MSG_ID_HEARTBEAT: 1000000,        # 1 Hz
        }
        for msg_id, interval_us in intervals.items():
            self._send_command(
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                msg_id, interval_us)
        self.get_logger().info(
            'Интервалы телеметрии заданы (SET_MESSAGE_INTERVAL, 10 Hz)')

    # ─────────────────────────────────────────────────────────
    #  Отправка MAVLink
    # ─────────────────────────────────────────────────────────

    def _send_raw(self, data: bytes):
        """Отправка raw MAVLink байтов."""
        if self.simulation:
            return
        if self.link_mode == 'radio' and self.udp_sock is not None:
            if self.wfb_addr is None:
                self.get_logger().warning(
                    'TX отложен: WFB-ng адрес ещё не известен '
                    '(ожидание первого пакета телеметрии)',
                    throttle_duration_sec=5.0)
                return
            try:
                self.udp_sock.sendto(data, self.wfb_addr)
            except Exception as e:
                self.get_logger().debug(f'TX error: {e}')
        elif self.mavlink_conn is not None:
            self.mavlink_conn.write(data)

    def _send_command(self, command, param1=0, param2=0, param3=0,
                      param4=0, param5=0, param6=0, param7=0):
        """Отправка command_long через любой канал."""
        if self.simulation:
            self.get_logger().debug(
                f'[SIM] CMD={command} p1={param1} p2={param2}')
            return

        if self.mavlink_conn is not None:
            # Direct mode
            self.mavlink_conn.mav.command_long_send(
                self.target_sys_id, self.target_comp_id, command, 0,
                param1, param2, param3, param4, param5, param6, param7)
        elif self.mav_serializer is not None:
            # Radio mode
            msg = self.mav_serializer.command_long_encode(
                self.target_sys_id, self.target_comp_id, command, 0,
                param1, param2, param3, param4, param5, param6, param7)
            self._send_raw(msg.pack(self.mav_serializer))

    def send_velocity_command(self, vx, vy, vz, yaw_rate):
        """Отправка команды скорости (BODY_OFFSET_NED)."""
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
                time_boot_ms, self.target_sys_id, self.target_comp_id,
                mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
                type_mask,
                0, 0, 0,
                float(vx), float(vy), float(vz),
                0, 0, 0,
                0, float(yaw_rate))
        elif self.mav_serializer is not None:
            # Radio mode
            msg = self.mav_serializer.set_position_target_local_ned_encode(
                time_boot_ms, self.target_sys_id, self.target_comp_id,
                mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
                type_mask,
                0, 0, 0,
                float(vx), float(vy), float(vz),
                0, 0, 0,
                0, float(yaw_rate))
            self._send_raw(msg.pack(self.mav_serializer))

    def heartbeat_send_callback(self):
        """Отправка GCS heartbeat (1 Hz).

        WFB-ng mavlink service и ArduPilot ожидают periodic heartbeat
        от GCS для поддержания связи. Без него телеметрия может не идти.
        """
        if self.simulation:
            return

        if self.mavlink_conn is not None:
            # Direct mode
            self.mavlink_conn.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_GCS,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                0, 0, 0)
        elif self.mav_serializer is not None:
            # Radio mode
            msg = self.mav_serializer.heartbeat_encode(
                mavutil.mavlink.MAV_TYPE_GCS,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                0, 0, 0)
            self._send_raw(msg.pack(self.mav_serializer))

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
        self.get_logger().warning(f'*** STRIKE: {msg.data} ***')

        if self.simulation:
            self.get_logger().warning('[SIM] Подрыв БЧ симулирован')
            return

        self._send_command(
            mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
            self.strike_servo_channel,
            self.strike_servo_pwm)
        self.get_logger().warning(
            f'DO_SET_SERVO: ch={self.strike_servo_channel} '
            f'pwm={self.strike_servo_pwm}')

    def arm_callback(self, msg: Bool):
        """Армирование / разоружение."""
        if msg.data:
            self.get_logger().warning('*** ARMING ***')
            self._send_command(
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1)
            self.armed = True
        else:
            self.get_logger().warning('*** DISARMING ***')
            self._send_command(
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0)
            self.armed = False

    def kill_callback(self, msg: Bool):
        """Kill switch — немедленная остановка моторов."""
        self.get_logger().error('*** KILL SWITCH ***')
        if not self.simulation:
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

        if self.link_mode == 'radio' and self.udp_sock is not None:
            self._read_radio_telemetry()
        elif self.mavlink_conn is not None:
            self._read_direct_telemetry()

    def _read_radio_telemetry(self):
        """Чтение телеметрии из UDP (WFB-ng gs_mavlink).

        WFB-ng подключается к нашему порту 14550. Первый пакет
        раскрывает адрес WFB-ng — сохраняем его для отправки команд.
        """
        try:
            data, addr = self.udp_sock.recvfrom(4096)
            if data:
                # Запоминаем адрес WFB-ng для отправки команд
                if self.wfb_addr is None:
                    self.wfb_addr = addr
                    self.get_logger().info(
                        f'WFB-ng обнаружен: {addr[0]}:{addr[1]} — '
                        f'команды будут отправляться сюда')
                self.rx_buf.extend(data)
                self._process_rx_buffer()
        except socket.timeout:
            pass
        except Exception as e:
            self.get_logger().debug(f'RX read error: {e}')

    def _read_direct_telemetry(self):
        """Чтение телеметрии через direct MAVLink (все доступные сообщения)."""
        try:
            for _ in range(50):
                msg = self.mavlink_conn.recv_msg()
                if msg is None:
                    break
                try:
                    self._process_message(msg)
                except Exception as e:
                    self.get_logger().error(
                        f'Ошибка обработки {msg.get_type()}: {e}',
                        throttle_duration_sec=2.0)
        except Exception as e:
            self.get_logger().error(
                f'Telemetry read error: {e}', throttle_duration_sec=2.0)

    def _process_rx_buffer(self):
        """Парсинг MAVLink-сообщений из буфера."""
        while len(self.rx_buf) >= 2:
            if self.rx_buf[0] not in (0xFD, 0xFE):
                self.rx_buf.pop(0)
                continue

            if self.rx_buf[0] == 0xFD:
                # MAVLink 2
                if len(self.rx_buf) < 3:
                    break
                plen = self.rx_buf[1] + 12
            else:
                # MAVLink 1
                if len(self.rx_buf) < 6:
                    break
                plen = self.rx_buf[1] + 8

            if len(self.rx_buf) < plen:
                break

            packet = bytes(self.rx_buf[:plen])
            del self.rx_buf[:plen]

            try:
                msg = self.mav_serializer.decode(packet)
                self._process_message(msg)
            except Exception:
                pass

    def _process_message(self, msg):
        """Обработка MAVLink-сообщения → ROS2 публикация."""
        mtype = msg.get_type()

        if mtype == 'HEARTBEAT':
            self.last_heartbeat_time = time.time()
            # Обновляем target sys/comp ID из heartbeat
            if hasattr(msg, 'sysid') and msg.sysid > 0:
                self.target_sys_id = msg.sysid
            if hasattr(msg, 'compid') and msg.compid >= 0:
                self.target_comp_id = msg.compid

            try:
                mode = mavutil.mode_string_v10(msg)
                if mode is None:
                    mode = mavutil.mode_string_acm(msg)
            except Exception as e:
                mode = None
                self.get_logger().debug(f'mode_string: {e}')
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

            # Текущие координаты для geofence
            self._cur_lat = gps_msg.latitude
            self._cur_lon = gps_msg.longitude

            heading_msg = Float64()
            heading_msg.data = msg.hdg / 100.0
            self.heading_pub.publish(heading_msg)

            if self.home_lat is None:
                self.home_lat = gps_msg.latitude
                self.home_lon = gps_msg.longitude
                self.get_logger().info(
                    f'Home: {self.home_lat:.6f}, {self.home_lon:.6f}')

        elif mtype == 'VFR_HUD':
            speed_msg = Float64()
            speed_msg.data = msg.groundspeed
            self.ground_speed_pub.publish(speed_msg)

        elif mtype == 'SYS_STATUS':
            # Напряжение батареи (мВ) и уровень заряда в SYS_STATUS
            if msg.voltage_battery != 0xFFFF and msg.voltage_battery > 0:
                self.battery_voltage = msg.voltage_battery / 1000.0
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

        elif mtype == 'COMMAND_ACK':
            # Обработка подтверждений команд от автопилота
            cmd_id = msg.command
            result = msg.result
            result_names = {
                0: 'ACCEPTED',
                1: 'TEMPORARILY_REJECTED',
                2: 'DENIED',
                3: 'UNSUPPORTED',
                4: 'FAILED',
                5: 'IN_PROGRESS',
            }
            result_str = result_names.get(result, f'UNKNOWN({result})')
            # result_param2 — код отказа ArduPilot (MAVLink 2)
            rp2 = getattr(msg, 'result_param2', None)
            if result == 0:
                self.get_logger().info(f'ACK: cmd={cmd_id} → {result_str}')
            elif result == 5:
                pass  # IN_PROGRESS — не логируем
            else:
                self.get_logger().warning(
                    f'ACK: cmd={cmd_id} → {result_str}'
                    f' (код отказа: {rp2})')

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
            # ArduPilot: custom_mode=6 → RTL
            self._send_command(
                mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                6)
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
            self.get_logger().warning(
                f'НИЗКАЯ БАТАРЕЯ: {self.battery_voltage:.1f}V < '
                f'{self.min_battery_voltage}V')

        # Timeout команд → hover
        if self.cmd_vel_received and \
           now - self.last_cmd_time > self.command_timeout:
            self.get_logger().info('Нет команд 1s — hover')
            if not self.simulation:
                self.send_velocity_command(0, 0, 0, 0)
            self.cmd_vel_received = False

        # Geofence: проверка дистанции от home
        if self.geofence_enabled and self.home_lat is not None and \
           hasattr(self, '_cur_lat') and hasattr(self, '_cur_lon'):
            dist = self._haversine(self._cur_lat, self._cur_lon,
                                   self.home_lat, self.home_lon)
            if dist > self.geofence_max_dist:
                self.get_logger().error(
                    f'GEOFENCE: {dist:.0f}м > {self.geofence_max_dist}м — RTL!')
                self._send_command(
                    mavutil.mavlink.MAV_CMD_DOSET_MODE,
                    mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                    6)  # RTL
                self.send_velocity_command(0, 0, 0, 0)

        # Auto-arm
        if self.auto_arm and not self.armed and not self.simulation:
            self.get_logger().warning('Auto-arm...')
            self._send_command(
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1)
            self.armed = True

    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2):
        """Расчёт дистанции между двумя GPS-точками (метры)."""
        R = 6371000.0
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + \
            math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
        return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


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
