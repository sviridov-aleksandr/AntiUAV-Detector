#!/usr/bin/env python3
"""Настройка TELEM2 (UART2) полётника CUAV X7+ Pro для WFB-ng.
Подключается к /dev/ttyACM1@460800 и устанавливает:
  SERIAL2_PROTOCOL = 2 (MAVLink2)
  SERIAL2_BAUD     = 921 (921600)
"""
import sys
import time

from pymavlink import mavutil

DEV = "/dev/ttyACM1"
BAUD = 460800

PARAMS = {
    "SERIAL2_PROTOCOL": 2,
    "SERIAL2_BAUD": 921,
}

def main():
    print(f"Connecting to {DEV} @ {BAUD}...")
    m = mavutil.mavlink_connection(DEV, baud=BAUD)
    # Ждём первый HEARTBEAT
    t0 = time.time()
    while time.time() - t0 < 15:
        msg = m.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
        if msg:
            print(f"Connected! HEARTBEAT from sysid={m.target_system}, compid={m.target_component}, type={msg.type}")
            break
    else:
        print("ERROR: no HEARTBEAT from flight controller")
        sys.exit(1)

    # Читаем текущие значения (чтобы не перезаписывать зря)
    for name in PARAMS:
        m.mav.param_request_read_send(m.target_system, m.target_component, name.encode(), -1)
        msg = m.recv_match(type="PARAM_VALUE", blocking=True, timeout=5)
        if msg:
            print(f"  {name} current = {msg.param_value}")

    # Устанавливаем параметры
    for name, val in PARAMS.items():
        print(f"Setting {name} = {val}...")
        m.mav.param_set_send(m.target_system, m.target_component,
                             name.encode(), float(val), mavutil.mavlink.MAV_PARAM_TYPE_INT32)
        time.sleep(1.0)
        # Подтверждение
        msg = m.recv_match(type="PARAM_VALUE", blocking=True, timeout=5)
        if msg:
            print(f"  -> confirmed {msg.param_id} = {msg.param_value}")

    print("Done. Reboot the flight controller to apply changes.")
    m.close()

if __name__ == "__main__":
    main()
