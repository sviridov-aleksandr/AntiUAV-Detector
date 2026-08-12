# WiFi Broadcast Receiver for OpenIPC FPV Drone

Приёмник WiFi broadcast видеопотока с FPV дрона (OpenIPC камера) на USRP B200.

## Архитектура

```
[OpenIPC Camera] --WiFi(802.11n)--> [USRP B200] --> [GNU Radio] --> [MATLAB PHY] --> [UDP Extractor] --> [Video Player]
                                                                                              ↓
                                                                                        [FPGA Xilinx - PHY]
```

## Компоненты

### 1. MATLAB (matlab/)
- Прототип PHY-стека 802.11n
- Синхронизация (timing, frequency)
- OFDM демодуляция
- Декодирование LTF/STF
- Валидация на захваченных данных

### 2. GNU Radio (gnuradio/)
- Spectrum analyzer для определения канала
- USRP B200 sink/source
- Захват сырых I/Q данных
- gr-ieee802-11 для полного стека

### 3. Python (python/)
- Извлечение UDP/Multicast видеопотока
- RTP депакинг
- Сохранение H.264 в файл
- Визуализация в реальном времени

### 4. FPGA (fpga/)
- VHDL модули PHY для Xilinx
- OFDM Demodulator (FFT-based)
- Timing/Frequency synchronizer
- AGC

## Требования

- USRP B200/B200mini + UHD drivers
- MATLAB + Communications Toolbox + Phased Array System Toolbox
- GNU Radio 3.8+
- Python 3.8+ (scapy, numpy, pyrtlsdr)
- Xilinx Vitis/Vivado (для FPGA)

## Быстрый старт

### Шаг 1: Анализ спектра
```bash
cd gnuradio
python spectrum_analyzer.py --center-freq 2437 --bw 20e6
```

### Шаг 2: Захват I/Q данных
```bash
cd gnuradio
python capture_samples.py --center-freq 2437 --rate 20e6 --duration 10 --file captured_samples.dat
```

### Шаг 3: MATLAB прототип
```matlab
cd matlab
run_main.m  % Запуск полного конвейера
```

### Шаг 4: Извлечение видео
```bash
cd python
python udp_video_extractor.py --port 5000 --output stream.h264
```

## Стандарты
- 802.11n (2.4 GHz, 20 MHz channel)
- OFDM: 64 subcarriers, 4 pilot, 52 data
- CP length: 0.8 us (80 samples @ 20 MHz)
- Modulation: BPSK/QPSK/16QAM/64QAM (CBR)
