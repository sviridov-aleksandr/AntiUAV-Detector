#!/usr/bin/env python3
"""
spectrum_analyzer.py - Spectrum analyzer for WiFi channel detection
Uses GNU Radio + USRP B200 to scan 2.4 GHz band and detect WiFi channels.

Usage:
    python spectrum_analyzer.py --center-freq 2437 --bw 20 --gain 30
    python spectrum_analyzer.py --scan 2400-2480 --step 5
"""

import argparse
import numpy as np
from gnuradio import blocks, filter, gru, blocks as blk
from gnuradio import analog, blocks, fft, gr, eng_notation
from gnuradio.eng_arg import eng_float_int_str
import sys
import time

try:
    from gnuradio import uhd
except ImportError:
    print("ERROR: GNU Radio UHD block not found. Install gr-uhd.")
    sys.exit(1)


class SpectrumAnalyzer(gr.top_block):
    """Spectrum analyzer for WiFi channel detection."""
    
    def __init__(self, center_freq, sample_rate, gain, fft_size=4096):
        gr.top_block.__init__(self, "WiFi Spectrum Analyzer")
        
        self.center_freq = center_freq
        self.sample_rate = sample_rate
        self.gain = gain
        self.fft_size = fft_size
        
        # USRP source
        self.usrp = uhd.usrp_source(device_args="",
                                     io_type=uhd.io_type.COMPLEX_FLOAT32,
                                     num_channels=0)
        self.usrp.set_samp_rate(sample_rate)
        self.usrp.set_center_freq(center_freq * 1e6, 0)
        self.usrp.set_gain(gain, 0)
        self.usrp.set_bandwidth(sample_rate, 0)
        
        # Decimation to reduce data rate for display
        self.decim = 100  # Decimation factor
        self.throttle = blocks.throttle(gr.sizeof_gr_complex * self.decim, 
                                         sample_rate // self.decim)
        
        # FFT block
        self.fft = fft.fft_vcc(fft_size, True, [], False)
        
        # Power calculation
        self.power = blocks.complex_to_mag_squared(gr.sizeof_gr_complex)
        
        # Average for smoother display
        self.avg = blocksMovingAverage(fft_size, 10)
        
        # Vector to file for MATLAB processing
        self.vector_sink = blocks.vector_sink_f(fft_size)
        
        # Connect
        self.connect(self.usrp, self.throttle, self.fft, self.power, 
                     self.vector_sink)
    
    def get_spectrum(self, duration_sec=5):
        """Capture spectrum for given duration and return frequency data."""
        self.start()
        time.sleep(duration_sec)
        self.stop()
        
        spectrum = self.vector_sink.data()
        freqs = np.fft.fftfreq(self.fft_size, 1.0 / self.sample_rate) * 1e6
        
        return freqs, spectrum


class MovingAverage(blocks.block):
    """Simple moving average filter."""
    
    def __init__(self, fft_size, window_size):
        blocks.block.__init__(self, "moving_avg",
                              gr.io_signature(1, 1, gr.sizeof_float * fft_size),
                              gr.io_signature(1, 1, gr.sizeof_float * fft_size))
        self.window_size = window_size
        self.buffer = np.zeros(fft_size)
    
    def work(self, input_items, output_items):
        in0 = input_items[0]
        out = output_items[0]
        
        # Simple moving average
        for i in range(len(in0)):
            self.buffer[i] = in0[i]
        
        # Apply moving average
        for i in range(len(out)):
            start = max(0, i - self.window_size // 2)
            end = min(len(self.buffer), i + self.window_size // 2 + 1)
            out[i] = np.mean(self.buffer[start:end])
        
        return len(out)


def scan_channels(center_freq, bandwidth, step=5):
    """Scan WiFi channels around center frequency."""
    print(f"\n{'='*60}")
    print(f"  Сканирование WiFi каналов")
    print(f"  Центр: {center_freq} MHz, Полоса: {bandwidth} MHz")
    print(f"{'='*60}")
    
    # WiFi channels in 2.4 GHz:
    # Channel 1:  2412 MHz
    # Channel 6:  2437 MHz
    # Channel 11: 2462 MHz
    # Channel 36: 5180 MHz (5 GHz)
    
    channels = {
        1:  2412,
        2:  2417,
        3:  2422,
        4:  2427,
        5:  2432,
        6:  2437,
        7:  2442,
        8:  2447,
        9:  2452,
        10: 2457,
        11: 2462,
        12: 2467,
        13: 2472,
        14: 2484,
    }
    
    print(f"\n{'Канал':>8} {'Частота (MHz)':>15} {'Мощность (dB)':>15}")
    print("-" * 42)
    
    detected = []
    
    for ch_num, ch_freq in channels.items():
        if abs(ch_freq - center_freq) <= bandwidth / 2:
            # Quick power measurement
            try:
                from gnuradio import uhd
                usrp = uhd.usrp_source("", 
                                       uhd.io_type.COMPLEX_FLOAT32, 0)
                usrp.set_samp_rate(1e6)  # Low rate for scanning
                usrp.set_center_freq(ch_freq * 1e6, 0)
                usrp.set_gain(30, 0)
                usrp.set_bandwidth(500e3, 0)
                
                # Capture short burst
                num_samples = int(1e6)  # 1 second at 1 Msps
                samples = usrp.read(num_samples)
                
                if len(samples) > 0:
                    power_db = 10 * np.log10(np.mean(np.abs(samples)**2) + 1e-10)
                    detected.append((ch_num, ch_freq, power_db))
                    print(f"  {ch_num:>8} {ch_freq:>15.1f} {power_db:>15.1f}")
                
                del usrp
                
            except Exception as e:
                print(f"  {ch_num:>8} {ch_freq:>15.1f}  ERROR: {e}")
    
    if detected:
        # Find strongest channel
        strongest = max(detected, key=lambda x: x[2])
        print(f"\n{'='*42}")
        print(f"  Сильнейший сигнал: канал {strongest[0]} "
              f"({strongest[1]} MHz, {strongest[2]:.1f} dB)")
        print(f"{'='*42}")
        
        return strongest[0], strongest[1]
    
    return None, None


def main():
    parser = argparse.ArgumentParser(description="WiFi Spectrum Analyzer")
    parser.add_argument("--center-freq", type=float, default=2437,
                        help="Center frequency in MHz (default: 2437)")
    parser.add_argument("--bw", type=float, default=20,
                        help="Bandwidth in MHz (default: 20)")
    parser.add_argument("--gain", type=int, default=30,
                        help="USRP gain in dB (default: 30)")
    parser.add_argument("--scan", type=str, default=None,
                        help="Scan range: min_freq-max_freq in MHz")
    parser.add_argument("--step", type=float, default=5,
                        help="Scan step in MHz (default: 5)")
    parser.add_argument("--duration", type=float, default=3,
                        help="Capture duration in seconds (default: 3)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save spectrum to file")
    
    args = parser.parse_args()
    
    if args.scan:
        # Scan mode
        parts = args.scan.split("-")
        min_freq = float(parts[0])
        max_freq = float(parts[1])
        
        print(f"Сканирование диапазона {min_freq}-{max_freq} MHz")
        
        for freq in np.arange(min_freq, max_freq, args.step):
            scan_channels(freq, args.step, args.step)
            time.sleep(0.5)
    
    else:
        # Single channel mode
        print(f"Анализатор спектра WiFi")
        print(f"Частота: {args.center_freq} MHz")
        print(f"Полоса: {args.bw} MHz")
        print(f"Усиление: {args.gain} dB\n")
        
        # Try to detect WiFi channel
        ch_num, ch_freq = scan_channels(args.center_freq, args.bw * 1e6, args.step)
        
        if ch_num:
            print(f"\nРекомендуемая конфигурация:")
            print(f"  center_freq = {ch_freq}e6  # Hz")
            print(f"  sample_rate = 20e6        # 20 MHz BW")
            print(f"  channel = {ch_num}")
            
            if args.output:
                print(f"\nСохранение данных в {args.output}...")
                # Save detected channel info
                with open(args.output, 'w') as f:
                    f.write(f"detected_channel={ch_num}\n")
                    f.write(f"detected_freq={ch_freq}\n")
                    f.write(f"center_freq={ch_freq}e6\n")
                    f.write(f"sample_rate=20e6\n")
                    f.write(f"gain={args.gain}\n")
                print("Готово.")


if __name__ == "__main__":
    main()