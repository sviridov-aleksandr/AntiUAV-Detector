#!/usr/bin/env python3
"""
capture_samples.py - Capture raw I/Q samples from USRP B200
Saves complex baseband samples for MATLAB processing.

Usage:
    python capture_samples.py --center-freq 2437 --rate 20 --duration 5 --file captured.dat
    python capture_samples.py --continuous --port 9999  # Stream to network
"""

import argparse
import numpy as np
import sys
import time
import struct
import socket

try:
    from gnuradio import uhd, blocks, analog, gr
except ImportError:
    print("ERROR: GNU Radio not found. Install gnuradio + gr-uhd.")
    sys.exit(1)


class USRPCapture(gr.top_block):
    """Capture I/Q samples from USRP B200 and save to file."""
    
    def __init__(self, center_freq, sample_rate, gain, duration, output_file):
        gr.top_block.__init__(self, "USRP B200 Capture")
        
        self.center_freq = center_freq
        self.sample_rate = sample_rate
        self.gain = gain
        self.duration = duration
        self.output_file = output_file
        
        # USRP B200 source
        self.usrp = uhd.usrp_source(
            device_args="",
            io_type=uhd.io_type.COMPLEX_FLOAT32,
            num_channels=0
        )
        
        # Configure USRP
        self.usrp.set_samp_rate(sample_rate)
        self.usrp.set_center_freq(center_freq * 1e6, 0)
        self.usrp.set_gain(gain, 0)
        self.usrp.set_bandwidth(sample_rate * 0.8, 0)
        
        # Verify USRP settings
        print(f"\nUSRP B200 Configuration:")
        print(f"  Sample rate:    {sample_rate:.1f} Msps")
        print(f"  Center freq:    {center_freq:.1f} MHz")
        print(f"  Gain:           {gain} dB")
        print(f"  Bandwidth:      {sample_rate * 0.8 / 1e6:.1f} MHz")
        print(f"  Duration:       {duration} sec")
        print(f"  Output file:    {output_file}")
        print(f"  Samples:        {int(sample_rate * duration)}\n")
        
        # Verify actual settings
        actual_rate = self.usrp.get_samp_rate()
        actual_freq = self.usrp.get_center_freq(0)
        print(f"  Actual rate:    {actual_rate:.1f} Msps")
        print(f"  Actual freq:    {actual_freq / 1e6:.1f} MHz\n")
        
        # Throttle to prevent buffer overflow
        self.throttle = blocks.throttle(
            gr.sizeof_gr_complex, 
            int(sample_rate)
        )
        
        # File sink
        self.file_sink = blocks.file_sink(
            gr.sizeof_gr_complex,  # complex float
            output_file,
            False  # Unbuffered
        )
        self.file_sink.set_unbuffered_on()
        
        # Connect
        self.connect(self.usrp, self.throttle, self.file_sink)
    
    def capture(self):
        """Execute capture and return statistics."""
        print("Starting capture... (press Ctrl+C to stop early)")
        
        self.start()
        
        try:
            # Wait for specified duration
            elapsed = 0
            while elapsed < self.duration:
                time.sleep(min(1, self.duration - elapsed))
                elapsed += 1
                remaining = self.duration - elapsed
                print(f"\r  Captured: {elapsed}/{self.duration} sec "
                      f"(remaining: {remaining:.0f}s)   ", end="", flush=True)
            
            self.stop()
            
        except KeyboardInterrupt:
            print("\n\nCapture interrupted by user.")
            self.stop()
        
        # Calculate statistics
        samples_captured = int(self.sample_rate * self.duration)
        file_size = os.path.getsize(self.output_file) if os.path.exists(self.output_file) else 0
        actual_samples = file_size // 8  # 8 bytes per complex float
        
        print(f"\n\nCapture complete:")
        print(f"  File size: {file_size / 1e6:.2f} MB")
        print(f"  Samples:   {actual_samples}")
        print(f"  Duration:  {actual_samples / self.sample_rate:.2f} sec")
        
        return actual_samples


def save_header(output_file, center_freq, sample_rate, gain):
    """Save metadata header alongside I/Q data."""
    header_file = output_file + '.hdr'
    with open(header_file, 'w') as f:
        f.write(f"# WiFi Broadcast Receiver - Capture Header\n")
        f.write(f"format: complex_float32\n")
        f.write(f"center_freq_hz: {center_freq * 1e6}\n")
        f.write(f"sample_rate_hz: {sample_rate}\n")
        f.write(f"gain_db: {gain}\n")
        f.write(f"device: USRP_B200\n")
        f.write(f"standard: 802.11n\n")
        f.write(f"bandwidth_mhz: {sample_rate / 1e6}\n")
        f.write(f"date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


def main():
    import os
    
    parser = argparse.ArgumentParser(description="Capture I/Q samples from USRP B200")
    parser.add_argument("--center-freq", type=float, default=2437,
                        help="Center frequency in MHz (default: 2437)")
    parser.add_argument("--rate", type=float, default=20,
                        help="Sample rate in Msps (default: 20)")
    parser.add_argument("--gain", type=int, default=30,
                        help="USRP LNA gain in dB (default: 30)")
    parser.add_argument("--duration", type=float, default=5,
                        help="Capture duration in seconds (default: 5)")
    parser.add_argument("--file", "-o", type=str, default="captured_samples.dat",
                        help="Output file name (default: captured_samples.dat)")
    parser.add_argument("--continuous", action="store_true",
                        help="Continuous capture mode")
    parser.add_argument("--port", type=int, default=9999,
                        help="Network port for streaming (default: 9999)")
    
    args = parser.parse_args()
    
    if args.continuous:
        print("Continuous capture mode - streaming to network port")
        start_network_streamer(args.port, args.center_freq, args.rate, args.gain)
    else:
        # Single capture
        tb = USRPCapture(args.center_freq, args.rate, args.gain, 
                        args.duration, args.file)
        tb.capture()
        save_header(args.file, args.center_freq, args.rate, args.gain)
        print(f"\nData saved to: {args.file}")
        print(f"Header saved to: {args.file}.hdr")
        print(f"\nTo process in MATLAB:")
        print(f"  load '{args.file}' -ascii  % or use fread for binary")
        print(f"  Or: python load_samples.py {args.file}")


def start_network_streamer(port, center_freq, sample_rate, gain):
    """Stream I/Q samples over UDP network."""
    print(f"\nStreaming I/Q data to UDP port {port}...")
    print(f"Receive with: nc -ul {port} > stream.dat\n")
    
    tb = gr.top_block()
    
    usrp = uhd.usrp_source("", 
                           uhd.io_type.COMPLEX_FLOAT32, 0)
    usrp.set_samp_rate(sample_rate)
    usrp.set_center_freq(center_freq * 1e6, 0)
    usrp.set_gain(gain, 0)
    usrp.set_bandwidth(sample_rate * 0.8, 0)
    
    # Network sink
    net_sink = blocks.network_sink_f(
        gr.sizeof_gr_complex,
        "",  # localhost
        port,
        1472,  # MTU
        0
    )
    
    tb.connect(usrp, net_sink)
    tb.start()
    
    try:
        while True:
            time.sleep(1)
            print(f"  Streaming... port {port}   ", end="\r", flush=True)
    except KeyboardInterrupt:
        print("\nStopping streamer.")
        tb.stop()


if __name__ == "__main__":
    main()
