#!/usr/bin/env python3
"""
udp_video_extractor.py - Extract UDP/Multicast video stream from captured WiFi packets
Parses 802.11 frames, extracts IP packets, and demuxes UDP video stream.

Usage:
    python udp_video_extractor.py --port 5000 --output stream.h264
    python udp_video_extractor.py --multicast 239.1.1.1 --port 5600
    python udp_video_extractor.py --pcap captured.pcap --output stream.h264
"""

import argparse
import numpy as np
import struct
import socket
import sys
import time
import os
from collections import deque

try:
    from scapy.all import rdpcap, IP, UDP, Raw, Ether, conf
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("WARNING: scapy not available. Using raw socket mode.")
    print("Install: pip install scapy")


class UDPVideoExtractor:
    """Extract UDP video stream from network captures."""
    
    # H.264 NAL unit types
    NAL_START_CODE = b'\x00\x00\x00\x01'
    NAL_START_CODE_SHORT = b'\x00\x00\x01'
    
    # H.264 NAL unit types
    NAL_SLICE = 1
    NAL_DPA = 2
    NAL_DPB = 3
    NAL_DPC = 4
    NAL_IDR = 5
    NAL_SEI = 6
    NAL_SPS = 7
    NAL_PPS = 8
    NAL_AUD = 9
    
    def __init__(self, port=5000, multicast_ip=None, output_file='stream.h264',
                 pcap_file=None):
        self.port = port
        self.multicast_ip = multicast_ip
        self.output_file = output_file
        self.pcap_file = pcap_file
        self.packet_count = 0
        self.video_bytes = 0
        self.nal_counts = {}
        self.start_time = time.time()
        
        # RTP parsing
        self.rtp_seq = None
        self.rtp_timestamp = None
        
        # Statistics
        self.stats = {
            'total_packets': 0,
            'udp_packets': 0,
            'ip_packets': 0,
            'h264_nals': 0,
            'sps_frames': 0,
            'pps_frames': 0,
            'idr_frames': 0,
            'drop_packets': 0,
        }
    
    def extract_from_pcap(self):
        """Extract video from pcap file."""
        if not SCAPY_AVAILABLE:
            print("ERROR: scapy required for pcap processing.")
            sys.exit(1)
        
        if not os.path.exists(self.pcap_file):
            print(f"ERROR: PCAP file not found: {self.pcap_file}")
            sys.exit(1)
        
        print(f"\nProcessing PCAP file: {self.pcap_file}")
        print(f"Loading packets...")
        
        packets = rdpcap(self.pcap_file)
        print(f"Loaded {len(packets)} packets\n")
        
        # Open output file
        with open(self.output_file, 'wb') as f:
            for i, pkt in enumerate(packets):
                self.stats['total_packets'] += 1
                
                # Check for IP packet
                if pkt.haslayer(IP):
                    self.stats['ip_packets'] += 1
                    
                    ip_layer = pkt[IP]
                    
                    # Check for UDP
                    if ip_layer.proto == 17 and ip_layer.haslayer(UDP):
                        self.stats['udp_packets'] += 1
                        udp_layer = ip_layer[UDP]
                        
                        # Check port
                        if udp_layer.dport == self.port or udp_layer.sport == self.port:
                            payload = bytes(udp_layer.payload)
                            
                            if len(payload) > 0:
                                # Try to extract H.264 NAL units
                                self._write_h264_nals(payload, f)
                                
                                # RTP parsing
                                if len(payload) >= 12:
                                    self._parse_rtp(payload)
                                
                                self.packet_count += 1
                                self.video_bytes += len(payload)
                                
                                if self.packet_count % 1000 == 0:
                                    self._print_progress()
        
        self._print_final_stats()
    
    def extract_from_live(self):
        """Capture and extract video from live network traffic."""
        print(f"\nCapturing live traffic on port {self.port}...")
        
        if self.multicast_ip:
            print(f"Joining multicast group: {self.multicast_ip}")
            self._join_multicast(self.multicast_ip, self.port)
        
        # Create UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('', self.port))
        sock.settimeout(1.0)
        
        print(f"Listening on port {self.port}... (Ctrl+C to stop)\n")
        
        try:
            with open(self.output_file, 'wb') as f:
                while True:
                    try:
                        data, addr = sock.recvfrom(65535)
                        self.stats['total_packets'] += 1
                        
                        if len(data) > 0:
                            self._write_h264_nals(data, f)
                            self.packet_count += 1
                            self.video_bytes += len(data)
                            
                            if self.packet_count % 1000 == 0:
                                self._print_progress()
                    
                    except socket.timeout:
                        continue
                        
        except KeyboardInterrupt:
            print("\nStopping capture.")
        
        sock.close()
        self._print_final_stats()
    
    def _write_h264_nals(self, data, output_file):
        """Extract and write H.264 NAL units from data."""
        # Search for NAL start codes
        pos = 0
        while pos < len(data) - 4:
            # Check for 4-byte start code
            if data[pos:pos+4] == self.NAL_START_CODE:
                nal_start = pos
                nal_end = self._find_next_start_code(data, pos + 4)
                
                if nal_end > pos + 4:
                    nal_unit = data[nal_start:nal_end]
                    nal_type = nal_unit[4] & 0x1F if len(nal_unit) > 4 else 0
                    
                    output_file.write(nal_unit)
                    self.stats['h264_nals'] += 1
                    
                    # Count NAL types
                    if nal_type not in self.nal_counts:
                        self.nal_counts[nal_type] = 0
                    self.nal_counts[nal_type] += 1
                    
                    # Track specific NAL types
                    if nal_type == self.NAL_SPS:
                        self.stats['sps_frames'] += 1
                    elif nal_type == self.NAL_PPS:
                        self.stats['pps_frames'] += 1
                    elif nal_type == self.NAL_IDR:
                        self.stats['idr_frames'] += 1
                    
                    pos = nal_end
                else:
                    pos += 1
            # Check for 3-byte start code
            elif data[pos:pos+3] == self.NAL_START_CODE_SHORT:
                nal_start = pos
                nal_end = self._find_next_start_code(data, pos + 3)
                
                if nal_end > pos + 3:
                    nal_unit = data[nal_start:nal_end]
                    nal_type = nal_unit[3] & 0x1F if len(nal_unit) > 3 else 0
                    
                    output_file.write(nal_unit)
                    self.stats['h264_nals'] += 1
                    
                    if nal_type not in self.nal_counts:
                        self.nal_counts[nal_type] = 0
                    self.nal_counts[nal_type] += 1
                    
                    if nal_type == self.NAL_SPS:
                        self.stats['sps_frames'] += 1
                    elif nal_type == self.NAL_PPS:
                        self.stats['pps_frames'] += 1
                    elif nal_type == self.NAL_IDR:
                        self.stats['idr_frames'] += 1
                    
                    pos = nal_end
                else:
                    pos += 1
            else:
                pos += 1
    
    def _find_next_start_code(self, data, start):
        """Find the next NAL start code after position."""
        pos = start
        while pos < len(data) - 4:
            if data[pos:pos+4] == self.NAL_START_CODE:
                return pos
            if data[pos:pos+3] == self.NAL_START_CODE_SHORT:
                return pos
            pos += 1
        return len(data)
    
    def _parse_rtp(self, payload):
        """Parse RTP header from payload."""
        if len(payload) < 12:
            return
        
        # RTP header parsing
        version = (payload[0] >> 6) & 0x03
        if version != 2:
            return
        
        payload_type = payload[1] & 0x7F
        seq_num = struct.unpack('!H', payload[2:4])[0]
        timestamp = struct.unpack('!I', payload[4:8])[0]
        ssrc = struct.unpack('!I', payload[8:12])[0]
        
        # Detect sequence number gaps
        if self.rtp_seq is not None:
            expected = (self.rtp_seq + 1) & 0xFFFF
            if seq_num != expected:
                gap = (seq_num - self.rtp_seq) & 0xFFFF
                if gap > 1 and gap < 1000:
                    self.stats['drop_packets'] += gap - 1
        
        self.rtp_seq = seq_num
        self.rtp_timestamp = timestamp
    
    def _join_multicast(self, multicast_ip, port):
        """Join multicast group."""
        mreq = socket.inet_aton(multicast_ip) + socket.inet_aton('0.0.0.0')
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    
    def _print_progress(self):
        """Print capture progress."""
        elapsed = time.time() - self.start_time
        rate = self.video_bytes / elapsed / 1024 if elapsed > 0 else 0
        
        print(f"\r  Packets: {self.packet_count:>8,}  "
              f"Data: {self.video_bytes/1e6:.2f} MB  "
              f"Rate: {rate:.1f} KB/s  "
              f"NALs: {self.stats['h264_nals']:,}   ", 
              end="", flush=True)
    
    def _print_final_stats(self):
        """Print final statistics."""
        elapsed = time.time() - self.start_time
        
        print(f"\n{'='*50}")
        print(f"  Video Extraction Complete")
        print(f"{'='*50}")
        print(f"  Duration:       {elapsed:.1f} sec")
        print(f"  Total packets:  {self.stats['total_packets']:,}")
        print(f"  UDP packets:    {self.stats['udp_packets']:,}")
        print(f"  Video data:     {self.video_bytes/1e6:.2f} MB")
        print(f"  H.264 NALs:     {self.stats['h264_nals']:,}")
        print(f"  SPS frames:     {self.stats['sps_frames']}")
        print(f"  PPS frames:     {self.stats['pps_frames']}")
        print(f"  IDR frames:     {self.stats['idr_frames']}")
        print(f"  Dropped pkts:   {self.stats['drop_packets']}")
        print(f"  Output file:    {self.output_file}")
        print(f"{'='*50}")
        
        if self.nal_counts:
            print(f"\nNAL Unit Distribution:")
            nal_names = {
                1: 'Non-IDR Slice',
                5: 'IDR Slice',
                6: 'SEI',
                7: 'SPS',
                8: 'PPS',
                9: 'AUD',
            }
            for nal_type, count in sorted(self.nal_counts.items()):
                name = nal_names.get(nal_type, f'Unknown ({nal_type})')
                print(f"  {name:>20}: {count:>8,}")


def main():
    parser = argparse.ArgumentParser(description="Extract UDP Video Stream from WiFi")
    parser.add_argument("--port", "-p", type=int, default=5000,
                        help="UDP port to capture (default: 5000)")
    parser.add_argument("--output", "-o", type=str, default="stream.h264",
                        help="Output H.264 file (default: stream.h264)")
    parser.add_argument("--multicast", "-m", type=str, default=None,
                        help="Multicast IP address to join")
    parser.add_argument("--pcap", type=str, default=None,
                        help="Input PCAP file (instead of live capture)")
    parser.add_argument("--play", action="store_true",
                        help="Play video after extraction (ffplay)")
    
    args = parser.parse_args()
    
    extractor = UDPVideoExtractor(
        port=args.port,
        multicast_ip=args.multicast,
        output_file=args.output,
        pcap_file=args.pcap
    )
    
    if args.pcap:
        extractor.extract_from_pcap()
    else:
        extractor.extract_from_live()
    
    if args.play and os.path.exists(args.output):
        print(f"\nPlaying video with ffplay...")
        os.system(f"ffplay -i {args.output} -autoexit -nodisp 2>/dev/null &")


if __name__ == "__main__":
    main()
