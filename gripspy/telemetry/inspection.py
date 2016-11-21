"""
Module for inspecting telemetry files
"""
from __future__ import division, absolute_import, print_function

from io import open
from os.path import getsize

import numpy as np

from .generators import packet_generator
from .parsers import parse_packet_header


__all__ = ['inspect']


def inspect(filename):
    """Report top-level information about the contents of one or more telemetry files

    Parameters
    ----------
    filename : str or list of str
        One or more telemetry files
    """
    # If a list is provided as input, recurse on each entry of the list
    if type(filename) == list:
        for entry in filename:
            inspect(entry)
            print()
        return

    packet_count = np.zeros((256, 256), dtype=np.uint64)
    byte_count = np.zeros((256, 256), dtype=np.uint64)
    previous_packet_counter = np.full((256, 256), 0xFFFFFFFF, dtype=np.uint64)
    gap_count = np.zeros((256, 256), dtype=np.uint64)

    not_coincident = np.zeros(16, dtype=bool)

    systime_start = None

    print("Inspecting {0}".format(filename))

    total_bytes = getsize(filename)
    print("{0} bytes".format(total_bytes))

    with open(filename, 'rb') as f:
        pg = packet_generator(f, raise_exceptions=True)
        for p in pg:
            header = parse_packet_header(p)
            systemid = header['systemid']
            tmtype = header['tmtype']

            packet_count[systemid, tmtype] += 1
            byte_count[systemid, tmtype] += 16 + header['length']

            if systime_start is None:
                systime_start = header['systime']

            if previous_packet_counter[systemid, tmtype] != 0xFFFFFFFF:
                packet_counter_difference = header['counter'] - previous_packet_counter[systemid, tmtype]
                if packet_counter_difference > 1:
                    # The counter works slightly differently for quicklook-spectrum packets
                    if not (packet_counter_difference == 6 and systemid == 0x10 and (tmtype & 0xF0) == 0x10):
                        gap_count[systemid, tmtype] += 1

            if (systemid & 0xF0) == 0x80 and tmtype == 0xF3:
                buf = bytearray(p)
                if buf[19:22] == b'\0\0\0' or buf[22:25] == b'\0\0\0':
                    not_coincident[systemid & 0x0F] = True

            previous_packet_counter[systemid, header['tmtype']] = header['counter']

    systime_end = header['systime']

    print("Packet breakdown:")
    for systemid in range(256):
        for tmtype in range(256):
            if packet_count[systemid, tmtype] != 0:
                print("  0x{0:02x} 0x{1:02x} : [{2:5.1%}] {3}".format(systemid, tmtype, byte_count[systemid, tmtype] / total_bytes,
                                                                      packet_count[systemid, tmtype]) + \
                     (" (includes non-coincident data!)" if (systemid & 0xF0) == 0x80 and tmtype == 0xF3 and not_coincident[systemid & 0x0F] else "") + \
                     (", plus {0} gaps".format(gap_count[systemid, tmtype]) if gap_count[systemid, tmtype] > 0 else ""))

    print("Elapsed gondola time: {0} ({1} minutes)".format(systime_end - systime_start, (systime_end - systime_start) * 1e-7 / 60))
