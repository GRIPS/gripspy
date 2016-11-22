"""
Module for parsing a telemetry packet
"""
from __future__ import division, absolute_import, print_function

import struct

import numpy as np


__all__ = ['parser', 'print_parsers']


INDEX_CHECKSUM = 2
INDEX_SYSTEMID = 4
INDEX_TMTYPE = 5
INDEX_LENGTH = 6
INDEX_COUNTER = 8
INDEX_SYSTIME = 10
INDEX_PAYLOAD = 16


# A 256x256 list of lists for what parsing function to call based on SystemID and TmType
# This registry is populated at the end of this module
parser_registry = []
for systemid in range(256):
    parser_registry.append([None] * 256)


def print_parsers():
    """
    Print the SystemID and TmType pairs that have a parsing function defined.
    The parsing function may not be fully implemented.
    """
    for systemid in range(256):
        for tmtype in range(256):
            if parser_registry[systemid][tmtype] is not None:
                print("0x{0:02x} 0x{1:02x} : {2}".format(systemid, tmtype, parser_registry[systemid][tmtype].func_name))


def parse_packet_header(packet):
    """
    Parse the header of a telemetry packet

    Parameters
    ----------
    packet : bytearray-like
        A full telemetry packet including the 16-byte header

    Returns
    -------
    out : dict
        The contents of the header of the telemetry packet
    """
    buf = bytearray(packet)

    header = {'systemid' : buf[INDEX_SYSTEMID],
              'tmtype'   : buf[INDEX_TMTYPE],
              'length'   : buf[INDEX_LENGTH]
                           | buf[INDEX_LENGTH + 1] << 8,
              'counter'  : buf[INDEX_COUNTER]
                           | buf[INDEX_COUNTER + 1] << 8,
              'systime'  : buf[INDEX_SYSTIME]
                           | buf[INDEX_SYSTIME + 1] << 8
                           | buf[INDEX_SYSTIME + 2] << 16
                           | buf[INDEX_SYSTIME + 3] << 24
                           | buf[INDEX_SYSTIME + 4] << 32
                           | buf[INDEX_SYSTIME + 5] << 40
             }

    return header


def parser(packet, filter_systemid=None, filter_tmtype=None):
    """
    Parse a telemetry packet for its contents

    Parameters
    ----------
    packet : bytearray-like
        A full telemetry packet including the 16-byte header
    filter_systemid : int
        If specified, only parse the packet if the SystemID matches
    filter_tmtype : int
        If specified, only parse the packet if the TmType matches

    Returns
    -------
    out : dict
        The contents of the telemetry packet
    """
    buf = bytearray(packet)

    header = parse_packet_header(buf)

    if filter_systemid is not None and filter_systemid != header['systemid']: return
    if filter_tmtype is not None and filter_tmtype != header['tmtype']: return

    function = parser_registry[header['systemid']][header['tmtype']]
    if function is not None:
        return function(buf, header)
    else:
        return header


def ge_event(buf, out):
    """Parse an event from a germanium detector (SystemID 0x8? and TmType 0xF3)
    
    Notes
    -----
    Assumes only one event in each event packet
    """
    if out['systemid'] & 0xF0 != 0x80 or out['tmtype'] != 0xF3:
        raise ValueError

    index = INDEX_PAYLOAD

    out['detector'] = out['systemid'] & 0x0F

    out['veto_lvgr'] = np.bool(buf[index] & 1)
    out['veto_hvgr'] = np.bool(buf[index] >> 1 & 1)
    out['veto_shield'] = np.bool(buf[index] >> 2 & 1)
    out['veto_multiple'] = np.bool(buf[index] >> 3 & 1)

    time_lsb = buf[index] >> 7 | buf[index + 1] << 1 | buf[index + 2] << 9
    time_diff = (out['systime'] & ((1 << 17) - 1)) - time_lsb
    if time_diff < 0:
        time_diff += 1 << 17
    out['event_time'] = (out['systime'] - time_diff) * 10 + (buf[index] >> 3 & 0x0E)

    index += 3

    num_cha = np.zeros(8, np.uint8)
    for side in range(2):
        num_cha[4 * side + 0] = buf[index + 0] & 0x3F
        num_cha[4 * side + 1] = buf[index + 0] >> 6 | (buf[index + 1] & 0x0F) << 2
        num_cha[4 * side + 2] = buf[index + 1] >> 4 | (buf[index + 2] & 0x03) << 4
        num_cha[4 * side + 3] = buf[index + 2] >> 2
        index += 3

    asiccha = np.zeros(512, np.uint16)
    has_conversion = np.zeros(512, np.bool)
    has_trigger = np.zeros(512, np.bool)
    adc = np.zeros(512, np.int16)
    has_glitch = np.zeros(512, np.bool)
    delta_time = np.zeros(512, np.int8)

    loc = 0

    for asic in range(8):
        for entry in range(num_cha[asic]):
            asiccha[loc] = (buf[index] & 0x3F) + asic * 64
            has_conversion[loc] = buf[index] >> 6 & 1
            has_trigger[loc] = buf[index] >> 7 & 1
            index += 1

            if has_conversion[loc]:
                adc[loc] = buf[index] | (buf[index + 1] & 0x7F) << 8
                has_glitch[loc] = buf[index + 1] >> 7
                index += 2

            if has_trigger[loc]:
                delta_time[loc] = buf[index]
                index += 1
            
            loc += 1

    out['asiccha'] = asiccha[:loc]
    out['has_conversion'] = has_conversion[:loc]
    out['has_trigger'] = has_trigger[:loc]
    out['adc'] = adc[:loc]
    out['has_glitch'] = has_glitch[:loc]
    out['delta_time'] = delta_time[:loc]

    return out

def ge_raw_event(buf, out):
    """Parse a raw event from a germanium detector (SystemID 0x8? and TmType 0xF1 or 0xF2)
    """
    if not (out['systemid'] & 0xF0 == 0x80 and (out['tmtype'] == 0xF1 or out['tmtype'] == 0xF2)):
        raise ValueError

    out['detector'] = out['systemid'] & 0x0F
    out['side'] = 'LV' if out['tmtype'] == 0xF1 else 'HV'

    out['veto_multiple'] = np.bool(out['systime'] >> 47 & 1)
    out['systime'] &= (1 << 48) - 1
    out['event_time'] = out['systime']

    adc = np.zeros(256, np.int16)
    has_glitch = np.zeros(256, np.bool)
    has_trigger = np.zeros(256, np.bool)
    trigger_time = np.zeros(256, np.uint16)

    index = INDEX_PAYLOAD

    for i in range(32):
        has_trigger[i*8:(i+1)*8] = np.array([buf[index] >> j & 1 for j in range(8)])
        index += 1

    for i in range(256):
        adc[i] = buf[index] | (buf[index+1] & 0x7F) << 8
        has_glitch[i] = buf[index+1] >> 7
        trigger_time[i] = buf[index+2] | buf[index+3] << 8
        index += 4

    out['adc'] = adc
    out['has_glitch'] = has_glitch
    out['has_trigger'] = has_trigger
    out['trigger_time'] = trigger_time

    return out


def bgo_event(buf, out):
    """Parse an event packet from the BGO shields (SystemID 0xB6 and TmType 0x80 or 0x82)

    Notes
    -----
    TmType 0x80 can have up to 64 events, while TmType 0x82 can have up to 102 events
    """
    if not (out['systemid'] == 0xB6 and (out['tmtype'] == 0x80 or out['tmtype'] == 0x82)):
        raise ValueError

    event_time = np.zeros(102, np.int64)
    clock_source = np.zeros(102, np.uint8)
    clock_synced = np.zeros(102, np.bool)
    channel = np.zeros(102, np.uint8)
    level = np.zeros(102, np.uint8)

    bytes_per_event = 5 if out['tmtype'] == 0x82 else 8

    index = INDEX_PAYLOAD
    loc = 0

    while index < INDEX_PAYLOAD + out['length']:
        time_lsb = buf[index] | buf[index + 1] << 8 | buf[index + 2] << 16 | (buf[index + 3] & 0x0F) << 24
        time_diff = (out['systime'] & ((1 << 28) - 1)) - time_lsb
        if time_diff < 0:
            time_diff += 1 << 28
        event_time[loc] = (out['systime'] - time_diff) * 10 + (buf[index + 3] >> 4) * 2

        clock_source[loc] = buf[index + 4] >> 7 & 1
        clock_synced[loc] = np.bool(buf[index + 4] >> 6 & 1)
        channel[loc] = buf[index + 4] >> 2 & 0x0F
        level[loc] = buf[index + 4] & 0x03

        index += bytes_per_event
        loc += 1

    out['event_time'] = event_time[:loc]
    out['clock_source'] = clock_source[:loc]
    out['clock_synced'] = clock_synced[:loc]
    out['channel'] = channel[:loc]
    out['level'] = level[:loc]

    return out


def bgo_counter(buf, out):
    """Parse a counters packet from the BGO shields (SystemID 0xB6 and TmType 0x81)
    """
    if not (out['systemid'] == 0xB6 and out['tmtype'] == 0x81):
        raise ValueError

    index = INDEX_PAYLOAD

    out['counter_time'] = buf[index] \
                          | buf[index + 1] << 8 \
                          | buf[index + 2] << 16 \
                          | buf[index + 3] << 24 \
                          | buf[index + 4] << 32 \
                          | buf[index + 5] << 40

    index += 6

    out['total_livetime'] = buf[index] \
                            | buf[index + 1] << 8 \
                            | buf[index + 2] << 16
    out['total_livetime'] /= 5.

    index += 3

    channel_livetime = np.zeros(12, np.int32)
    for i in range(12):
        channel_livetime[i] = buf[index] \
                              | buf[index + 1] << 8 \
                              | buf[index + 2] << 16
        index += 3
    out['channel_livetime'] = channel_livetime / 5.

    out['channel_count'] = np.fromstring(bytes(buf[index:index + 96]), dtype=np.uint16).reshape((12, 4))
    index += 96

    out['veto_count'] = buf[index] \
                        | buf[index + 1] << 8 \
                        | buf[index + 2] << 16

    return out


def gps(buf, out):
    """Parse a GPS packet from the flight computer (SystemID 0x05 and TmType 0x02)
    """
    if not (out['systemid'] == 0x05 and out['tmtype'] == 0x02):
        raise ValueError

    index = INDEX_PAYLOAD

    stubs = ['user', 'sip1', 'sip2']

    for stub in stubs:
        out[stub+'_latitude']   = struct.unpack('<f', buf[index      : index + 4 ])[0]
        out[stub+'_longitude']  = struct.unpack('<f', buf[index + 4  : index + 8 ])[0]
        out[stub+'_altitude']   = struct.unpack('<f', buf[index + 8  : index + 12])[0]
        out[stub+'_timeofweek'] = struct.unpack('<f', buf[index + 12 : index + 16])[0] # seconds since Sunday 12:00 AM
        out[stub+'_week']       = struct.unpack('<H', buf[index + 16 : index + 18])[0] # week #1 starts 1980 January 6
        out[stub+'_offset']     = struct.unpack('<f', buf[index + 18 : index + 22])[0] # subtract seconds to obtain UTC

        out[stub+'_pressure_low']  = struct.unpack('<H', buf[index + 22 : index + 24])[0]
        out[stub+'_pressure_mid']  = struct.unpack('<H', buf[index + 24 : index + 26])[0] # use when below 3165
        out[stub+'_pressure_high'] = struct.unpack('<H', buf[index + 26 : index + 28])[0] # use when below 3383

        out[stub+'_status1'] = buf[index + 28]
        out[stub+'_status2'] = buf[index + 29]
        out[stub+'_good']    = buf[index + 30]

        index += 31

    out['gps_source'] = buf[index]
    out['auto_update_time'] = bool(buf[index + 1])
    out['auto_sync'] = bool(buf[index + 1])

    return out


def fc2pcsc(buf, out):
    """Parse a FC->PCSC packet (SystemID 0x03 and TmType 0x02)
    """
    if not (out['systemid'] == 0x03 and out['tmtype'] == 0x02):
        raise ValueError

    index = INDEX_PAYLOAD

    as_floats = np.fromstring(bytes(buf[index + 2:]), np.float32)

    return out


def pcsc2fc(buf, out):
    """Parse a PCSC->FC packet (SystemID 0x03 and TmType 0x03)
    """
    if not (out['systemid'] == 0x03 and out['tmtype'] == 0x03):
        raise ValueError

    index = INDEX_PAYLOAD

    as_floats = np.fromstring(bytes(buf[index:]), np.float32)

    out['v_x'] = as_floats[0]
    out['v_y'] = as_floats[1]
    out['v_sum'] = as_floats[2]

    out['elevation_ae'] = as_floats[3]
    out['elevation_ie'] = as_floats[6]

    out['solar_elevation'] = as_floats[7]

    return out


# Populate the parser registry now that the functions are all defined

# Card-cage packets
for systemid in range(0x80, 0x90):
    parser_registry[systemid][0xF1] = parser_registry[systemid][0xF2] = ge_raw_event
    parser_registry[systemid][0xF3] = ge_event

# Shield-electronics packets
parser_registry[0xB6][0x80] = parser_registry[0xB6][0x82] = bgo_event
parser_registry[0xB6][0x81] = bgo_counter

# GPS packet
parser_registry[0x05][0x02] = gps

# PCSC packet
parser_registry[0x03][0x02] = fc2pcsc
parser_registry[0x03][0x03] = pcsc2fc
