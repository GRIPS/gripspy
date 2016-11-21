"""
Module with generators for handling telemetry files
"""
from __future__ import division, absolute_import, print_function

from sys import stdout

from .parsers import parser
from ..util.checksum import crc16


__all__ = ['packet_generator', 'parser_generator']


def packet_generator(f, verbose=False, raise_exceptions=False):
    """Generator that yields valid packets from a telemetry-file object.
    Packets that have invalid checksums are skipped past.

    Parameters
    ----------
    f : file object
        e.g., an already open file object
    verbose : boolean
        If True, output the percentage of the file as a progress indicator
    raise_exceptions : boolean
        If True, raise exceptions (e.g., if an invalid checksum is encountered)
    """
    current_location = f.tell()
    f.seek(0, 2)
    end_of_file = f.tell()
    f.seek(current_location)

    if verbose:
        percent_completed = int(100 * current_location  / end_of_file)
        stdout.write("{0:3d}%\b\b\b\b".format(percent_completed))

    while True:
        first_byte = f.read(1)
        if not first_byte:
            if verbose:
                print("No more bytes to read, stopping")
            break

        if first_byte == b'\x90':
            second_byte = f.read(1)

            if not second_byte:
                if verbose:
                    print("Not enough bytes remaining for a sync word, stopping")
                break

            # If there is no sync word, move pointer appropriately for next read
            if second_byte != b'\xeb':
                if second_byte == b'\x90':
                    f.seek(-1, 1)
                continue

            sync_word_position = f.tell() - 2
            remaining_bytes = end_of_file - sync_word_position

            if remaining_bytes < 16:
                if verbose:
                    print("Not enough bytes remaining for a header, stopping")
                break

            # Extract the payload length from the header
            packet = bytearray(b'\x90\xeb' + f.read(14))
            length = packet[6] | packet[7] << 8

            if remaining_bytes >= 16 + length:
                packet = packet + bytearray(f.read(length))

                claimed_checksum = packet[2] | packet[3] << 8
                packet[2:4] = [0, 0]
                calculated_checksum = crc16(packet)
                if claimed_checksum == calculated_checksum:
                    if verbose:
                        if (sync_word_position + length) >= ((percent_completed + 1) / 100.) * end_of_file:
                            percent_completed += 1
                            stdout.write("{0:3d}%\b\b\b\b".format(percent_completed))
                    yield packet
                else:
                    f.seek(sync_word_position + 2)
                    print("Invalid checksum: {0} != {1}".format(claimed_checksum, calculated_checksum))
                    if raise_exceptions:
                        raise RuntimeError("Invalid checksum: {0} != {1}".format(claimed_checksum, calculated_checksum))

            else:
                f.seek(sync_word_position + 2)
                if verbose:
                    print("Apparent payload length exceeds file length, skipping")
                if raise_exceptions:
                    raise RuntimeError("Apparent payload length exceeds file length, skipping")


def parser_generator(f, filter_systemid=None, filter_tmtype=None, verbose=False):
    """Generator that yields parsed packet contents from a telemetry-file object.
    Packets that have invalid checksums are skipped past.

    Parameters
    ----------
    f : file object
        e.g., an already open file object
    filter_systemid : int
        If specified, only yield packets that have a matching SystemId
    filter_tmtype : int
        If specified, only yield packets that have a match TmType
    verbose : boolean
        If True, output the percentage of the file as a progress indicator
    """
    pg = packet_generator(f, verbose=verbose)
    for packet in pg:
        out = parser(packet, filter_systemid=filter_systemid, filter_tmtype=filter_tmtype)
        if out is not None:
            yield out
