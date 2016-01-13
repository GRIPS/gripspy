from __future__ import division, absolute_import, print_function

from sys import stdout

from .parsers import parser
from ..util.checksum import crc16


__all__ = ['packet_generator', 'parser_generator']


def packet_generator(f, verbose=False):
    """Generator that yields valid GRIPS packets from a telemetry-file object.
    Packets that have invalid checksums are skipped past.

    Parameters
    ----------
    verbose : boolean
        If True, output the percentage of the file as a progress indicator
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
            print("No bytes to read, stopping")
            break

        if first_byte == '\x90':
            second_byte = f.read(1)

            if not second_byte:
                print("Not enough bytes remaining for a sync word, stopping")
                break

            # If there is no sync word, move pointer appropriately for next read
            if second_byte != '\xeb':
                if second_byte == '\x90':
                    f.seek(-1, 1)
                continue

            sync_word_position = f.tell() - 2
            remaining_bytes = end_of_file - sync_word_position

            if remaining_bytes < 16:
                print("Not enough bytes remaining for a header, stopping")
                break

            # Extract the payload length from the header
            packet = '\x90\xeb' + f.read(14)
            length = ord(packet[6]) | ord(packet[7]) << 8

            if remaining_bytes >= 16 + length:
                packet = bytearray(packet + f.read(length))

                claimed_checksum = packet[2] | packet[3] << 8
                packet[2:4] = [0, 0]
                if claimed_checksum == crc16(packet):
                    if verbose:
                        if (sync_word_position + length) >= ((percent_completed + 1) / 100.) * end_of_file:
                            percent_completed += 1
                            stdout.write("{0:3d}%\b\b\b\b".format(percent_completed))
                    yield packet
                else:
                    f.seek(sync_word_position + 2)

            else:
                print("Apparent payload length exceeds file length, skipping")
                f.seek(sync_word_position + 2)


def parser_generator(f, filter_systemid=None, filter_tmtype=None, verbose=False):
    pg = packet_generator(f, verbose=verbose)
    for packet in pg:
        out = parser(packet, filter_systemid=filter_systemid, filter_tmtype=filter_tmtype)
        if out is not None:
            yield out
