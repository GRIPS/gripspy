from __future__ import division, absolute_import, print_function

from sys import stdout

from . import parsers


__all__ = ['packet_generator', 'parser_generator']


def packet_generator(f, verbose=False):
    """Generator that yields GRIPS packets from a telemetry-file object.
    
    No packet validation is performed.

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

            # Calculate the number of remaining bytes
            sync_word_position = f.tell() - 2
            f.seek(0, 2)
            end_of_file = f.tell() # computed every time in case the file is growing
            remaining_bytes = end_of_file - sync_word_position

            if remaining_bytes < 16:
                print("Not enough bytes remaining for a header, stopping")
                break

            # Extract the payload length from the header
            f.seek(sync_word_position + 6)
            length = ord(f.read(1)) + (ord(f.read(1)) << 8)

            if remaining_bytes >= length:
                f.seek(sync_word_position)
                packet = f.read(16 + length)
                if verbose:
                    if (sync_word_position + length) >= ((percent_completed + 1) / 100.) * end_of_file:
                        percent_completed += 1
                        stdout.write("{0:3d}%\b\b\b\b".format(percent_completed))

                yield packet
            else:
                print("Apparent payload length exceeds file length, skipping")

            # Skip the sync word, but no further in case it's not a true sync word
            f.seek(sync_word_position + 2)


def parser_generator(f, filter_systemid=None, filter_tmtype=None, verbose=False):
    pg = packet_generator(f, verbose=verbose)
    for packet in pg:
        out = parsers.parser(packet, filter_systemid=filter_systemid, filter_tmtype=filter_tmtype)
        if out is not None:
            yield out
