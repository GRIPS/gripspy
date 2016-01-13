"""
Adapted from the C implmentation written by Lammert Bies
"""

__all__ = ['crc16']


# Precalculated table for CRC16 calculations
cdef unsigned short crc16_table[256]

cdef int i, j
cdef unsigned short crc, c

for i in range(256):
    crc = 0;
    c   = <unsigned short>i;
    for j in range(8):
        if (crc ^ c) & 0x0001:
            crc = ( crc >> 1 ) ^ 0xA001
        else:
            crc = crc >> 1
        c = c >> 1
    crc16_table[i] = crc


cpdef unsigned short crc16(unsigned char[:] data):
    """Compute the CRC16 checksum of a bytearray or equivalent.
    Because of Cython limitations, the input must be writeable (e.g., not a string)"""
    cdef unsigned short crc = 0xFFFF, tmp
    cdef int i
    for i in range(len(data)):
        tmp = crc ^ (0x00FF & <unsigned short>data[i])
        crc = (crc >> 8) ^ crc16_table[tmp & 0xFF]
    return crc
