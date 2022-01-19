#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Write .fcs files for flow cytometry"""
from __future__ import print_function, unicode_literals, division

import pathlib
import struct
import warnings

import numpy as np


def write_fcs(filename, data, chn_names, chn_short_names=None,
              split=None,
              endianness='big',
              compat_chn_names=True,
              compat_copy=True,
              compat_negative=True,
              compat_percent=True,
              compat_max_int16=10000):
    """Write numpy data to an .fcs file (FCS3.0 file format)
    
    Parameters
    ----------
    filename: str or pathlib.Path
        Path to the output .fcs file
    ch_names: list of str, length C
        Names of the output channels
    data: 2d ndarray of shape (N,C)
        The numpy array data to store as .fcs file format.
    endianness: str
        Set to "little" or "big" to define the byte order used.
    compat_chn_names: bool
        Compatibility mode for 3rd party flow analysis software:
        The characters " ", "?", and "_" are removed in the output
        channel names.
    compat_copy: bool
        Do not override the input array `data` when modified in
        compatibility mode.
    compat_negative: bool
        Compatibliity mode for 3rd party flow analysis software:
        Flip the sign of `data` if its mean is smaller than zero.
    compat_percent: bool
        Compatibliity mode for 3rd party flow analysis software:
        If a column in `data` contains values only between 0 and 1,
        they are multiplied by 100.
    compat_max_int16: int
        Compatibliity mode for 3rd party flow analysis software:
        If a column in `data` has a maximum above this value,
        then the display-maximum is set to 2**15.

    Notes
    -----
    - These commonly used unicode characters are replaced: "µ", "²"
    - If the input data contain NaN values, the corresponding rows
      are excluded due to incompatibility with the FCS file format.

    """
    if chn_short_names is None and split:
      chn_short_names = [c.split(split)[0] for c in chn_names]
      chn_names = [c.split(split)[-1] for c in chn_names]
    
    filename = pathlib.Path(filename)
    if not isinstance(data, np.ndarray):
      data = np.array(data, dtype=float)
    # remove rows with nan values
    nanrows = np.isnan(data).any(axis=1)
    if np.sum(nanrows):
        msg = "Rows containing NaNs are not written to {}!".format(filename)
        warnings.warn(msg)
        data = data[~nanrows]
    if endianness not in ['little', 'big']:
        raise ValueError("`endianness` must be 'little' or 'big'!")

    msg = "length of `chn_names` must match length of 2nd axis of `data`"
    assert len(chn_names) == data.shape[1], msg

    rpl = [['µ', 'u'],
           ['²', '2'],
           ]

    if compat_chn_names:
        # Compatibility mode: Clean up headers.
        rpl += [[' ', ''],
                ['?', ''],
                ['_', ''],
                ]

    for ii in range(len(chn_names)):
        for (a, b) in rpl:
            chn_names[ii] = chn_names[ii].replace(a, b)
            
    # DATA segment
    data1 = data.flatten().tolist()
    DATA = struct.pack('>%sf' % len(data1), *data1)

    # TEXT segment
    header_size = 256

    if endianness == "little":
        # use little endian
        byteord = '1,2,3,4'
    else:
        # use big endian
        byteord = '4,3,2,1'
    TEXT = '#$BEGINANALYSIS#0#$ENDANALYSIS#0'
    TEXT += '#$BEGINSTEXT#0#$ENDSTEXT#0'
    # Add placeholders for $BEGINDATA and $ENDDATA, because we don't
    # know yet how long TEXT is.
    TEXT += '#$BEGINDATA#{data_start_byte}#$ENDDATA#{data_end_byte}'
    TEXT += '#$BYTEORD#{0}#$DATATYPE#F'.format(byteord)
    TEXT += '#$MODE#L#$DATATYPE#F#$NEXTDATA#0#$TOT#{0}'.format(data.shape[0])
    TEXT += '#$PAR#{0}'.format(data.shape[1])
    TEXT += '#$FIL#{0}#'.format(filename)
    
    # Check for content of data columns and set range
    for jj in range(data.shape[1]):
      fmt_str = '$P{0}S#{1}#$P{0}N#{2}#$P{0}B#32#$P{0}R#65536#$P{0}E#0,0#'
      TEXT += fmt_str.format(jj+1, chn_names[jj], chn_short_names[jj] if chn_short_names else chn_names[jj])
        
    TEXT += '/'

    # SET $BEGINDATA and $ENDDATA using the current size of TEXT plus padding.
    text_padding = 47  # for visual separation and safety
    data_start_byte = header_size + len(TEXT) + text_padding
    data_end_byte = data_start_byte + len(DATA) - 1
    TEXT = TEXT.format(data_start_byte=data_start_byte,
                       data_end_byte=data_end_byte)
    lentxt = len(TEXT)
    # Pad TEXT segment with spaces until data_start_byte
    TEXT = TEXT.ljust(data_start_byte - header_size, ' ')

    # HEADER segment
    ver = 'FCS3.0'

    textfirst = '{0: >8}'.format(header_size)
    textlast = '{0: >8}'.format(lentxt + header_size - 1)

    # Starting with FCS 3.0, data segment can end beyond byte 99,999,999,
    # in which case a zero is written in each of the two header fields (the
    # values are given in the text segment keywords $BEGINDATA and $ENDDATA)
    if data_end_byte <= 99999999:
        datafirst = '{0: >8}'.format(data_start_byte)
        datalast = '{0: >8}'.format(data_end_byte)
    else:
        datafirst = '{0: >8}'.format(0)
        datalast = '{0: >8}'.format(0)

    anafirst = '{0: >8}'.format(0)
    analast = '{0: >8}'.format(0)

    HEADER = '{0: <256}'.format(ver + '    '
                                + textfirst
                                + textlast
                                + datafirst
                                + datalast
                                + anafirst
                                + analast)

    # Write data
    with filename.open('wb') as fd:
        fd.write(HEADER.encode('ascii', 'replace'))
        fd.write(TEXT.encode('ascii', 'replace'))
        fd.write(DATA)
        fd.write(b'00000000')
