"""python3 tbsum.py [FILE...]

Produce a histogram of events summed over time bins.

Extract events from raw data files, sum over the time bins, and save a
histogram in a file called tbsum.png.

Parameters
----------
FILE : str
    list of arbitrary length of paths to raw data files, relative to this
    file's directory.  If this parameter is absent, process all files in
    `defaults.DEFAULT_DATA_FOLDER`.
"""
import defaults
import o32reader as rdr
import adcarray as adc
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys

array_shape = (12, 144, 30)
num = 2000
bins = np.linspace(1, 20000, num)

def sum_events(path):
    """Extract events from a file, and sum the data arrays.
    
    Parameters
    ----------
    path : str
        Path to a raw data file
    """
    reader = rdr.o32reader(path)
    analyser = adc.adcarray()
    inner_hist, bin_edges = np.histogram((), bins)

    for evno, raw_data in enumerate(reader):
        if evno % defaults.PRINT_EVNO_EVERY == 0:
            print("Proccessing events %d–%d"
                    % (evno, evno + defaults.PRINT_EVNO_EVERY))

        if evno == 0:
            # Skip the first event as it may be a configuration event
            # depending on run configurations.
            continue 

        try:
            analyser.analyse_event(raw_data)
        except adc.datafmt_error as e:
            continue
        data = analyser.data[:12] # The last four rows are zeros.
        data[defaults.DATA_EXCLUDE_MASK] = 0.0
        tbsum = np.sum(data, 2)

        foo, bar = np.histogram(tbsum.flatten(), bins)
        inner_hist += foo
    return inner_hist

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        files = glob.glob(defaults.DEFAULT_DATA_FOLDER + '/*')
    else:
        files = sys.argv[1:]
    hist, bin_edges = np.histogram((), bins)
    for f in files:
        hist += sum_events(f)
    print('The maximum in hist is %g.' % (np.amax(hist)))
    plt.bar(bins[:-1], hist, bins[1] - bins[0], log=True)
    plt.xlabel('ADC Value')
    plt.savefig('tbsum.png', bbox_inches='tight')
