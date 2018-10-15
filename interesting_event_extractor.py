#!/usr/bin/env python3
#450 spike at event 617
#703 spike at event 1858
import adcarray as adc
import pylab as pl
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os
import defaults
import glob
import o32reader as rdr

def extract_interesting_events(data_folder = defaults.DEFAULT_DATA_FOLDER, filename=defaults.CURRENT_FILE, threshold = defaults.DEFAULT_BASELINE, interesting_output_directory = defaults.DEFAULT_INTERESTING_DATA_FOLDER):
    print_evno_every = 100

    reader = rdr.o32reader(data_folder + filename)
    analyser = adc.adcarray()

    output_dir = interesting_output_directory + filename + '/'

    try:
        os.makedirs(output_dir)
        print('Output directory created.')
    except FileExistsError:
        print('Output directory already exists.')


    def data_is_interesting(data):
        return np.max(data) > threshold

    maxevno = 0
    minevno = 0
    absmax = 0
    absmin = 2**10
    for evno, raw_data in enumerate(reader):
        if evno % print_evno_every == 0:
            print("Proccessing events %d - %d" % (evno, evno + print_evno_every))

        if evno == 0:
            continue        #Skip first event (may be a configuration event depending on run configurations).

        analyser.analyse_event(raw_data)
        data = analyser.data[:12]           #Last four rows are zeros. (Ask Dittel).
        data[defaults.DATA_EXCLUDE_MASK] = 0.0

        maxval = np.max(data)
        minval = np.min(data)

        if maxval > absmax:
            absmax = maxval
            maxevno = evno

        if minval < absmin:
            absmin = minval
            minevno = evno

        if data_is_interesting(data):
            output_file = output_dir + str(evno) + '_thresh_' + str(threshold) + '.npy'
            print("Found interesting data. Saving to", output_file, ' max value ', np.max(data))
            np.save(output_file, data)

    print("Max value", absmax, 'obtained in event', maxevno)
    print("Min value", absmin, 'obtained in event', minevno)

if __name__=='__main__':
    extract_interesting_events()

    # fils = glob.glob(defaults.DEFAULT_DATA_FOLDER+'/*')
    # for fil in fils:
    #     extract_interesting_events(filename=fil.split('/')[1])