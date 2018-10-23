import matplotlib.pyplot as plt
import numpy as np
import glob
import defaults
import adcarray as adc
import o32reader as rdr

def plot_tbavg_spectrum(data_file_path = defaults.DEFAULT_DATA_FOLDER + defaults.CURRENT_FILE):
    bins = np.arange(defaults.ADC_BIT_RESOLUTION + 1) - 0.5
    bin_centres = np.arange(defaults.ADC_BIT_RESOLUTION)
    tbavg_histogram = np.zeros(bins.shape[0]-1)

    reader = rdr.o32reader(data_file_path)
    analyser = adc.adcarray()

    evno = 0
    for evno, raw_data in enumerate(reader):
        if evno % defaults.PRINT_EVNO_EVERY == 0:
            print("Proccessing events %d - %d" % (evno, evno + defaults.PRINT_EVNO_EVERY))

        if evno==0:
            continue

        try:
            analyser.analyse_event(raw_data)
        except adc.datafmt_error as e:
            print(repr(e))
            continue
        evt = analyser.data[:12]
        evt[defaults.DATA_EXCLUDE_MASK] = 0.0
        hist, bedges = np.histogram(np.mean(evt, axis=2).flatten(), bins)
        tbavg_histogram += hist

    plt.bar(bin_centres, tbavg_histogram)
    plt.yscale('log', basey=10)
    plt.show()
    return tbavg_histogram

if __name__=='__main__':
    plot_tbavg_spectrum()
