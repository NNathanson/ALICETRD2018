import matplotlib.pyplot as plt
import numpy as np
import glob
import defaults
import adcarray as adc
import o32reader as rdr

#this funciton adds current event to all the previous ones
def addADCEvent(evt, total):
    summed = np.sum(evt, axis=0)
    total_adc = np.sum(summed, axis=0)
    return total + total_adc

#plots the total array
def adc_plot(evt, title=''):
    time = np.linspace(0,30,30)
    plt.plot(time, evt)
    plt.show()

if __name__=='__main__':
    data_path = defaults.DEFAULT_DATA_FOLDER + defaults.CURRENT_FILE

    #Goes through all all .npy files in the interesting data folder, sums them together and plots them
    totalCount = np.zeros(30)

    reader = rdr.o32reader(data_path)
    analyser = adc.adcarray()

    for evno, raw_data in enumerate(reader):
        if evno==0:
            continue
        analyser.analyse_event(raw_data)
        evt = analyser.data[:12]
        evt[defaults.DATA_EXCLUDE_MASK] = 0.0
        totalCount = addADCEvent(evt, totalCount)
        if evno % 100 == 0:
            print("Proccessing events %d - %d" % (evno, evno + 100))

    adc_plot(totalCount)
