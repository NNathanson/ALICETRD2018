import matplotlib.pyplot as plt
import numpy as np
import glob
import defaults
import adcarray as adc

#this funciton adds current event to all the previous ones
def addADCEvent(evt, total):
    summed = np.sum(evt, axis=0)
    total_adc = np.sum(summed, axis=0)
    total = np.add(average,total_adc)

#plots the total array
def adc_plot(evt, title=''):
    time = np.linspace(0,30,30)
    plt.plot(time, evt)
    plt.xlabel('Time')
    plt.show()




if __name__=='__main__':
    interesting_data_folder = defaults.DEFAULT_INTERESTING_DATA_FOLDER + defaults.CURRENT_FILE + '/'
    event_files = glob.glob(interesting_data_folder+'*.npy')
    #evt = np.load('617_thresh_100.npy')
    #adc_plot(evt,title='Test')

    #Goes through all all .npy files in the interesting data folder, sums them together and plots them
    totalCount = np.zeroes(30)
    for file in event_files:
        filename = file.split('/')[-1].split('.')[0]
        event_num = int(filename.split('_')[0])
        evt = np.load(file)
        addADCEvent(evt,totalCount)
        adc_plot(evt, title='Event ' + str(event_num) + '. Max ADC value: ' + str(np.max(evt)))

    adc_plot(totalCount)
