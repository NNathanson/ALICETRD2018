import matplotlib.pyplot as plt
import numpy as np
import glob
import defaults
import adcarray as adc
import o32reader as rdr
import angular_distribution as ang
import math

#this funciton adds current event to all the previous ones
def addADCEvent(evt, total, beta_x, beta_y,k):
    
    hasData=0
    for i in range(30):
        
        count =0
        xpos = round(beta_x[1]*i+beta_x[0])
        ypos = round(beta_y[1]*i+beta_y[0])
        if (xpos>11):
            continue 
        
        if (ypos>143):
            continue
            
        for j in range(-5,6):
            yCurr = round(ypos +j)
            if(yCurr>143):
               continue
                
                
            count = count + evt[xpos][yCurr][i]
            
        
        if (count != 0):
            hasData=hasData+1
        
        #print(count)
        total[i] = total[i]+ count
        
    #summed = np.sum(evt, axis=0)
    #total_adc = np.sum(summed, axis=0)
    
    #total = np.add(total,total_adc)
    if(hasData >0):
        k=k+1
    return total,k

#plots the total array
def adc_plot(evt, title=''):
    time = np.linspace(0,30,30)
    plt.figure(figsize=(8,6))
    plt.grid()
    plt.plot(time, evt)
    plt.xlim(0,30)
    plt.xlabel('Time(units)', fontsize=12)
    plt.ylabel('ADC value', fontsize=12)
    #plt.title('Pulse height spectrum showing the average ADC value from 1217 events over time')
    #plt.errorbar(x,y,u,0,'k.',label='')
    #plt.legend()
    #plt.savefig('PulseHeight.png')
    plt.show()


"""def readFromFile(data_folder = defaults.DEFAULT_DATA_FOLDER, filename=defaults.CURRENT_FILE, threshold = defaults.DEFAULT_BASELINE, interesting_output_directory = defaults.DEFAULT_INTERESTING_DATA_FOLDER):
    reader = rdr.o32reader(data_folder + filename)
    analyser = adc.adcarray()
    output_dir = interesting_output_directory + filename + '/'
    try:
        os.makedirs(output_dir)
        print('Output directory created.')
    except FileExistsError:
        print('Output directory already exists.')
        1
    for evno, raw_data in enumerate(reader):
        analyser.analyse_event(raw_data)
        data = analyser.data[:12]           #Last four rows are zeros. (Ask Dittel).
        data[defaults.DATA_EXCLUDE_MASK] = 0.0"""   

if __name__=='__main__':
   interesting_data_folder = defaults.DEFAULT_INTERESTING_DATA_FOLDER
   event_files = glob.glob(interesting_data_folder+'*.npy')
   # evt = np.load('763_thresh_300.npy')
   # adc_plot(evt,title='Test')

    #Goes through all all .npy files in the interesting data folder, sums them together and plots them
   totalCount = np.zeros(30)
   
   
   num=0
   count=0
   for file in event_files:
       filename = file.split('/')[-1].split('.')[0]
       #event_num = int(filename.split('_')[0])
       evt = np.load(file)
       #print(evt[5][50])
       
       try:
           beta_x, beta_y, vec_func = ang.linear_fit(evt, threshold=0)
           count = count+1
           #print('beta_x:', beta_x, 'beta_y:', beta_y, 'theta:', angs[0], 'phi:', angs[1])
       except Exception as e:
           print('UNABLE TO LINEAR FIT. Error:', repr(e))

       totalCount,num = addADCEvent(evt,totalCount, beta_x, beta_y,num)
       
       #plot_event(evt, title='Event ' + str(event_num) + '. Max ADC value: ' + str(np.max(evt)))
       
   
   totalCount = totalCount/num
   adc_plot(totalCount)
   print (np.max(totalCount))
   print(num)
