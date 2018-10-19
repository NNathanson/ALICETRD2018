from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import glob
import defaults
import matplotlib
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True

def noise(evt):
    x = evt.shape[0]
    y = evt.shape[1]
    z = evt.shape[2]
    arrlen = (x*y*z)
    point_totl = np.ones(arrlen)
    point_excl = np.reshape(np.logical_not(defaults.DATA_EXCLUDE_MASK), arrlen)
    sum_val = sum(np.reshape(evt, arrlen))
    mean = sum_val/np.dot(point_totl,point_excl)
    return mean

if __name__=='__main__':
    interesting_data_folder = defaults.DEFAULT_INTERESTING_DATA_FOLDER + defaults.CURRENT_FILE + '/'
    event_files = glob.glob(interesting_data_folder+'*.npy')
    NOISE = []
    for file in event_files:
        filename = file.split('/')[-1].split('.')[0]
        event_num = int(filename.split('_')[0])
        evt = np.load(file)
        NOISE.append(noise(evt))
NOISE = np.array(NOISE)
plt.rcParams["patch.force_edgecolor"] = True
plt.figure(figsize=(8,6))
#plt.grid()
plt.xlabel('Mean ADC counts')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.hist(NOISE,bins='fd')
plt.legend()
#plt.savefig(plotdir+title+'.png')
plt.show()
