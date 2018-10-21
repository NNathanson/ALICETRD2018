import numpy as np

TIME_RESOLUTION = 2e-6 / 30
NUMBER_OF_TIME_BINS = 30
NUMBER_OF_PADS = 144 * 12
ADC_BIT_RESOLUTION = 2**10

INTERESTING_THRESHOLD = 300
FITTING_BASELINE = 100
DEFAULT_PLOT_BASELINE = 13
DEFAULT_EVENT_SHAPE = np.asarray([12, 144, 30])
DEFAULT_DETECTOR_DIMENSIONS = np.asarray([1.0, 1.0, 0.1])
DEFAULT_SPACINGS = DEFAULT_DETECTOR_DIMENSIONS / (DEFAULT_EVENT_SHAPE - 1)
DEFAULT_DATA_FOLDER = 'DATA/'
DEFAULT_INTERESTING_DATA_FOLDER = "DATA_INTERESTING/"
DATA_EXCLUDE_MASK = np.zeros((12, 144, 30), dtype=bool)
# DATA_EXCLUDE_MASK[4:8,0:144,:] = True
PRINT_EVNO_EVERY = 100

CURRENT_FILE = '1910_101_+1400_-1400_31k_0402'
